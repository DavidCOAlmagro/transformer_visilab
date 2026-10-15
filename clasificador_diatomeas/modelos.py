"""
--------------------------------------
Los modelos: YOLO busca las diatomeas en la foto y DINOv2 (y, si se
activa, también ResNet50) dice de qué especie es cada recorte.

Todo va en CPU y sin internet. La arquitectura y el preprocesado son los
mismos que en Inferir/infer_and_split_resnet_single_folder.py, con el que
se midieron los resultados.
--------------------------------------
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from .dinov2 import Dinov2, Preprocesado

# Recortes por pasada: si una foto tiene muchas diatomeas no se dispara la memoria
LOTE = 32

# ResNet tiene 78 clases: las 77 de DINOv2 (dos con el nombre mal escrito) más Planothidium_fp,
# que DINOv2 no tiene y se descarta
ALIAS_RESNET = {"Denticula_tenius": "Denticula_tenuis", "Nitzschia_dessertorum": "Nitzschia_desertorum"}


class ErrorModelos(Exception):
    """Falta un fichero en la carpeta de modelos o no es válido."""


def cargar_pesos(ruta: Path) -> dict[str, torch.Tensor]:
    """
    Lee un .pth y devuelve sus pesos. Acepta los formatos habituales
    (con o sin "state_dict" dentro y con el prefijo "module." de DataParallel).
    """
    if not ruta.is_file():
        raise ErrorModelos(f"No existe {ruta}")
    try:
        datos = torch.load(ruta, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError) as error:
        raise ErrorModelos(f"No se puede cargar {ruta} (¿copia incompleta?): {error}") from error

    if isinstance(datos, dict):
        for clave in ("state_dict", "model_state_dict", "model"):
            if isinstance(datos.get(clave), dict):
                datos = datos[clave]
                break

    pesos = {}
    for nombre, valor in datos.items():
        if isinstance(valor, torch.Tensor):
            pesos[nombre.removeprefix("module.")] = valor
    return pesos


def con_prefijo(pesos: dict[str, torch.Tensor], prefijo: str) -> dict[str, torch.Tensor]:
    """Solo los pesos que empiezan por el prefijo, sin el prefijo."""
    return {nombre.removeprefix(prefijo): valor for nombre, valor in pesos.items() if nombre.startswith(prefijo)}


def normalizar_caja(caja: list[float], ancho: int, alto: int) -> tuple[int, int, int, int] | None:
    """
    Pasa la caja de YOLO a píxeles enteros dentro de la foto. Devuelve None si
    no es válida. Es igual que normalize_bbox del script de investigación.
    """
    if len(caja) != 4 or not all(math.isfinite(float(valor)) for valor in caja):
        return None
    x1, y1, x2, y2 = (float(valor) for valor in caja)
    if x2 <= x1 or y2 <= y1 or ancho <= 0 or alto <= 0:
        return None
    x1 = max(0, min(ancho, math.floor(x1)))
    y1 = max(0, min(alto, math.floor(y1)))
    x2 = max(0, min(ancho, math.ceil(x2)))
    y2 = max(0, min(alto, math.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


class Detector:
    """YOLO: encuentra las diatomeas de la foto."""

    def __init__(self, pesos: Path, confianza: float, tamano: int):
        if not pesos.is_file():
            raise ErrorModelos(f"No existe {pesos}")

        # Que ultralytics no se conecte a internet ni intente instalar paquetes con pip
        os.environ.setdefault("YOLO_OFFLINE", "true")
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        from ultralytics import YOLO

        if getattr(sys, "frozen", False):
            # Dentro de un .exe (PyInstaller) sys.executable es la propia app, y ultralytics la
            # volvería a abrir para hacer "python -m pip show ultralytics" si está dentro de un repo git
            from ultralytics.utils import checks
            checks.check_multiple_install = lambda: None

        self.modelo = YOLO(str(pesos))
        self.confianza = confianza
        self.tamano = tamano

    def detectar(self, imagen: Image.Image) -> list[tuple[int, tuple[int, int, int, int], str]]:
        """
        Devuelve (número, caja, clase de YOLO) de cada diatomea. El número es el
        orden en que la da YOLO, empezando en 1.
        """
        resultado = self.modelo.predict(source=imagen, conf=self.confianza, imgsz=self.tamano, device="cpu",
                                        save=False, save_txt=False, verbose=False)[0]
        cajas = resultado.boxes
        if cajas is None or len(cajas) == 0:
            return []

        ancho, alto = imagen.size
        detecciones = []
        numero = 0
        for caja, clase in zip(cajas.xyxy.tolist(), cajas.cls.tolist()):
            numero += 1
            caja_normalizada = normalizar_caja(caja, ancho, alto)
            if caja_normalizada is not None:
                detecciones.append((numero, caja_normalizada, resultado.names[int(clase)]))
        return detecciones


class ModeloDino:
    """DINOv2 ajustado con las diatomeas + la red pequeña (MLP) que da la especie."""

    def __init__(self, pesos: Path, carpeta_config: Path, n_clases: int):
        todos = cargar_pesos(pesos)
        try:
            self.backbone = Dinov2(carpeta_config / "config.json")
            self.backbone.load_state_dict(con_prefijo(todos, "backbone."), strict=True)
            # Misma MLP que en el entrenamiento (src/clasificador.py): 768 -> 512 -> 256 -> especies
            self.tronco = nn.Sequential(
                nn.Linear(self.backbone.hidden_size, 512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            )
            self.cabeza = nn.Linear(256, n_clases)
            self.tronco.load_state_dict(con_prefijo(todos, "tronco."), strict=True)
            self.cabeza.load_state_dict(con_prefijo(todos, "cabeza_especie."), strict=True)
        except (OSError, RuntimeError, ValueError) as error:
            raise ErrorModelos(f"Los pesos de DINOv2 ({pesos.name}) no encajan con el modelo: {error}") from error

        self.procesador = Preprocesado(carpeta_config / "preprocessor_config.json")
        self.n_clases = n_clases
        self.backbone.eval()
        self.tronco.eval()
        self.cabeza.eval()

    @torch.inference_mode()
    def probabilidades(self, recortes: list[Image.Image]) -> np.ndarray:
        """Probabilidad de cada especie para cada recorte (una fila por recorte)."""
        partes = []
        for inicio in range(0, len(recortes), LOTE):
            pixeles = self.procesador(recortes[inicio:inicio + LOTE])
            cls = self.backbone(pixeles)
            # El vector CLS se normaliza (longitud 1) antes de la MLP, igual que al entrenar
            cls = cls / cls.norm(dim=1, keepdim=True).clamp_min(1e-12)
            logits = self.cabeza(self.tronco(cls))
            partes.append(torch.softmax(logits.float(), dim=1).numpy())
        if not partes:
            return np.zeros((0, self.n_clases), dtype=np.float32)
        return np.concatenate(partes)


class ModeloResNet:
    """ResNet50 del grupo con su cabeza de 78 clases. Los recortes se pasan a 256x256 como en la referencia."""

    def __init__(self, pesos: Path, n_clases: int):
        self.modelo = models.resnet50(weights=None)
        self.modelo.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(self.modelo.fc.in_features, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, n_clases),
        )
        try:
            self.modelo.load_state_dict(cargar_pesos(pesos), strict=True)
        except RuntimeError as error:
            raise ErrorModelos(f"Los pesos de ResNet ({pesos.name}) no encajan con ResNet50 de {n_clases} clases") from error
        self.modelo.eval()
        self.transformar = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.n_clases = n_clases

    @torch.inference_mode()
    def probabilidades(self, recortes: list[Image.Image]) -> np.ndarray:
        partes = []
        for inicio in range(0, len(recortes), LOTE):
            lote = torch.stack([self.transformar(recorte.convert("RGB")) for recorte in recortes[inicio:inicio + LOTE]])
            partes.append(torch.softmax(self.modelo(lote).float(), dim=1).numpy())
        if not partes:
            return np.zeros((0, self.n_clases), dtype=np.float32)
        return np.concatenate(partes)


def indices_resnet(clases_resnet: list[str], clases_dino: list[str]) -> list[int]:
    """Para cada clase de DINOv2 (en su orden), en qué posición está en la salida de ResNet."""
    corregidas = [ALIAS_RESNET.get(clase, clase) for clase in clases_resnet]
    faltan = [clase for clase in clases_dino if clase not in corregidas]
    if faltan:
        raise ErrorModelos(f"Clases de DINOv2 que no están en ResNet: {faltan}")
    return [corregidas.index(clase) for clase in clases_dino]


def combinar(p_dino: np.ndarray, p_resnet: np.ndarray | None, indices: list[int], peso_dino: float) -> np.ndarray:
    """
    Mezcla las probabilidades de los dos modelos: peso_dino * DINOv2 + (1 - peso_dino) * ResNet.
    Con peso_dino = 1 es DINOv2 solo. Planothidium_fp (solo en ResNet) se queda fuera sin renormalizar.
    """
    if p_resnet is None or peso_dino >= 1:
        return p_dino
    return peso_dino * p_dino + (1 - peso_dino) * p_resnet[:, indices]


def top3(probabilidades: np.ndarray, clases: list[str]) -> list[list[tuple[str, float]]]:
    """Las tres clases más probables de cada recorte, de más a menos probable."""
    orden = np.argsort(-probabilidades, axis=1, kind="stable")[:, :3]
    resultado = []
    for fila, indices in zip(probabilidades, orden):
        resultado.append([(clases[i], float(fila[i])) for i in indices])
    return resultado
