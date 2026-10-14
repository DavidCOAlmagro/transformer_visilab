"""Los dos clasificadores de recortes, en CPU y fp32, sin conexión a internet.

La arquitectura y el preprocesado son los mismos que en ``Inferir/infer_and_split_resnet_single_folder.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .dinov2 import Dinov2, Preprocesado
from .errores import ErrorModelos
from .preprocesado import preparar_para_dinov2

LOTE = 32  # recortes por pasada; acota la memoria en fotos con muchas diatomeas
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIACION_IMAGENET = (0.229, 0.224, 0.225)


def cargar_pesos(ruta: Path) -> dict[str, torch.Tensor]:
    """state_dict de un ``.pth`` (acepta los envoltorios habituales y quita el prefijo ``module.``)."""
    if not ruta.is_file():
        raise ErrorModelos(f"No existe {ruta}")
    try:
        datos = torch.load(ruta, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError) as error:
        raise ErrorModelos(f"No se puede cargar {ruta}: {error}") from error
    if isinstance(datos, dict):
        for clave in ("state_dict", "model_state_dict", "model"):
            if isinstance(datos.get(clave), dict):
                datos = datos[clave]
                break
    return {k.removeprefix("module."): v for k, v in datos.items() if isinstance(v, torch.Tensor)}


def con_prefijo(estado: dict[str, torch.Tensor], prefijo: str) -> dict[str, torch.Tensor]:
    return {k.removeprefix(prefijo): v for k, v in estado.items() if k.startswith(prefijo)}


def _por_lotes(recortes: list[Image.Image], funcion, n_clases: int) -> np.ndarray:
    partes = [funcion(recortes[i:i + LOTE]) for i in range(0, len(recortes), LOTE)]
    return np.concatenate(partes) if partes else np.zeros((0, n_clases), dtype=np.float32)


class ModeloDino:
    """DINOv2-base ajustado + MLP. La configuración de DINOv2 va en la carpeta de modelos: no se descarga nada."""

    def __init__(self, pesos: Path, carpeta_config: Path, n_clases: int) -> None:
        estado = cargar_pesos(pesos)
        try:
            self.backbone = Dinov2(carpeta_config / "config.json")
            self.backbone.load_state_dict(con_prefijo(estado, "backbone."), strict=True)
            self.tronco = torch.nn.Sequential(
                torch.nn.Linear(self.backbone.hidden_size, 512), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            )
            self.cabeza = torch.nn.Linear(256, n_clases)
            self.tronco.load_state_dict(con_prefijo(estado, "tronco."), strict=True)
            self.cabeza.load_state_dict(con_prefijo(estado, "cabeza_especie."), strict=True)
        except (OSError, RuntimeError, ValueError) as error:
            raise ErrorModelos(f"Los pesos de DINOv2 ({pesos.name}) no encajan con el modelo: {error}") from error
        self.procesador = Preprocesado(carpeta_config / "preprocessor_config.json")
        self.n_clases = n_clases
        for modulo in (self.backbone, self.tronco, self.cabeza):
            modulo.eval()

    @torch.inference_mode()
    def _lote(self, recortes: list[Image.Image]) -> np.ndarray:
        cls = self.backbone(self.procesador([preparar_para_dinov2(r) for r in recortes]))
        logits = self.cabeza(self.tronco(cls / cls.norm(dim=1, keepdim=True).clamp_min(1e-12)))
        return torch.softmax(logits.float(), dim=1).numpy()

    def probabilidades(self, recortes: list[Image.Image]) -> np.ndarray:
        return _por_lotes(recortes, self._lote, self.n_clases)


class ModeloResNet:
    """ResNet50 con su cabeza de 78 clases. Los recortes se redimensionan a 256×256 como en la referencia."""

    def __init__(self, pesos: Path, n_clases: int) -> None:
        from torchvision import models, transforms

        self.modelo = models.resnet50(weights=None)
        self.modelo.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5), torch.nn.Linear(self.modelo.fc.in_features, 256), torch.nn.ReLU(),
            torch.nn.Dropout(0.3), torch.nn.Linear(256, n_clases),
        )
        try:
            self.modelo.load_state_dict(cargar_pesos(pesos), strict=True)
        except RuntimeError as error:
            raise ErrorModelos(f"Los pesos de ResNet ({pesos.name}) no encajan con ResNet50 de {n_clases} clases") from error
        self.modelo.eval()
        self.transformar = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(MEDIA_IMAGENET, DESVIACION_IMAGENET),
        ])
        self.n_clases = n_clases

    @torch.inference_mode()
    def _lote(self, recortes: list[Image.Image]) -> np.ndarray:
        lote = torch.stack([self.transformar(r.convert("RGB")) for r in recortes])
        return torch.softmax(self.modelo(lote).float(), dim=1).numpy()

    def probabilidades(self, recortes: list[Image.Image]) -> np.ndarray:
        return _por_lotes(recortes, self._lote, self.n_clases)
