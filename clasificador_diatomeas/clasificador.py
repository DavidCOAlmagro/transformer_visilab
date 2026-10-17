"""
--------------------------------------
Clasificador de diatomeas.

    clf = Clasificador("modelos")              # carga los modelos (una vez)
    resultado = clf.clasificar("foto.tif")     # diatomeas de la foto y su especie
    imagen = clf.anotar("foto.tif", resultado)  # la foto con los recuadros
    clf.guardar_excel(resultados, "resultados.xlsx")

Por dentro: YOLO busca las diatomeas de la foto, cada una se recorta y
DINOv2 da la probabilidad de cada especie.
--------------------------------------
"""
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import dinov2, excel, fotos

VERSION = "1.0 (modelo 77_campo)"
CONFIANZA_YOLO = 0.30   # YOLO solo devuelve las cajas con al menos esta confianza
TAMANO_YOLO = 1024      # tamaño al que YOLO redimensiona la foto
LOTE = 32               # recortes que se clasifican a la vez


@dataclass
class Recorte:
    """Una diatomea de la foto."""
    numero: int                        # orden en que la encontró YOLO (1, 2, ...)
    caja: tuple[int, int, int, int]    # x1, y1, x2, y2 en píxeles
    especie: str                       # la más probable, por ejemplo "Navicula gregaria"
    tipo: str                          # "especie", "vista pleural" o "resto"
    top3: list[tuple[str, float]]      # las 3 más probables: (especie, porcentaje)


@dataclass
class Resultado:
    """Lo que sale de una foto. estado: "ok", "sin detecciones" o "error"."""
    imagen: str
    estado: str
    recortes: list[Recorte] = field(default_factory=list)
    mensaje: str = ""


def cargar_yolo(ruta: Path):
    # Que ultralytics no se conecte a internet ni intente instalar paquetes
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO

    if getattr(sys, "frozen", False):
        # Dentro de un .exe de PyInstaller, ultralytics volvería a abrir la app para
        # ejecutar "pip show" si la carpeta está en un repositorio git. Se desactiva.
        from ultralytics.utils import checks
        checks.check_multiple_install = lambda: None

    if not ruta.is_file():
        raise FileNotFoundError(f"No existe {ruta}")
    return YOLO(str(ruta))


class Clasificador:

    def __init__(self, carpeta_modelos):
        carpeta = Path(carpeta_modelos)
        # Las especies, en el mismo orden que al entrenar
        self.clases = (carpeta / "clases.txt").read_text(encoding="utf-8").split()
        self.yolo = cargar_yolo(carpeta / "yolo.pt")
        self.modelo = dinov2.cargar_modelo(carpeta / "dinov2.pth", len(self.clases))

    def buscar_diatomeas(self, imagen) -> list[tuple[int, tuple]]:
        """Cajas de las diatomeas que encuentra YOLO, como (número, (x1, y1, x2, y2))."""
        prediccion = self.yolo.predict(source=imagen, conf=CONFIANZA_YOLO, imgsz=TAMANO_YOLO, device="cpu",
                                       save=False, verbose=False)[0]
        if prediccion.boxes is None:
            return []
        ancho, alto = imagen.size
        cajas = []
        for numero, (x1, y1, x2, y2) in enumerate(prediccion.boxes.xyxy.tolist(), start=1):
            # A píxeles enteros y sin salirse de la foto
            x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
            x2, y2 = min(ancho, math.ceil(x2)), min(alto, math.ceil(y2))
            if x2 > x1 and y2 > y1:
                cajas.append((numero, (x1, y1, x2, y2)))
        return cajas

    def probabilidades(self, recortes) -> torch.Tensor:
        """Probabilidad de cada especie para cada recorte (una fila por recorte)."""
        partes = []
        with torch.inference_mode():
            for inicio in range(0, len(recortes), LOTE):
                partes.append(self.modelo(dinov2.preparar(recortes[inicio:inicio + LOTE])))
        return torch.cat(partes)

    def clasificar(self, foto) -> Resultado:
        """
        Clasifica una foto (ruta o imagen de PIL). Si no se puede abrir, devuelve
        el resultado con estado "error" y el motivo en mensaje.
        """
        nombre = Path(foto).name if isinstance(foto, (str, Path)) else "imagen"
        try:
            imagen = fotos.abrir(foto)
        except (OSError, ValueError):
            return Resultado(nombre, "error", mensaje=f"No se puede abrir {nombre}: no es una imagen o está dañada")

        cajas = self.buscar_diatomeas(imagen)
        if not cajas:
            return Resultado(nombre, "sin detecciones")

        recortes = [imagen.crop(caja) for _, caja in cajas]
        probabilidades = self.probabilidades(recortes)

        resultado = Resultado(nombre, "ok")
        for (numero, caja), fila in zip(cajas, probabilidades):
            mejores = torch.argsort(fila, descending=True, stable=True)[:3]
            top3 = [(excel.nombre_legible(self.clases[i]), round(100 * float(fila[i]), 1)) for i in mejores]
            clase = self.clases[mejores[0]]
            resultado.recortes.append(Recorte(numero, caja, excel.nombre_legible(clase), excel.tipo(clase), top3))
        return resultado

    def anotar(self, foto, resultado: Resultado):
        """La foto con un recuadro verde en cada diatomea y el nombre de su especie."""
        return fotos.anotar(fotos.abrir(foto), resultado.recortes)

    def guardar_excel(self, resultados: list[Resultado], ruta):
        """Excel con todos los resultados (hojas Recortes, Imagenes e Info)."""
        excel.guardar(resultados, ruta, VERSION)
