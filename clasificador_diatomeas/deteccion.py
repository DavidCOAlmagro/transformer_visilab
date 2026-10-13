"""Detección de diatomeas con YOLO."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from .errores import ErrorModelos


def normalizar_caja(caja: list[float], ancho: int, alto: int) -> tuple[int, int, int, int] | None:
    """Caja entera dentro de la imagen, o None si no es válida (igual que ``normalize_bbox`` de la referencia)."""
    if len(caja) != 4 or not all(math.isfinite(float(v)) for v in caja):
        return None
    x1, y1, x2, y2 = (float(v) for v in caja)
    if x2 <= x1 or y2 <= y1 or ancho <= 0 or alto <= 0:
        return None
    x1, y1 = max(0, min(ancho, math.floor(x1))), max(0, min(alto, math.floor(y1)))
    x2, y2 = max(0, min(ancho, math.ceil(x2))), max(0, min(alto, math.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


class Detector:
    def __init__(self, pesos: Path, confianza: float, tamano: int) -> None:
        if not pesos.is_file():
            raise ErrorModelos(f"No existe {pesos}")
        from ultralytics import YOLO

        self.modelo = YOLO(str(pesos))
        self.confianza = confianza
        self.tamano = tamano

    def detectar(self, imagen: Image.Image) -> list[tuple[int, tuple[int, int, int, int], str]]:
        """``(número, caja, clase de YOLO)`` de cada detección. El número es el orden de YOLO empezando en 1."""
        resultado = self.modelo.predict(
            source=imagen, conf=self.confianza, imgsz=self.tamano, device="cpu",
            save=False, save_txt=False, verbose=False,
        )[0]
        cajas = resultado.boxes
        if cajas is None or len(cajas) == 0:
            return []
        ancho, alto = imagen.size
        detecciones = []
        for numero, (caja, clase) in enumerate(zip(cajas.xyxy.tolist(), cajas.cls.tolist()), start=1):
            normalizada = normalizar_caja(caja, ancho, alto)
            if normalizada is not None:
                detecciones.append((numero, normalizada, resultado.names[int(clase)]))
        return detecciones
