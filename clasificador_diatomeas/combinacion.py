"""Combinación de las probabilidades de DINOv2 y ResNet50 para cada recorte.

ResNet tiene 78 clases: las 77 de DINOv2 (dos con erratas en el nombre) y ``Planothidium_fp``, que DINOv2
no tiene y se descarta sin renormalizar.
"""

from __future__ import annotations

import numpy as np

from .errores import ErrorModelos

ALIAS_RESNET = {"Denticula_tenius": "Denticula_tenuis", "Nitzschia_dessertorum": "Nitzschia_desertorum"}


def indices_resnet(clases_resnet: list[str], clases_dino: list[str]) -> list[int]:
    """Posición en la salida de ResNet de cada clase de DINOv2, en el orden de DINOv2."""
    corregidas = [ALIAS_RESNET.get(clase, clase) for clase in clases_resnet]
    faltan = [clase for clase in clases_dino if clase not in corregidas]
    if faltan:
        raise ErrorModelos(f"Clases de DINOv2 que no están en ResNet: {faltan}")
    return [corregidas.index(clase) for clase in clases_dino]


def combinar(p_dino: np.ndarray, p_resnet: np.ndarray | None, indices: list[int], peso_dino: float) -> np.ndarray:
    """``peso_dino · p_dino + (1 − peso_dino) · p_resnet``. Con ``peso_dino = 1`` es DINOv2 solo."""
    if p_resnet is None or peso_dino >= 1:
        return p_dino
    return peso_dino * p_dino + (1 - peso_dino) * p_resnet[:, indices]


def top3(probabilidades: np.ndarray, clases: list[str]) -> list[list[tuple[str, float]]]:
    """Las tres clases más probables de cada fila, de más a menos probable."""
    orden = np.argsort(-probabilidades, axis=1, kind="stable")[:, :3]
    return [[(clases[j], float(fila[j])) for j in indices] for fila, indices in zip(probabilidades, orden)]
