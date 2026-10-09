"""
--------------------------------------
Utilidades de datos: recorrer las carpetas de imágenes, normalizar los nombres
de especie, géneros y semilla de reproducibilidad.
--------------------------------------
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from constantes import VARIABLES_GLOBALES


def normalizar_nombre_especie(nombre_carpeta: str) -> str:
    """
    Convierte el nombre de una carpeta en el nombre de especie canónico:
    espacios y puntos pasan a '_' (p. ej. 'Fistulifera saprophila' ->
    'Fistulifera_saprophila'). Sin esto, 387 imágenes de Fistulifera
    quedaban fuera de los splits (spec 009).
    """
    return "_".join(nombre_carpeta.replace(".", " ").split())


def rutas_imagenes(especies: set[str] | None = None) -> list[tuple[Path, str]]:
    """
    Recorre ``data/imagenes_visilab(raw)/<grupo>/<especie>/`` y devuelve
    (ruta, especie normalizada). Si ``especies`` es None, devuelve todas.
    """
    raiz = VARIABLES_GLOBALES["RUTA_BASE"] / "imagenes_visilab(raw)"
    if not raiz.is_dir():
        raise FileNotFoundError(f"No se encontró la carpeta de imágenes: {raiz}")
    grupos = sorted(p for p in raiz.iterdir() if p.is_dir())
    if not grupos:
        raise FileNotFoundError(f"No se encontraron subcarpetas de grupo en: {raiz}")

    imagenes: list[tuple[Path, str]] = []
    for grupo in grupos:
        print(f"Recorriendo {grupo}...")
        for carpeta in sorted(p for p in grupo.iterdir() if p.is_dir()):
            especie = normalizar_nombre_especie(carpeta.name)
            if especies is not None and especie not in especies:
                continue
            if especie != carpeta.name:
                print(f"Aviso: carpeta '{carpeta.name}' tratada como '{especie}'.")
            imagenes.extend(
                (archivo, especie) for archivo in sorted(carpeta.iterdir())
                if archivo.suffix.lower() in VARIABLES_GLOBALES["EXTENSIONES_VALIDAS"]
            )
    return imagenes


def obtener_genero(especie: str) -> str:
    """Extrae el género de una especie: la primera palabra antes del '_'."""
    return especie.split("_")[0]


def construir_numero_genero(especies: set[str]) -> dict[str, int]:
    """Mapea cada género presente en las especies a un índice numérico."""
    return {genero: i for i, genero in enumerate(sorted({obtener_genero(e) for e in especies}))}


def fijar_semilla(semilla: int) -> None:
    """Fija las semillas para que los resultados sean reproducibles."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
