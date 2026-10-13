"""Lectura de fotos y búsqueda en carpetas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .errores import ErrorEntrada

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MODOS_16_BITS = {"I;16", "I;16B", "I;16L", "I;16N", "I", "F"}


def abrir_imagen(foto: str | Path | Image.Image) -> Image.Image:
    """Devuelve la foto en RGB de 8 bits, girada según su EXIF.

    Las de 16 bits se escalan a 8 bits con su valor máximo (convertirlas sin más las deja casi blancas).
    """
    if isinstance(foto, Image.Image):
        imagen = foto.copy()
    else:
        try:
            with Image.open(foto) as abierta:
                abierta.load()
                imagen = abierta.copy()
        except (OSError, ValueError) as error:
            raise ErrorEntrada(f"No se puede leer {foto}: {error}") from error
    imagen = ImageOps.exif_transpose(imagen)
    if imagen.mode in MODOS_16_BITS:
        valores = np.asarray(imagen, dtype=np.float32)
        maximo = float(valores.max()) if valores.size else 0.0
        if maximo > 255:
            valores = valores * (255.0 / maximo)
        imagen = Image.fromarray(np.clip(valores, 0, 255).astype(np.uint8))
    return imagen.convert("RGB")


def buscar_imagenes(carpeta: Path, excluir: Path | None = None) -> list[Path]:
    """Todas las fotos de la carpeta y sus subcarpetas, ordenadas. Se salta solo la carpeta ``excluir``."""
    excluida = excluir.resolve() if excluir else None
    return sorted(
        ruta for ruta in carpeta.rglob("*")
        if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES
        and not (excluida and ruta.resolve().is_relative_to(excluida))
    )
