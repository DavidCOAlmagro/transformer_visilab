"""Fotos anotadas: recuadro verde y el nombre de la especie, sin porcentaje."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .resultados import Recorte

COLOR = (0, 200, 0)
FUENTE = Path(__file__).parent / "fuentes" / "DejaVuSans-Bold.ttf"


@lru_cache(maxsize=8)
def _fuente(tamano: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(FUENTE), tamano)
    except OSError:
        return ImageFont.load_default(size=tamano)


def anotar(imagen: Image.Image, recortes: list[Recorte]) -> Image.Image:
    """Copia de la foto con cada caja y su nombre. Letra y grosor proporcionales al tamaño de la foto."""
    anotada = imagen.copy()
    dibujo = ImageDraw.Draw(anotada)
    ancho, alto = anotada.size
    lado_corto = min(ancho, alto)
    fuente = _fuente(max(24, lado_corto // 30))
    grosor = max(4, lado_corto // 200)
    margen = max(4, grosor)
    for recorte in recortes:
        dibujo.rectangle(recorte.caja, outline=COLOR, width=grosor)
    ocupadas: list[tuple[int, int, int, int]] = []
    for recorte in recortes:
        x1, y1, x2, y2 = recorte.caja
        izquierda, arriba, derecha, abajo = dibujo.textbbox((0, 0), recorte.especie, font=fuente)
        ancho_e, alto_e = derecha - izquierda + 2 * margen, abajo - arriba + 2 * margen
        x = max(0, min(x1, ancho - ancho_e))
        # Encima de la caja, debajo o dentro: la primera posición que quepa sin pisar otra etiqueta
        caben = [y for y in (y1 - alto_e, y2, y1) if 0 <= y and y + alto_e <= alto]
        y = next(
            (y for y in caben if not any(x < ox2 and ox1 < x + ancho_e and y < oy2 and oy1 < y + alto_e
                                         for ox1, oy1, ox2, oy2 in ocupadas)),
            caben[0] if caben else max(0, y1),
        )
        ocupadas.append((x, y, x + ancho_e, y + alto_e))
        dibujo.rectangle(ocupadas[-1], fill=COLOR)
        dibujo.text((x + margen - izquierda, y + margen - arriba), recorte.especie, fill=(0, 0, 0), font=fuente)
    return anotada
