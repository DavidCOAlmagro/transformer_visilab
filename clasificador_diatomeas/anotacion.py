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


def posicion_etiqueta(caja: tuple[int, int, int, int], ancho_e: int, alto_e: int, ancho: int, alto: int,
                      ocupadas: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """Rectángulo de la etiqueta: encima de la caja, debajo o dentro; si se pisa con otra, se apila más arriba o más abajo."""
    x1, y1, x2, y2 = caja
    x = max(0, min(x1, ancho - ancho_e))
    candidatas = [y1 - alto_e, y2, y1] + [y for k in range(1, 6) for y in (y2 + k * alto_e, y1 - (k + 1) * alto_e)]
    caben = [y for y in candidatas if 0 <= y and y + alto_e <= alto]
    libre = (y for y in caben if not any(x < ox2 and ox1 < x + ancho_e and y < oy2 and oy1 < y + alto_e
                                         for ox1, oy1, ox2, oy2 in ocupadas))
    y = next(libre, caben[0] if caben else max(0, y1))
    return x, y, x + ancho_e, y + alto_e


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
        izquierda, arriba, derecha, abajo = dibujo.textbbox((0, 0), recorte.especie, font=fuente)
        rectangulo = posicion_etiqueta(recorte.caja, derecha - izquierda + 2 * margen, abajo - arriba + 2 * margen,
                                       ancho, alto, ocupadas)
        ocupadas.append(rectangulo)
        dibujo.rectangle(rectangulo, fill=COLOR)
        dibujo.text((rectangulo[0] + margen - izquierda, rectangulo[1] + margen - arriba), recorte.especie,
                    fill=(0, 0, 0), font=fuente)
    return anotada
