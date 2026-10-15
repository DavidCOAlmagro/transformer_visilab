"""
--------------------------------------
Todo lo que tiene que ver con las fotos: abrirlas, buscarlas en una
carpeta y dibujar encima las cajas con el nombre de cada especie.
--------------------------------------
"""
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MODOS_16_BITS = {"I;16", "I;16B", "I;16L", "I;16N", "I", "F"}
COLOR = (0, 200, 0)


class ErrorEntrada(Exception):
    """Una foto no se puede leer (fichero dañado o formato no soportado)."""


def abrir_imagen(foto: str | Path | Image.Image) -> Image.Image:
    """
    Devuelve la foto en RGB de 8 bits y girada según su EXIF.
    Las fotos de 16 bits se escalan a 8 bits con su valor máximo: si se
    convierten sin más salen casi blancas y YOLO no ve nada.
    """
    if isinstance(foto, Image.Image):
        imagen = foto.copy()
    else:
        try:
            with Image.open(foto) as abierta:
                abierta.load()
                imagen = abierta.copy()
        except (OSError, ValueError) as error:
            raise ErrorEntrada(f"No se puede abrir {Path(foto).name}: no es una imagen o está dañada") from error

    imagen = ImageOps.exif_transpose(imagen)

    if imagen.mode in MODOS_16_BITS:
        valores = np.asarray(imagen, dtype=np.float32)
        maximo = float(valores.max()) if valores.size else 0.0
        if maximo > 255:
            valores = valores * (255.0 / maximo)
        imagen = Image.fromarray(np.clip(valores, 0, 255).astype(np.uint8))

    return imagen.convert("RGB")


def buscar_imagenes(carpeta: Path, excluir: Path | None = None) -> list[Path]:
    """
    Todas las fotos de la carpeta y sus subcarpetas, ordenadas. Solo se salta
    la carpeta "excluir" (la de resultados, por si está dentro de la de fotos).
    """
    excluida = excluir.resolve() if excluir else None
    fotos = []
    for ruta in carpeta.rglob("*"):
        if not ruta.is_file() or ruta.suffix.lower() not in EXTENSIONES:
            continue
        if excluida and ruta.resolve().is_relative_to(excluida):
            continue
        fotos.append(ruta)
    return sorted(fotos)


def nombre_fichero(relativa: Path) -> str:
    """
    Nombre para la foto anotada a partir de su ruta: "muestra 1/foto.tif" -> "muestra_1__foto".
    Es único para cada foto y vale en cualquier sistema.
    """
    partes = []
    for parte in relativa.with_suffix("").parts:
        partes.append(re.sub(r"[^0-9A-Za-zÀ-ÿ.-]+", "_", parte).strip(" ._") or "imagen")
    return "__".join(partes) or "imagen"


@lru_cache(maxsize=8)
def _fuente(tamano: int) -> ImageFont.ImageFont:
    """
    Letra DejaVu Sans Bold. Viene con matplotlib, que se instala con ultralytics,
    así no hay que llevar ningún fichero de letra aparte (ni acordarse de él al
    empaquetar la app). Si no estuviera, se usa la de Pillow.
    """
    try:
        import matplotlib
        ruta = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        return ImageFont.truetype(str(ruta), tamano)
    except (ImportError, OSError):
        return ImageFont.load_default(size=tamano)


def posicion_etiqueta(caja: tuple[int, int, int, int], ancho_etiqueta: int, alto_etiqueta: int,
                      ancho: int, alto: int, ocupadas: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """
    Dónde poner la etiqueta de una caja: encima, debajo o dentro. Si ahí ya hay
    otra etiqueta, se va apilando más abajo o más arriba hasta encontrar hueco.
    """
    x1, y1, x2, y2 = caja
    x = max(0, min(x1, ancho - ancho_etiqueta))

    candidatas = [y1 - alto_etiqueta, y2, y1]
    for k in range(1, 6):
        candidatas.append(y2 + k * alto_etiqueta)
        candidatas.append(y1 - (k + 1) * alto_etiqueta)
    caben = [y for y in candidatas if 0 <= y and y + alto_etiqueta <= alto]

    for y in caben:
        pisa_otra = False
        for ox1, oy1, ox2, oy2 in ocupadas:
            if x < ox2 and ox1 < x + ancho_etiqueta and y < oy2 and oy1 < y + alto_etiqueta:
                pisa_otra = True
                break
        if not pisa_otra:
            return x, y, x + ancho_etiqueta, y + alto_etiqueta

    # Si no hay hueco libre, la primera que cabe (o encima de la caja)
    y = caben[0] if caben else max(0, y1)
    return x, y, x + ancho_etiqueta, y + alto_etiqueta


def anotar(imagen: Image.Image, cajas: list[tuple[int, int, int, int]], nombres: list[str]) -> Image.Image:
    """
    Copia de la foto con un recuadro verde en cada diatomea y su nombre encima
    (sin porcentaje). La letra y el grosor van según el tamaño de la foto.
    """
    anotada = imagen.copy()
    dibujo = ImageDraw.Draw(anotada)
    ancho, alto = anotada.size
    lado_corto = min(ancho, alto)
    fuente = _fuente(max(24, lado_corto // 30))
    grosor = max(4, lado_corto // 200)
    margen = max(4, grosor)

    # Primero todos los recuadros y después las etiquetas, para que ninguna caja tape un nombre
    for caja in cajas:
        dibujo.rectangle(caja, outline=COLOR, width=grosor)

    ocupadas = []
    for caja, nombre in zip(cajas, nombres):
        izquierda, arriba, derecha, abajo = dibujo.textbbox((0, 0), nombre, font=fuente)
        ancho_etiqueta = derecha - izquierda + 2 * margen
        alto_etiqueta = abajo - arriba + 2 * margen
        etiqueta = posicion_etiqueta(caja, ancho_etiqueta, alto_etiqueta, ancho, alto, ocupadas)
        ocupadas.append(etiqueta)
        dibujo.rectangle(etiqueta, fill=COLOR)
        dibujo.text((etiqueta[0] + margen - izquierda, etiqueta[1] + margen - arriba), nombre, fill=(0, 0, 0), font=fuente)

    return anotada
