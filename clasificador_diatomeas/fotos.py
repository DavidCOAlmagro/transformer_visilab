"""
--------------------------------------
Abrir las fotos y dibujar encima las diatomeas encontradas.
--------------------------------------
"""
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

VERDE = (0, 200, 0)


def abrir(foto) -> Image.Image:
    """
    Abre la foto (ruta o imagen de PIL) en RGB, girada según su EXIF.
    Las fotos de 16 bits se pasan a 8 bits; si no, salen casi blancas.
    Si el fichero no es una imagen, Pillow lanza OSError.
    """
    if isinstance(foto, Image.Image):
        imagen = foto.copy()
    else:
        with Image.open(foto) as abierta:
            abierta.load()
            imagen = abierta.copy()
    imagen = ImageOps.exif_transpose(imagen)

    if imagen.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
        valores = np.asarray(imagen, dtype=np.float32)
        if valores.max() > 255:
            valores = valores * (255.0 / valores.max())
        imagen = Image.fromarray(np.clip(valores, 0, 255).astype(np.uint8))

    return imagen.convert("RGB")


def fuente(tamano: int) -> ImageFont.ImageFont:
    """La letra DejaVu Sans Bold, que viene con matplotlib (se instala con ultralytics)."""
    try:
        return ImageFont.truetype(f"{matplotlib.get_data_path()}/fonts/ttf/DejaVuSans-Bold.ttf", tamano)
    except OSError:
        return ImageFont.load_default(size=tamano)


def se_pisan(a: tuple, b: tuple) -> bool:
    """True si dos rectángulos (x1, y1, x2, y2) se solapan."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def sitio_etiqueta(caja: tuple, ancho_etiqueta: int, alto_etiqueta: int, ancho: int, alto: int,
                   ocupados: list[tuple]) -> tuple:
    """
    Rectángulo para el nombre de una caja: encima, debajo o dentro de ella. Si ahí
    ya hay otro nombre, se prueba más abajo y más arriba hasta encontrar hueco.
    """
    x1, y1, x2, y2 = caja
    x = max(0, min(x1, ancho - ancho_etiqueta))
    posibles = [y1 - alto_etiqueta, y2, y1]
    for k in range(1, 6):
        posibles.append(y2 + k * alto_etiqueta)
        posibles.append(y1 - (k + 1) * alto_etiqueta)
    caben = [y for y in posibles if 0 <= y and y + alto_etiqueta <= alto]

    for y in caben:
        sitio = (x, y, x + ancho_etiqueta, y + alto_etiqueta)
        if not any(se_pisan(sitio, otro) for otro in ocupados):
            return sitio
    # Si no hay hueco libre, el primero que cabe
    y = caben[0] if caben else max(0, y1)
    return (x, y, x + ancho_etiqueta, y + alto_etiqueta)


def anotar(imagen: Image.Image, recortes: list) -> Image.Image:
    """
    Copia de la foto con un recuadro verde en cada diatomea y su nombre (sin porcentaje).
    La letra y el grosor van según el tamaño de la foto.
    """
    anotada = imagen.copy()
    dibujo = ImageDraw.Draw(anotada)
    ancho, alto = anotada.size
    letra = fuente(max(24, min(ancho, alto) // 30))
    grosor = max(4, min(ancho, alto) // 200)
    margen = max(4, grosor)

    # Primero todos los recuadros y luego los nombres, para que ningún recuadro tape un nombre
    for recorte in recortes:
        dibujo.rectangle(recorte.caja, outline=VERDE, width=grosor)

    ocupados = []
    for recorte in recortes:
        izquierda, arriba, derecha, abajo = dibujo.textbbox((0, 0), recorte.especie, font=letra)
        sitio = sitio_etiqueta(recorte.caja, derecha - izquierda + 2 * margen, abajo - arriba + 2 * margen,
                               ancho, alto, ocupados)
        ocupados.append(sitio)
        dibujo.rectangle(sitio, fill=VERDE)
        dibujo.text((sitio[0] + margen - izquierda, sitio[1] + margen - arriba), recorte.especie, fill=(0, 0, 0), font=letra)

    return anotada
