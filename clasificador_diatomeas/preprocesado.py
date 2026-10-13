"""Relleno a cuadrado antes de DINOv2.

Es una copia de ``src/preprocesado.py`` (el que se usó para entrenar) para que el paquete no dependa del
resto del repo. Tiene que dar exactamente lo mismo: lo comprueba ``tests/test_clasificador.py``.
"""

from __future__ import annotations

from PIL import Image, ImageStat


def _color_borde(imagen: Image.Image) -> tuple[int, int, int]:
    """Color medio de una franja del borde de la imagen."""
    rgb = imagen.convert("RGB")
    ancho, alto = rgb.size
    if ancho == 0 or alto == 0:
        return (128, 128, 128)
    grosor = max(1, min(ancho, alto) // 20)
    pixeles = []
    pixeles.extend(rgb.crop((0, 0, ancho, min(grosor, alto))).getdata())
    pixeles.extend(rgb.crop((0, max(0, alto - grosor), ancho, alto)).getdata())
    pixeles.extend(rgb.crop((0, 0, min(grosor, ancho), alto)).getdata())
    pixeles.extend(rgb.crop((max(0, ancho - grosor), 0, ancho, alto)).getdata())
    muestra = Image.new("RGB", (len(pixeles), 1))
    muestra.putdata(pixeles)
    media = ImageStat.Stat(muestra).mean
    if not media:
        return (128, 128, 128)
    return tuple(max(0, min(255, round(valor))) for valor in media)


def preparar_para_dinov2(imagen: Image.Image) -> Image.Image:
    """Centra el recorte en un lienzo cuadrado del color del borde, sin deformarlo ni recortarlo."""
    rgb = imagen.convert("RGB")
    ancho, alto = rgb.size
    lado = max(ancho, alto)
    if lado <= 0:
        raise ValueError("La imagen debe tener dimensiones positivas.")
    lienzo = Image.new("RGB", (lado, lado), color=_color_borde(rgb))
    lienzo.paste(rgb, ((lado - ancho) // 2, (lado - alto) // 2))
    return lienzo
