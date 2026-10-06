"""Preprocesado común para DINOv2.

El modelo recibe imágenes cuadradas para evitar que el center crop del
``AutoImageProcessor`` elimine parte de un ROI alargado. El padding conserva
la relación de aspecto y se aplica tanto al generar embeddings como al inferir.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image, ImageStat


@dataclass(frozen=True)
class ConfiguracionPreprocesado:
    version: str = "dinov2-pad-square-v1"
    modo: str = "pad_square"
    color_relleno: str = "mean_border"
    tamano: int = 224

    def como_dict(self) -> dict[str, Any]:
        return asdict(self)


CONFIGURACION_PREPROCESADO = ConfiguracionPreprocesado()


def _color_borde(imagen: Image.Image) -> tuple[int, int, int]:
    """Obtiene un color estable a partir del borde de la imagen."""
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


def pad_to_square(
    imagen: Image.Image,
    color: tuple[int, int, int] | str = "mean_border",
) -> Image.Image:
    """Centra una imagen en un lienzo cuadrado sin deformarla ni recortarla."""
    rgb = imagen.convert("RGB")
    ancho, alto = rgb.size
    lado = max(ancho, alto)
    if lado <= 0:
        raise ValueError("La imagen debe tener dimensiones positivas.")
    if isinstance(color, str):
        if color == "mean_border":
            color = _color_borde(rgb)
        elif color == "gray":
            color = (128, 128, 128)
        else:
            raise ValueError(f"Color de relleno no soportado: {color}")
    lienzo = Image.new("RGB", (lado, lado), color=color)
    lienzo.paste(rgb, ((lado - ancho) // 2, (lado - alto) // 2))
    return lienzo


def preparar_para_dinov2(imagen: Image.Image) -> Image.Image:
    """Aplica el preprocesado recomendado antes de ``AutoImageProcessor``."""
    return pad_to_square(imagen, CONFIGURACION_PREPROCESADO.color_relleno)
