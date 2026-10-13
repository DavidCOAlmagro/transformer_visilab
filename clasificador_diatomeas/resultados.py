from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .nombres import nombre_legible, tipo_clase


@dataclass
class Recorte:
    """Una diatomea detectada en la foto."""

    numero: int                          # orden de la detección en la foto (1, 2, ...)
    caja: tuple[int, int, int, int]      # x1, y1, x2, y2 en píxeles de la foto
    top3: list[tuple[str, float]]        # (clase, probabilidad de 0 a 1), de más a menos probable
    clase_detector: str = ""             # lo que dijo YOLO: Diatom / Not_diatom

    @property
    def codigo(self) -> str:
        return self.top3[0][0]

    @property
    def especie(self) -> str:
        return nombre_legible(self.codigo)

    @property
    def tipo(self) -> str:
        return tipo_clase(self.codigo)


@dataclass
class Resultado:
    """Resultado de una foto. ``estado``: ``ok``, ``sin detecciones`` o ``error``."""

    imagen: str
    estado: str
    recortes: list[Recorte] = field(default_factory=list)
    mensaje: str = ""
    probabilidades: dict[str, Any] | None = field(default=None, repr=False)  # solo con guardar_probabilidades
