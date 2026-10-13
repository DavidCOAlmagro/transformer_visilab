"""Nombres legibles de las clases del modelo (``Cocconeis_placentula_var_euglypta`` -> ``Cocconeis placentula var. euglypta``)."""

from __future__ import annotations

RESTOS = {"Debris": "Restos", "Fragments": "Fragmento"}
ABREVIATURAS = {"var": "var.", "f": "f.", "sp": "sp."}


def tipo_clase(codigo: str) -> str:
    """``especie``, ``vista pleural`` (diatomea de lado: solo se sabe el género) o ``resto``."""
    if codigo in RESTOS:
        return "resto"
    if codigo.endswith("_fp"):
        return "vista pleural"
    return "especie"


def nombre_legible(codigo: str) -> str:
    if codigo in RESTOS:
        return RESTOS[codigo]
    if codigo.endswith("_fp"):
        return f"{codigo.removesuffix('_fp')} (vista pleural)"
    return " ".join(ABREVIATURAS.get(parte, parte) for parte in codigo.split("_"))
