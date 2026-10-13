"""Excel de resultados: hojas Recortes (una fila por diatomea), Imagenes (una por foto) e Info."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .nombres import nombre_legible
from .resultados import Resultado
from .version import VERSION

COLUMNAS_RECORTES = ["imagen", "recorte", "x1", "y1", "x2", "y2", "especie", "tipo",
                     "top1", "% top1", "top2", "% top2", "top3", "% top3"]
COLUMNAS_IMAGENES = ["imagen", "estado", "diatomeas", "mensaje"]


def filas_recortes(resultados: list[Resultado]) -> list[dict]:
    filas = []
    for resultado in resultados:
        for recorte in resultado.recortes:
            fila = {"imagen": resultado.imagen, "recorte": recorte.numero,
                    **dict(zip(("x1", "y1", "x2", "y2"), recorte.caja)),
                    "especie": recorte.especie, "tipo": recorte.tipo}
            for posicion in range(3):
                codigo, probabilidad = recorte.top3[posicion] if posicion < len(recorte.top3) else ("", None)
                fila[f"top{posicion + 1}"] = nombre_legible(codigo) if codigo else ""
                fila[f"% top{posicion + 1}"] = None if probabilidad is None else round(100 * probabilidad, 1)
            filas.append(fila)
    return filas


def guardar_excel(resultados: list[Resultado], ruta: Path, version_modelos: str) -> Path:
    recortes = pd.DataFrame(filas_recortes(resultados), columns=COLUMNAS_RECORTES)
    imagenes = pd.DataFrame([{"imagen": r.imagen, "estado": r.estado, "diatomeas": len(r.recortes), "mensaje": r.mensaje}
                             for r in resultados], columns=COLUMNAS_IMAGENES)
    info = pd.DataFrame([
        ("Versión del clasificador", VERSION),
        ("Modelos", version_modelos),
        ("Fecha", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Fotos", len(resultados)),
        ("Diatomeas", len(recortes)),
    ], columns=["", ""])
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(ruta, engine="xlsxwriter") as excel:
        for nombre, tabla, anchos in (("Recortes", recortes, [40, 8, 7, 7, 7, 7, 38, 14, 38, 8, 38, 8, 38, 8]),
                                      ("Imagenes", imagenes, [40, 16, 10, 60]),
                                      ("Info", info, [26, 30])):
            tabla.to_excel(excel, sheet_name=nombre, index=False, header=nombre != "Info")
            hoja = excel.sheets[nombre]
            for columna, ancho in enumerate(anchos):
                hoja.set_column(columna, columna, ancho)
            if nombre != "Info":
                hoja.freeze_panes(1, 0)
    return ruta
