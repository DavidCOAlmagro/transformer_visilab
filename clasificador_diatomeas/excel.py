"""
--------------------------------------
El Excel de resultados y los nombres que aparecen en él.

    Recortes: una fila por diatomea, con su especie y su top-3
    Imagenes: una fila por foto, con su estado (ok, sin detecciones o error)
    Info:     versión del clasificador y de los modelos
--------------------------------------
"""
from datetime import datetime
from pathlib import Path

import pandas as pd

COLUMNAS_RECORTES = ["imagen", "recorte", "x1", "y1", "x2", "y2", "especie", "tipo",
                     "top1", "% top1", "top2", "% top2", "top3", "% top3"]
COLUMNAS_IMAGENES = ["imagen", "estado", "diatomeas", "mensaje"]

RESTOS = {"Debris": "Restos", "Fragments": "Fragmento"}
ABREVIATURAS = {"var": "var.", "f": "f.", "sp": "sp."}


def tipo_clase(codigo: str) -> str:
    """
    "especie", "vista pleural" (la diatomea se ve de lado y solo se sabe el
    género) o "resto" (restos y fragmentos).
    """
    if codigo in RESTOS:
        return "resto"
    if codigo.endswith("_fp"):
        return "vista pleural"
    return "especie"


def nombre_legible(codigo: str) -> str:
    """
    Nombre de la clase tal como se escribe: "Cocconeis_placentula_var_euglypta"
    -> "Cocconeis placentula var. euglypta", "Gomphonema_fp" -> "Gomphonema (vista pleural)".
    """
    if codigo in RESTOS:
        return RESTOS[codigo]
    if codigo.endswith("_fp"):
        return f"{codigo.removesuffix('_fp')} (vista pleural)"
    palabras = []
    for parte in codigo.split("_"):
        palabras.append(ABREVIATURAS.get(parte, parte))
    return " ".join(palabras)


def filas_recortes(resultados: list) -> list[dict]:
    """Una fila por diatomea de todas las fotos."""
    filas = []
    for resultado in resultados:
        for recorte in resultado.recortes:
            x1, y1, x2, y2 = recorte.caja
            fila = {"imagen": resultado.imagen, "recorte": recorte.numero, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "especie": recorte.especie, "tipo": recorte.tipo}
            for posicion in range(3):
                if posicion < len(recorte.top3):
                    codigo, probabilidad = recorte.top3[posicion]
                    fila[f"top{posicion + 1}"] = nombre_legible(codigo)
                    fila[f"% top{posicion + 1}"] = round(100 * probabilidad, 1)
                else:
                    fila[f"top{posicion + 1}"] = ""
                    fila[f"% top{posicion + 1}"] = None
            filas.append(fila)
    return filas


def guardar_excel(resultados: list, ruta: Path, version: str, version_modelos: str) -> Path:
    """Escribe el Excel con las tres hojas y devuelve su ruta."""
    recortes = pd.DataFrame(filas_recortes(resultados), columns=COLUMNAS_RECORTES)

    filas_imagenes = []
    for resultado in resultados:
        filas_imagenes.append({"imagen": resultado.imagen, "estado": resultado.estado,
                               "diatomeas": len(resultado.recortes), "mensaje": resultado.mensaje})
    imagenes = pd.DataFrame(filas_imagenes, columns=COLUMNAS_IMAGENES)

    info = pd.DataFrame([
        ("Versión del clasificador", version),
        ("Modelos", version_modelos),
        ("Fecha", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Fotos", len(resultados)),
        ("Diatomeas", len(recortes)),
    ], columns=["", ""])

    hojas = [
        ("Recortes", recortes, [40, 8, 7, 7, 7, 7, 38, 14, 38, 8, 38, 8, 38, 8]),
        ("Imagenes", imagenes, [40, 16, 10, 60]),
        ("Info", info, [26, 30]),
    ]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(ruta, engine="xlsxwriter") as excel:
        for nombre, tabla, anchos in hojas:
            tabla.to_excel(excel, sheet_name=nombre, index=False, header=nombre != "Info")
            hoja = excel.sheets[nombre]
            for columna, ancho in enumerate(anchos):
                hoja.set_column(columna, columna, ancho)
            if nombre != "Info":
                hoja.freeze_panes(1, 0)  # la cabecera se queda fija al bajar
    return ruta
