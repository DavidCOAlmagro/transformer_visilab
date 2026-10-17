"""
--------------------------------------
Nombres de las especies y Excel de resultados.

El Excel tiene tres hojas:
    Recortes: una fila por diatomea, con su especie y su top-3
    Imagenes: una fila por foto, con su estado (ok, sin detecciones o error)
    Info:     versión y fecha
--------------------------------------
"""
from datetime import datetime

import pandas as pd


def nombre_legible(clase: str) -> str:
    """
    De la clase del modelo al nombre que se muestra:
        Cocconeis_placentula_var_euglypta -> Cocconeis placentula var. euglypta
        Gomphonema_fp                     -> Gomphonema (vista pleural)
        Debris / Fragments                -> Restos / Fragmento
    """
    if clase == "Debris":
        return "Restos"
    if clase == "Fragments":
        return "Fragmento"
    if clase.endswith("_fp"):
        return clase[:-3] + " (vista pleural)"
    palabras = clase.split("_")
    for i, palabra in enumerate(palabras):
        if palabra in ("var", "f", "sp"):
            palabras[i] = palabra + "."
    return " ".join(palabras)


def tipo(clase: str) -> str:
    """
    "especie", "vista pleural" (la diatomea está de lado y solo se sabe el género)
    o "resto" (restos y fragmentos).
    """
    if clase in ("Debris", "Fragments"):
        return "resto"
    if clase.endswith("_fp"):
        return "vista pleural"
    return "especie"


def guardar(resultados: list, ruta, version: str):
    """Escribe el Excel con las tres hojas."""
    filas_recortes = []
    filas_imagenes = []
    for resultado in resultados:
        filas_imagenes.append({"imagen": resultado.imagen, "estado": resultado.estado,
                               "diatomeas": len(resultado.recortes), "mensaje": resultado.mensaje})
        for recorte in resultado.recortes:
            x1, y1, x2, y2 = recorte.caja
            fila = {"imagen": resultado.imagen, "recorte": recorte.numero,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "especie": recorte.especie, "tipo": recorte.tipo}
            for posicion, (especie, porcentaje) in enumerate(recorte.top3, start=1):
                fila[f"top{posicion}"] = especie
                fila[f"% top{posicion}"] = porcentaje
            filas_recortes.append(fila)

    columnas_recortes = ["imagen", "recorte", "x1", "y1", "x2", "y2", "especie", "tipo",
                         "top1", "% top1", "top2", "% top2", "top3", "% top3"]
    recortes = pd.DataFrame(filas_recortes, columns=columnas_recortes)
    imagenes = pd.DataFrame(filas_imagenes, columns=["imagen", "estado", "diatomeas", "mensaje"])
    info = pd.DataFrame([("Versión", version), ("Fecha", datetime.now().strftime("%Y-%m-%d %H:%M")),
                         ("Fotos", len(resultados)), ("Diatomeas", len(recortes))])

    with pd.ExcelWriter(ruta, engine="xlsxwriter") as excel:
        recortes.to_excel(excel, sheet_name="Recortes", index=False)
        imagenes.to_excel(excel, sheet_name="Imagenes", index=False)
        info.to_excel(excel, sheet_name="Info", index=False, header=False)
        # Columnas un poco más anchas para que se lean los nombres
        excel.sheets["Recortes"].set_column(0, 0, 40)
        excel.sheets["Recortes"].set_column(6, 13, 30)
        excel.sheets["Imagenes"].set_column(0, 3, 30)
        excel.sheets["Info"].set_column(0, 1, 25)
