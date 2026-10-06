"""
--------------------------------------
Evaluación en campo (ambiente no controlado), por IMAGEN.

Parte de los Excel por recorte que genera
``Inferir/infer_and_split_resnet_single_folder.py`` (una fila por detección YOLO)
y de una tabla de etiquetas por imagen. Cada imagen recibe una especie
agregando sus recortes con una regla (voto, recorte mayor o suma de
confianza) y se compara con la etiqueta.

Las predicciones que no son una especie no deben votar:
- ``Debris`` y ``Fragments`` (no son diatomeas completas).
- ``*_fp``: posición pleural (vista lateral, solo identificable a nivel de
  género). Una imagen etiquetada con una especie puede contener recortes en
  posición pleural de esa misma especie; si votan, desplazan a la especie.

Uso:
    python3 src/evaluar_campo.py --dino C:/VISILAB/classification_results_dinov2.xlsx \
        --resnet C:/VISILAB/classification_results_resnet.xlsx \
        --etiquetas C:/VISILAB/cruce_ground_truth.xlsx --salida informe_campo.xlsx
--------------------------------------
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CLASES_DINO = RAIZ / "Inferir" / "txt_classes" / "classes_77(dino).txt"
NO_ESPECIES = {"Debris", "Fragments"}
# Erratas de classes_78(resnet).txt respecto a los nombres de carpeta.
ALIAS_CLASES = {
    "Denticula_tenius": "Denticula_tenuis",
    "Nitzschia_dessertorum": "Nitzschia_desertorum",
}
CLAVE_RECORTE = ["image", "item", "x1", "y1", "x2", "y2"]
REGLAS = ("voto", "mayor", "suma_conf")


def es_posicion_pleural(clase: str) -> bool:
    """``Genero_fp`` es la vista pleural del género, no una especie."""
    return clase.endswith("_fp")


def obtener_genero(clase: str) -> str:
    """Primer token antes de '_' (mismo criterio que preparar_datos.obtener_genero)."""
    return clase.split("_")[0]


def clase_excluida(clase: str, excluir_fp: bool) -> bool:
    """Indica si una predicción no debe participar en la agregación."""
    return (not isinstance(clase, str) or not clase or clase in NO_ESPECIES
            or (excluir_fp and es_posicion_pleural(clase)))


def agregar_imagen(recortes: list[tuple[str, float, float]], regla: str,
                   excluir_fp: bool = False) -> str:
    """
    Decide la especie de una imagen a partir de sus recortes
    ``(clase, confianza_%, area)``. Devuelve "" si no queda ningún recorte válido.
    - voto: clase más repetida; empate -> mayor suma de confianza.
    - mayor: clase del recorte de mayor área.
    - suma_conf: clase con mayor suma de confianza.
    """
    validos = [r for r in recortes if not clase_excluida(r[0], excluir_fp)]
    if not validos:
        return ""
    if regla == "mayor":
        return max(validos, key=lambda r: r[2])[0]
    votos: dict[str, int] = defaultdict(int)
    suma: dict[str, float] = defaultdict(float)
    for clase, confianza, _ in validos:
        votos[clase] += 1
        suma[clase] += confianza
    if regla == "voto":
        return max(votos, key=lambda c: (votos[c], suma[c]))
    if regla == "suma_conf":
        return max(suma, key=suma.get)
    raise ValueError(f"Regla desconocida: {regla}")


def cargar_recortes(ruta: Path, prefijo: str) -> pd.DataFrame:
    """Lee un Excel de inferencia y normaliza columnas: clave + clase/confianza top-1..3."""
    df = pd.read_excel(ruta)
    columnas = {f"{prefijo}_top{i}": f"top{i}" for i in (1, 2, 3)}
    columnas.update({f"{prefijo}_top{i}_confianza_%": f"conf{i}" for i in (1, 2, 3)})
    faltan = [c for c in [*CLAVE_RECORTE, *columnas] if c not in df.columns]
    if faltan:
        raise ValueError(f"{ruta} no tiene las columnas {faltan}")
    df = df[[*CLAVE_RECORTE, *columnas]].rename(columns=columnas)
    for i in (1, 2, 3):
        df[f"top{i}"] = df[f"top{i}"].replace(ALIAS_CLASES)
        df[f"conf{i}"] = pd.to_numeric(df[f"conf{i}"], errors="coerce").fillna(0.0)
    df["area"] = (df.x2 - df.x1).clip(lower=0) * (df.y2 - df.y1).clip(lower=0)
    return df


def cargar_etiquetas(ruta: Path) -> pd.DataFrame:
    """Etiquetas por imagen: hoja ``Por_imagen`` del cruce o CSV/Excel con ``imagen, etiqueta``."""
    if ruta.suffix.lower() == ".csv":
        df = pd.read_csv(ruta)
    else:
        hojas = pd.ExcelFile(ruta).sheet_names
        df = pd.read_excel(ruta, sheet_name="Por_imagen" if "Por_imagen" in hojas else 0)
    if not {"imagen", "etiqueta"} <= set(df.columns):
        raise ValueError(f"{ruta} debe tener columnas 'imagen' y 'etiqueta'")
    return df[["imagen", "etiqueta"]].drop_duplicates("imagen")


def predecir_imagenes(recortes: pd.DataFrame, regla: str, excluir_fp: bool) -> pd.Series:
    """Especie predicha por imagen (índice = image)."""
    tuplas = recortes.groupby("image")[["top1", "conf1", "area"]].apply(
        lambda g: list(g.itertuples(index=False, name=None)))
    return tuplas.map(lambda r: agregar_imagen(r, regla, excluir_fp))


def evaluar(modelos: dict[str, pd.DataFrame], etiquetas: pd.DataFrame,
            clases_evaluables: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve (tabla resumen, predicciones por imagen). Solo cuentan las
    imágenes cuya etiqueta es una especie que conocen los modelos.
    Imágenes sin recortes válidos cuentan como fallo.
    """
    evaluables = etiquetas[etiquetas.etiqueta.isin(clases_evaluables)].set_index("imagen")
    predicciones = evaluables.copy()
    filas = []
    # El ensamble suma los recortes de ambos modelos (misma clave de recorte).
    combinados = dict(modelos)
    if len(modelos) > 1:
        combinados["ensamble"] = pd.concat(modelos.values(), ignore_index=True)
    for excluir_fp in (False, True):
        for regla in REGLAS:
            fila = {"regla": regla, "excluye_fp": excluir_fp}
            for nombre, recortes in combinados.items():
                pred = predecir_imagenes(recortes, regla, excluir_fp).reindex(evaluables.index).fillna("")
                col = f"{nombre}_{regla}{'_sinfp' if excluir_fp else ''}"
                predicciones[col] = pred
                fila[f"acc_{nombre}"] = round((pred == evaluables.etiqueta).mean(), 4)
                fila[f"acc_genero_{nombre}"] = round(
                    (pred.map(obtener_genero) == evaluables.etiqueta.map(obtener_genero)).mean(), 4)
            filas.append(fila)
    resumen = pd.DataFrame(filas)
    resumen.insert(0, "n_imagenes", len(evaluables))
    return resumen, predicciones.reset_index()


def por_especie(predicciones: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Accuracy por especie de la etiqueta para las columnas de predicción indicadas."""
    aciertos = predicciones[columnas].eq(predicciones.etiqueta, axis=0)
    aciertos["etiqueta"] = predicciones.etiqueta
    tabla = aciertos.groupby("etiqueta").mean().round(3)
    tabla.insert(0, "n_imagenes", predicciones.groupby("etiqueta").size())
    return tabla.sort_values(columnas[0]).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Accuracy por imagen en campo (DINOv2 / ResNet50).")
    parser.add_argument("--dino", type=Path)
    parser.add_argument("--resnet", type=Path)
    parser.add_argument("--etiquetas", type=Path, required=True)
    parser.add_argument("--clases", type=Path, default=CLASES_DINO,
                        help="Especies evaluables (por defecto, las 77 de DINOv2).")
    parser.add_argument("--salida", type=Path, help="Excel con resumen, por especie y por imagen.")
    args = parser.parse_args()

    modelos: dict[str, pd.DataFrame] = {}
    if args.dino:
        modelos["dinov2"] = cargar_recortes(args.dino, "dinov2")
    if args.resnet:
        modelos["resnet"] = cargar_recortes(args.resnet, "resnet")
    if not modelos:
        raise SystemExit("Indica al menos --dino o --resnet.")

    clases = {linea.strip() for linea in args.clases.read_text(encoding="utf-8").splitlines()
              if linea.strip()} - NO_ESPECIES
    resumen, predicciones = evaluar(modelos, cargar_etiquetas(args.etiquetas), clases)
    print(resumen.to_string(index=False))

    if args.salida:
        columnas = [c for c in predicciones.columns if c.endswith("suma_conf_sinfp")]
        with pd.ExcelWriter(args.salida) as escritor:
            resumen.to_excel(escritor, sheet_name="Resumen", index=False)
            por_especie(predicciones, columnas).to_excel(escritor, sheet_name="Por_especie", index=False)
            predicciones.to_excel(escritor, sheet_name="Por_imagen", index=False)
        print(f"Informe guardado en: {args.salida}")


if __name__ == "__main__":
    main()
