"""
--------------------------------------
Elige los recortes YOLO de las fotos de campo del POOL que se aceptan como
ground truth para entrenar (spec 011). No necesita las imágenes: usa los Excel
de inferencia de DINOv2 y ResNet50 y la etiqueta de cada foto.

Criterio (recorte -> especie de la foto):
- uno de los dos modelos acierta con confianza >= 80 % y el otro tiene la
  especie en su top-3, o
- los dos dicen la misma `<Genero>_fp` y el género es el de la foto
  (posición pleural de esa especie).
Tope de recortes por clase; primero los que DINOv2 falla (los que más enseñan).
Las fotos de recursos/campo_test.txt no se usan nunca.

Uso:
    python src/seleccionar_recortes_campo.py --dino <excel> --resnet <excel> --etiquetas cruce_ground_truth.xlsx
--------------------------------------
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from dividir_campo import clave_grupo
from evaluar_campo import cargar_etiquetas, cargar_recortes, obtener_genero

RAIZ = Path(__file__).resolve().parent.parent
CLAVE = ["image", "item", "x1", "y1", "x2", "y2"]


def nombre_recorte(image: str, item: int, grupo: str) -> str:
    """Nombre del PNG de un recorte. Empieza por su grupo (muestra) para poder repartir por foto sin partirla."""
    seguro = lambda texto: re.sub(r"[^0-9A-Za-z.-]+", "_", texto).strip("_")
    return f"{seguro(grupo)}__{seguro(Path(image).stem)}__roi{item}.png"


def grupo_de_recorte(nombre_archivo: str) -> str:
    return nombre_archivo.split("__", 1)[0]


def leer_lista(ruta: Path) -> set[str]:
    return {linea.strip() for linea in ruta.read_text(encoding="utf-8").splitlines() if linea.strip()}


def seleccionar(recortes: pd.DataFrame, clases: set[str], confianza: float = 80.0,
                tope: int = 300, semilla: int = 42) -> pd.DataFrame:
    """``recortes``: una fila por recorte con etiqueta, top1-3 y conf1 de cada modelo (sufijos _d y _r)."""
    r = recortes[recortes.etiqueta.isin(clases)].copy()
    en_top3 = {s: r.apply(lambda x, s=s: x.etiqueta in (x[f"top1_{s}"], x[f"top2_{s}"], x[f"top3_{s}"]), axis=1)
               for s in ("d", "r")}
    acierta = {s: (r[f"top1_{s}"] == r.etiqueta) & (r[f"conf1_{s}"] >= confianza) for s in ("d", "r")}
    especie = (acierta["d"] & en_top3["r"]) | (acierta["r"] & en_top3["d"])

    pleural = (r.top1_d == r.top1_r) & r.top1_d.str.endswith("_fp") & r.top1_d.isin(clases)
    pleural &= r.top1_d.map(obtener_genero) == r.etiqueta.map(obtener_genero)

    r["especie"] = None
    r.loc[especie, "especie"] = r.loc[especie, "etiqueta"]
    r.loc[pleural & ~especie, "especie"] = r.loc[pleural & ~especie, "top1_d"]
    elegidos = r[r.especie.notna()].copy()
    elegidos["dificil"] = elegidos.top1_d != elegidos.especie
    # Tope por clase, con prioridad a los recortes que DINOv2 falla
    elegidos = elegidos.sample(frac=1, random_state=semilla).sort_values("dificil", ascending=False, kind="stable")
    elegidos = elegidos.groupby("especie", group_keys=False).head(tope)
    elegidos["grupo"] = elegidos.image.map(clave_grupo)
    return elegidos[CLAVE + ["especie", "grupo", "dificil"]].sort_values(CLAVE).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recortes de campo aceptados como ground truth (spec 011).")
    parser.add_argument("--dino", type=Path, required=True)
    parser.add_argument("--resnet", type=Path, required=True)
    parser.add_argument("--etiquetas", type=Path, required=True)
    parser.add_argument("--pool", type=Path, default=RAIZ / "recursos" / "campo_pool.txt")
    parser.add_argument("--test", type=Path, default=RAIZ / "recursos" / "campo_test.txt")
    parser.add_argument("--clases", type=Path, default=RAIZ / "Inferir" / "txt_classes" / "classes_77(dino).txt")
    parser.add_argument("--tope", type=int, default=300)
    parser.add_argument("--salida", type=Path, default=RAIZ / "recursos" / "recortes_campo.csv")
    args = parser.parse_args()

    pool, test = leer_lista(args.pool), leer_lista(args.test)
    if pool & test:
        raise SystemExit("campo_pool y campo_test se solapan: revisa dividir_campo.py.")
    recortes = cargar_recortes(args.dino, "dinov2").merge(cargar_recortes(args.resnet, "resnet"), on=CLAVE,
                                                          suffixes=("_d", "_r"))
    etiquetas = cargar_etiquetas(args.etiquetas).set_index("imagen").etiqueta
    recortes = recortes[recortes.image.isin(pool)].copy()
    recortes["etiqueta"] = recortes.image.map(etiquetas)
    clases = leer_lista(args.clases)

    elegidos = seleccionar(recortes, clases, tope=args.tope)
    assert not elegidos.image.isin(test).any(), "Un recorte de campo_test se ha colado en la selección"
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    elegidos.to_csv(args.salida, index=False, encoding="utf-8")
    print(f"{len(elegidos)} recortes de {elegidos.image.nunique()} fotos, {elegidos.especie.nunique()} clases "
          f"({int(elegidos.dificil.sum())} que DINOv2 falla) -> {args.salida}")


if __name__ == "__main__":
    main()
