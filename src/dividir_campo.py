"""
--------------------------------------
Divide las imágenes de campo con etiqueta en un TEST FIJO y un POOL (spec 010).

- campo_test: solo para medir; nunca se usa para entrenar ni para decidir.
- campo_pool: candidatas a entrenar o validar (tras revisión del ground truth).

La división es estratificada por especie y por GRUPOS, para que no haya fuga:
las fotos que vienen de la misma muestra van siempre al mismo lado.
- Diatomeas DBO5 GT: el número de muestra (`Especie_98865_DC_07.tif` -> 98865).
- Imagenes Aqualitas: carpeta de especie + número base de la foto
  (`Especie_16.tif` y `Especie_16.2.tif` -> mismo grupo). Supuesto prudente:
  no se sabe si el número es una muestra compartida entre especies.

Uso:
    python src/dividir_campo.py --etiquetas C:/VISILAB/cruce_ground_truth.xlsx
--------------------------------------
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from evaluar_campo import cargar_etiquetas

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "recursos"
PATRON_DBO5 = re.compile(r"_(\d{4,})_DC_", re.IGNORECASE)
PATRON_FOTO = re.compile(r"_(\d+)(?:\.\d+)?\.[A-Za-z]+$")


def clave_grupo(imagen: str) -> str:
    """Grupo (muestra) de una imagen de campo; las de un mismo grupo no se separan."""
    partes = imagen.replace("\\", "/").split("/")
    nombre = partes[-1]
    muestra = PATRON_DBO5.search(nombre)
    if muestra:
        return f"dbo5:{muestra.group(1)}"
    foto = PATRON_FOTO.search(nombre)
    carpeta = partes[-2] if len(partes) > 1 else ""
    if foto:
        return f"{carpeta}:{foto.group(1)}"
    return f"{carpeta}:{nombre}"


def dividir(etiquetas: pd.DataFrame, partes: int = 3, semilla: int = 42) -> tuple[list[str], list[str]]:
    """Devuelve (test, pool): 1 de `partes` pliegues estratificados por especie y agrupados por muestra."""
    grupos = etiquetas.imagen.map(clave_grupo)
    pliegues = StratifiedGroupKFold(n_splits=partes, shuffle=True, random_state=semilla)
    _, indices_test = next(pliegues.split(etiquetas, etiquetas.etiqueta, grupos))
    en_test = etiquetas.index.isin(etiquetas.index[indices_test])
    return sorted(etiquetas.imagen[en_test]), sorted(etiquetas.imagen[~en_test])


def guardar_lista(rutas: list[str], ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(rutas) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test de campo fijo y pool (spec 010).")
    parser.add_argument("--etiquetas", type=Path, required=True)
    parser.add_argument("--salida", type=Path, default=SALIDA)
    args = parser.parse_args()

    etiquetas = cargar_etiquetas(args.etiquetas).reset_index(drop=True)
    test, pool = dividir(etiquetas)
    guardar_lista(test, args.salida / "campo_test.txt")
    guardar_lista(pool, args.salida / "campo_pool.txt")

    etiquetas["lado"] = etiquetas.imagen.isin(set(test)).map({True: "test", False: "pool"})
    etiquetas["grupo"] = etiquetas.imagen.map(clave_grupo)
    mezclados = etiquetas.groupby("grupo").lado.nunique().gt(1).sum()
    conteo = etiquetas.groupby("etiqueta").lado.value_counts().unstack(fill_value=0)
    sin_test = conteo[(conteo.sum(axis=1) >= 5) & (conteo.get("test", 0) == 0)].index.tolist()
    print(f"Imágenes: {len(etiquetas)} | test {len(test)} ({len(test) / len(etiquetas):.1%}) | pool {len(pool)}")
    print(f"Grupos: {etiquetas.grupo.nunique()} | grupos repartidos entre test y pool: {mezclados}")
    print(f"Especies con >= 5 imágenes y ninguna en test: {sin_test or 'ninguna'}")
    print(f"Guardado en: {args.salida / 'campo_test.txt'} y {args.salida / 'campo_pool.txt'}")


if __name__ == "__main__":
    main()
