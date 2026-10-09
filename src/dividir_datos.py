"""
--------------------------------------
Genera el reparto train/val/test (70/15/15) de las imágenes de entrenamiento,
estratificado por especie y reproducible (semilla 42), como lista de rutas
RELATIVAS a ``data/imagenes_visilab(raw)`` para que sirva en cualquier máquina:

    recursos/splits_<prueba>.txt.gz   (una línea por imagen: split<TAB>ruta)

Las especies son las de ESPECIES_FILTRADAS (constantes.py) y los nombres de
carpeta se normalizan (spec 009). La fuente `campo_pool` (recortes de campo,
spec 011) se reparte aparte, solo en train/val y sin partir ninguna foto: el
test interno sigue siendo de laboratorio y el de campo es recursos/campo_test.txt.

Uso:
    python src/dividir_datos.py --prueba 75_objetivo_ft
--------------------------------------
"""

from __future__ import annotations

import argparse
import gzip
from collections import Counter
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split

from constantes import VARIABLES_GLOBALES
from preparar_datos import rutas_imagenes
from seleccionar_recortes_campo import grupo_de_recorte

RAIZ = Path(__file__).resolve().parent.parent
FUENTE_CAMPO = "campo_pool"


def dividir(imagenes: list[tuple[Path, str]], semilla: int = 42) -> dict[str, list[tuple[Path, str]]]:
    """70/15/15 estratificado por especie."""
    especies = [e for _, e in imagenes]
    train, resto = train_test_split(imagenes, test_size=0.30, stratify=especies, random_state=semilla)
    val, test = train_test_split(resto, test_size=0.50, stratify=[e for _, e in resto], random_state=semilla)
    return {"train": train, "val": val, "test": test}


def dividir_campo(recortes: list[tuple[Path, str]], semilla: int = 42) -> dict[str, list[tuple[Path, str]]]:
    """Recortes de campo: ~85/15 train/val por foto (todos los de una foto, al mismo lado)."""
    if not recortes:
        return {"train": [], "val": []}
    grupos = [grupo_de_recorte(ruta.name) for ruta, _ in recortes]
    especies = [e for _, e in recortes]
    try:
        pliegues = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=semilla)
        indices_train, indices_val = next(pliegues.split(recortes, especies, grupos))
    except ValueError:
        # Muy pocos recortes para estratificar: reparto por foto sin estratificar
        partir = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=semilla)
        indices_train, indices_val = next(partir.split(recortes, especies, grupos))
    return {"train": [recortes[i] for i in indices_train], "val": [recortes[i] for i in indices_val]}


def guardar(splits: dict[str, list[tuple[Path, str]]], raiz_imagenes: Path, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(ruta, "wt", encoding="utf-8") as archivo:
        for nombre, items in splits.items():
            for imagen, _ in sorted(items):
                archivo.write(f"{nombre}\t{imagen.relative_to(raiz_imagenes).as_posix()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reparto train/val/test (rutas relativas).")
    parser.add_argument("--prueba", default=VARIABLES_GLOBALES["PRUEBA"])
    parser.add_argument("--salida", type=Path, default=RAIZ / "recursos")
    args = parser.parse_args()

    especies = set(VARIABLES_GLOBALES["ESPECIES_FILTRADAS"])
    imagenes = rutas_imagenes(especies)
    conteo = Counter(e for _, e in imagenes)
    faltan = sorted(especies - set(conteo))
    if faltan:
        print(f"Advertencia: especies sin imágenes: {faltan}")
    for especie, cantidad in sorted(conteo.items()):
        if cantidad < VARIABLES_GLOBALES["MINIMO_IMAGENES_POR_ESPECIE"]:
            print(f"Advertencia: {especie} solo tiene {cantidad} imágenes.")

    raiz_imagenes = VARIABLES_GLOBALES["RUTA_BASE"] / "imagenes_visilab(raw)"
    es_campo = [imagen.relative_to(raiz_imagenes).parts[0] == FUENTE_CAMPO for imagen, _ in imagenes]
    splits = dividir([item for item, campo in zip(imagenes, es_campo) if not campo])
    campo = dividir_campo([item for item, c in zip(imagenes, es_campo) if c])
    for nombre, items in campo.items():
        splits[nombre] = splits[nombre] + items
    if campo["train"]:
        print(f"Recortes de campo: train {len(campo['train'])} | val {len(campo['val'])}")
    ruta = args.salida / f"splits_{args.prueba}.txt.gz"
    guardar(splits, raiz_imagenes, ruta)
    print(" | ".join(f"{s}: {len(i)}" for s, i in splits.items()) + f" → {ruta}")


if __name__ == "__main__":
    main()
