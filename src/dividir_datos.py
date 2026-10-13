"""
--------------------------------------
Genera el reparto train/val/test de las imágenes de entrenamiento como lista
de rutas RELATIVAS a ``data/imagenes_visilab(raw)`` para que sirva en cualquier
máquina:

    recursos/splits_<prueba>.txt.gz   (una línea por imagen: split<TAB>ruta)

Las imágenes que ya están en el reparto BASE (con el que se entrenó el modelo de
partida de --desde-modelo, por defecto splits_75_objetivo_relativos) conservan su
split: si se sorteara de nuevo, dos tercios de val y test serían imágenes que ese
modelo ya vio al entrenar. Las nuevas se reparten 70/15/15 estratificado por
especie (semilla 42). Con --sin-base, todo se sortea de nuevo (entrenar desde cero).

Las especies son las de ESPECIES_FILTRADAS (constantes.py) y los nombres de
carpeta se normalizan (spec 009). La fuente `campo_pool` (recortes de campo,
spec 011) se reparte aparte, solo en train/val y sin partir ninguna foto: el
test interno sigue siendo de laboratorio y el de campo es recursos/campo_test.txt.

Uso:
    python src/dividir_datos.py --prueba 77_campo
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
BASE = RAIZ / "recursos" / "splits_75_objetivo_relativos.txt.gz"
MINIMO_NUEVAS = 10  # una especie con menos imágenes nuevas no se puede estratificar: van a train


def dividir(imagenes: list[tuple[Path, str]], semilla: int = 42) -> dict[str, list[tuple[Path, str]]]:
    """70/15/15 estratificado por especie."""
    especies = [e for _, e in imagenes]
    train, resto = train_test_split(imagenes, test_size=0.30, stratify=especies, random_state=semilla)
    val, test = train_test_split(resto, test_size=0.50, stratify=[e for _, e in resto], random_state=semilla)
    return {"train": train, "val": val, "test": test}


def leer_base(ruta: Path) -> dict[str, str]:
    """{ruta relativa: split} del reparto con el que se entrenó el modelo de partida."""
    base = {}
    with gzip.open(ruta, "rt", encoding="utf-8") as archivo:
        for linea in archivo:
            split, imagen = linea.rstrip("\n").split("\t", 1)
            base[imagen] = split
    return base


def dividir_con_base(imagenes: list[tuple[Path, str]], raiz_imagenes: Path, base: dict[str, str],
                     semilla: int = 42) -> dict[str, list[tuple[Path, str]]]:
    """Las imágenes de ``base`` conservan su split; las nuevas se reparten 70/15/15 entre ellas."""
    splits: dict[str, list[tuple[Path, str]]] = {"train": [], "val": [], "test": []}
    nuevas = []
    for item in imagenes:
        split = base.get(item[0].relative_to(raiz_imagenes).as_posix())
        (splits[split] if split else nuevas).append(item)
    conteo = Counter(e for _, e in nuevas)
    repartibles = [item for item in nuevas if conteo[item[1]] >= MINIMO_NUEVAS]
    if repartibles:
        for nombre, items in dividir(repartibles, semilla).items():
            splits[nombre] += items
    splits["train"] += [item for item in nuevas if conteo[item[1]] < MINIMO_NUEVAS]
    return splits


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
    parser.add_argument("--base", type=Path, default=BASE, help="Reparto con el que se entrenó el modelo de partida.")
    parser.add_argument("--sin-base", action="store_true", help="Sortear todo de nuevo (entrenamiento desde cero).")
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
    laboratorio = [item for item, campo in zip(imagenes, es_campo) if not campo]
    if args.sin_base:
        splits = dividir(laboratorio)
    else:
        base = leer_base(args.base)
        claves = {imagen.relative_to(raiz_imagenes).as_posix() for imagen, _ in laboratorio}
        print(f"Reparto base {args.base.name}: {len(claves & base.keys())} conservan su split, "
              f"{len(claves - base.keys())} nuevas, {len(base.keys() - claves)} del base no están en disco")
        splits = dividir_con_base(laboratorio, raiz_imagenes, base)
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
