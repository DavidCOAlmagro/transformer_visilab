"""
--------------------------------------
Recorta de las fotos de campo los recortes elegidos en recursos/recortes_campo.csv
(spec 011) y los guarda como una fuente más de entrenamiento:

    data/imagenes_visilab(raw)/campo_pool/<especie>/<grupo>__<foto>__roi<n>.png

Las cajas son las mismas que usó la inferencia (coordenadas enteras del Excel).

Uso (donde estén las fotos, normalmente Ubuntu):
    python3 src/extraer_recortes_campo.py --entrada Inferir/imagenes_inferencia
--------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile

from seleccionar_recortes_campo import nombre_recorte

ImageFile.LOAD_TRUNCATED_IMAGES = True
RAIZ = Path(__file__).resolve().parent.parent
CARPETA_CAMPO = RAIZ / "data" / "imagenes_visilab(raw)" / "campo_pool"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae los recortes de campo elegidos (spec 011).")
    parser.add_argument("--csv", type=Path, default=RAIZ / "recursos" / "recortes_campo.csv")
    parser.add_argument("--entrada", type=Path, default=RAIZ / "Inferir" / "imagenes_inferencia")
    parser.add_argument("--salida", type=Path, default=CARPETA_CAMPO)
    args = parser.parse_args()

    recortes = pd.read_csv(args.csv)
    faltan, hechos = [], 0
    for image, filas in recortes.groupby("image"):
        ruta = args.entrada.joinpath(*image.replace("\\", "/").split("/"))
        if not ruta.is_file():
            faltan.append(image)
            continue
        with Image.open(ruta) as foto:
            foto = foto.convert("RGB")
            for fila in filas.itertuples():
                destino = args.salida / fila.especie / nombre_recorte(image, fila.item, fila.grupo)
                destino.parent.mkdir(parents=True, exist_ok=True)
                foto.crop((fila.x1, fila.y1, fila.x2, fila.y2)).save(destino)
                hechos += 1
    fotos = recortes.image.nunique()
    print(f"{hechos} recortes guardados en {args.salida} | fotos no encontradas: {len(faltan)}/{fotos}")
    if len(faltan) > 0.01 * fotos:
        print("Ejemplos:\n  " + "\n  ".join(faltan[:5]))
        raise SystemExit("Falta más del 1 % de las fotos: revisa --entrada.")


if __name__ == "__main__":
    main()
