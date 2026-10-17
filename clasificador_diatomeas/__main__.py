"""
--------------------------------------
Para probar el clasificador sin la app:

    python -m clasificador_diatomeas carpeta_fotos carpeta_resultados --modelos modelos

Clasifica todas las fotos de la carpeta y guarda resultados.xlsx y las fotos anotadas.
--------------------------------------
"""
import argparse
from pathlib import Path

from .clasificador import Clasificador

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description="Clasifica las diatomeas de una carpeta de fotos.")
    parser.add_argument("fotos", type=Path)
    parser.add_argument("salida", type=Path)
    parser.add_argument("--modelos", type=Path, default=Path("modelos"))
    args = parser.parse_args()

    clasificador = Clasificador(args.modelos)
    (args.salida / "anotadas").mkdir(parents=True, exist_ok=True)

    fotos = sorted(f for f in args.fotos.rglob("*") if f.suffix.lower() in EXTENSIONES)
    resultados = []
    for i, foto in enumerate(fotos, start=1):
        print(f"{i}/{len(fotos)}  {foto.name}")
        resultado = clasificador.clasificar(foto)
        resultado.imagen = foto.relative_to(args.fotos).as_posix()  # con la subcarpeta, si la hay
        resultados.append(resultado)
        if resultado.estado != "error":
            nombre = "__".join(foto.relative_to(args.fotos).with_suffix("").parts)
            clasificador.anotar(foto, resultado).save(args.salida / "anotadas" / f"{nombre}.jpg", quality=90)

    clasificador.guardar_excel(resultados, args.salida / "resultados.xlsx")
    print(f"Hecho: {args.salida / 'resultados.xlsx'}")


if __name__ == "__main__":
    main()
