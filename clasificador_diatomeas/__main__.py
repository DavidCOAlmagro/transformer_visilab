"""
--------------------------------------
Para probar el clasificador sin la app:

    python -m clasificador_diatomeas FOTOS SALIDA --modelos CARPETA_MODELOS
--------------------------------------
"""
import argparse
import sys
from pathlib import Path

from .clasificador import Clasificador
from .imagenes import ErrorEntrada
from .modelos import ErrorModelos


def main() -> int:
    """
    Función trabajo principal
    """
    parser = argparse.ArgumentParser(prog="python -m clasificador_diatomeas",
                                     description="Clasifica las diatomeas de una foto o de una carpeta de fotos.")
    parser.add_argument("fotos", type=Path, help="una foto o una carpeta con fotos")
    parser.add_argument("salida", type=Path, help="carpeta donde se guardan resultados.xlsx y las fotos anotadas")
    parser.add_argument("--modelos", type=Path, default=Path("modelos"), help="carpeta de modelos (por defecto: modelos)")
    parser.add_argument("--sin-anotadas", action="store_true", help="no guardar las fotos anotadas")
    args = parser.parse_args()

    def mostrar_progreso(hechas: int, total: int, foto: str) -> None:
        print(f"{hechas}/{total}  {foto}", flush=True)

    try:
        clasificador = Clasificador(args.modelos)
        ruta = clasificador.procesar_carpeta(args.fotos, args.salida, anotar_fotos=not args.sin_anotadas,
                                             al_avanzar=mostrar_progreso)
    except ErrorModelos as error:
        print(f"Error en los modelos: {error}", file=sys.stderr)
        return 2
    except ErrorEntrada as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Resultados: {ruta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
