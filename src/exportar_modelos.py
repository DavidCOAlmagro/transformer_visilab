"""
--------------------------------------
Prepara la carpeta de modelos que usa clasificador_diatomeas y que se
entrega aparte:

    <salida>/yolo.pt        detector de diatomeas
    <salida>/dinov2.pth     DINOv2 ajustado + MLP (el .pth del entrenamiento)
    <salida>/clases.txt     las especies, una por línea, en el orden del entrenamiento

No sobrescribe una carpeta que ya exista.

Uso:
    python src/exportar_modelos.py --dino modelos/77_campo/modelo_77_campo.pth --salida dist/modelos
--------------------------------------
"""
import argparse
import json
import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
YOLO = RAIZ / "Inferir" / "yolo_dinov2" / "yolo_best.pt"
CLASES_DINO = RAIZ / "Inferir" / "txt_classes" / "classes_77(dino).txt"


def clases_del_modelo(pesos: Path) -> list[str]:
    """Las del metadatos_modelo.json del entrenamiento; si no está, la lista de Inferir/txt_classes."""
    metadatos = pesos.parent / "metadatos_modelo.json"
    if metadatos.is_file():
        return json.loads(metadatos.read_text(encoding="utf-8"))["especies_filtradas"]
    return CLASES_DINO.read_text(encoding="utf-8").split()


def main() -> None:
    parser = argparse.ArgumentParser(description="Carpeta de modelos para clasificador_diatomeas.")
    parser.add_argument("--dino", type=Path, default=RAIZ / "modelos" / "77_campo" / "modelo_77_campo.pth")
    parser.add_argument("--salida", type=Path, default=RAIZ / "dist" / "modelos")
    args = parser.parse_args()

    if args.salida.exists():
        raise SystemExit(f"{args.salida} ya existe: bórrala o elige otra --salida.")
    if not args.dino.is_file():
        raise SystemExit(f"No existe {args.dino}")

    args.salida.mkdir(parents=True)
    shutil.copy2(YOLO, args.salida / "yolo.pt")
    shutil.copy2(args.dino, args.salida / "dinov2.pth")
    (args.salida / "clases.txt").write_text("\n".join(clases_del_modelo(args.dino)) + "\n", encoding="utf-8")
    print(f"Modelos en {args.salida}")


if __name__ == "__main__":
    main()
