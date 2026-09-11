"""Agrupa las imágenes de los splits test y val por especie.

Configura CARPETA_ENTRADA con una carpeta de imagenes_visilab(raw). Las
imágenes originales no se modifican: se copian a CARPETA_SALIDA/<especie>.
"""
from collections import defaultdict
from pathlib import Path
import shutil


BASE_DIR = Path(__file__).resolve().parent.parent
CARPETA_ENTRADA = BASE_DIR / "data" / \
    "imagenes_visilab(raw)" / "dataset_aq_dbo5_splits"
CARPETA_SALIDA = CARPETA_ENTRADA.parent / \
    f"{CARPETA_ENTRADA.name}_test_val_agrupado"
SPLITS = {"test", "val"}
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def buscar_imagenes_por_especie() -> dict[str, list[Path]]:
    imagenes_por_especie: dict[str, list[Path]] = defaultdict(list)

    for carpeta_split in CARPETA_ENTRADA.rglob("*"):
        if not carpeta_split.is_dir() or carpeta_split.name.lower() not in SPLITS:
            continue

        if carpeta_split.parent == CARPETA_ENTRADA:
            carpetas_especie = [
                carpeta for carpeta in carpeta_split.iterdir()
                if carpeta.is_dir()
            ]
        else:
            carpetas_especie = [carpeta_split]

        for carpeta_especie in carpetas_especie:
            especie = carpeta_especie.name if carpeta_split.parent == CARPETA_ENTRADA else carpeta_split.parent.name
            for imagen in carpeta_especie.rglob("*"):
                if imagen.is_file() and imagen.suffix.lower() in EXTENSIONES:
                    imagenes_por_especie[especie].append(imagen)

    return imagenes_por_especie


def ruta_destino_unica(destino: Path) -> Path:
    if not destino.exists():
        return destino

    contador = 1
    while True:
        candidato = destino.with_name(
            f"{destino.stem}_{contador}{destino.suffix}")
        if not candidato.exists():
            return candidato
        contador += 1


def main() -> None:
    if not CARPETA_ENTRADA.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta de entrada: {CARPETA_ENTRADA}")
    if CARPETA_SALIDA.resolve() == CARPETA_ENTRADA.resolve():
        raise ValueError(
            "La carpeta de salida no puede ser la misma que la de entrada")

    imagenes_por_especie = buscar_imagenes_por_especie()
    if not imagenes_por_especie:
        print(f"No se encontraron carpetas test/val en: {CARPETA_ENTRADA}")
        return

    total = 0
    for especie, imagenes in sorted(imagenes_por_especie.items(), key=lambda item: item[0].lower()):
        carpeta_especie = CARPETA_SALIDA / especie
        carpeta_especie.mkdir(parents=True, exist_ok=True)

        for imagen in sorted(imagenes):
            destino = ruta_destino_unica(carpeta_especie / imagen.name)
            shutil.copy2(imagen, destino)
            total += 1

        print(f"{especie}: {len(imagenes)} imágenes")

    print(f"\nTotal copiadas: {total}")
    print(f"Salida: {CARPETA_SALIDA}")


if __name__ == "__main__":
    main()
