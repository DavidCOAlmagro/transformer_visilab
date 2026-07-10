from collections import Counter
from pathlib import Path
import zipfile

base_dir = Path(r"C:\Users\david\OneDrive - Universidad de Castilla-La Mancha\Escritorio\siu\VISILAB\transformer\proyecto_transformer\data\imagenes_visilab(raw)")

extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def contar_imagenes_en_carpeta(carpeta: Path) -> int:
    return sum(
        1
        for archivo in carpeta.rglob("*")
        if archivo.is_file() and archivo.suffix.lower() in extensiones
    )


def contar_imagenes_en_zip(zip_path: Path) -> Counter:
    contador = Counter()
    with zipfile.ZipFile(zip_path) as archivo_zip:
        for nombre in archivo_zip.namelist():
            ruta_relativa = Path(nombre)
            if nombre.endswith("/") or ruta_relativa.suffix.lower() not in extensiones:
                continue
            if ruta_relativa.parts:
                contador[ruta_relativa.parts[0]] += 1
            else:
                contador[zip_path.stem] += 1
    return contador


if not base_dir.exists():
    raise FileNotFoundError(f"No existe la carpeta: {base_dir}")

total_por_carpeta = Counter()

for elemento in sorted(base_dir.iterdir(), key=lambda p: p.name.lower()):
    if elemento.is_dir():
        total_por_carpeta[elemento.name] += contar_imagenes_en_carpeta(
            elemento)
    elif elemento.suffix.lower() == ".zip":
        total_por_carpeta.update(contar_imagenes_en_zip(elemento))

for nombre_carpeta, cantidad in sorted(total_por_carpeta.items()):
    print(f"{nombre_carpeta}: {cantidad}")

print(f"Total general: {sum(total_por_carpeta.values())}")
