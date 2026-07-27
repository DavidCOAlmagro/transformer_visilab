from collections import Counter
from pathlib import Path


extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

base_dir = Path(__file__).resolve().parent.parent
dataset_roots = [
    base_dir / "data" /"imagenes_visilab(raw)" / "dataset_aq_dbo5",
    base_dir / "data" /"imagenes_visilab(raw)" / "Common_Species",
    base_dir / "data" /"imagenes_visilab(raw)" / "Seleccion_5_especies_por_especie",
    base_dir / "data" /"imagenes_visilab(raw)" / "UDE_Diatoms_84k_normalizadas_reinhard",
    base_dir / "data" /"imagenes_visilab(raw)" / "Unique_Species"
]


def normalizar_especie(nombre: str) -> str:
    return " ".join(nombre.replace("_", " ").split())


def contar_imagenes_en_carpeta(carpeta: Path) -> int:
    return sum(
        1
        for archivo in carpeta.rglob("*")
        if archivo.is_file() and archivo.suffix.lower() in extensiones
    )


def iterar_carpetas_especie(dataset_root: Path):
    for carpeta_principal in sorted((p for p in dataset_root.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
        subcarpetas = [p for p in carpeta_principal.iterdir() if p.is_dir()]
        if subcarpetas:
            for especie_dir in sorted(subcarpetas, key=lambda p: p.name.lower()):
                yield especie_dir
        else:
            yield carpeta_principal


conteo_por_especie = Counter()

for dataset_root in dataset_roots:
    if not dataset_root.exists():
        raise FileNotFoundError(f"No existe la carpeta: {dataset_root}")

    for especie_dir in iterar_carpetas_especie(dataset_root):
        especie = normalizar_especie(especie_dir.name)
        cantidad = contar_imagenes_en_carpeta(especie_dir)
        if cantidad:
            conteo_por_especie[especie] += cantidad


lineas_salida = [
    f"{especie}: {cantidad}"
    for especie, cantidad in sorted(
        conteo_por_especie.items(),
        key=lambda item: (-item[1], item[0].lower())
    )
]

for linea in lineas_salida:
    print(linea)

salida_txt = Path(__file__).with_name("conteo_especies.txt")
salida_txt.write_text("\n".join(lineas_salida), encoding="utf-8")
