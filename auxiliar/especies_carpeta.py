from collections import Counter
from pathlib import Path


EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_ROOT = BASE_DIR / "data" / "imagenes_visilab(raw)"
SALIDA = Path(__file__).with_name("conteo_especies_por_carpeta.txt")


def normalizar_especie(nombre: str) -> str:
    return " ".join(nombre.replace("_", " ").split())


def contar_imagenes(carpeta: Path) -> int:
    return sum(
        1
        for archivo in carpeta.rglob("*")
        if archivo.is_file() and archivo.suffix.lower() in EXTENSIONES
    )


def carpetas_de_especies(carpeta_principal: Path) -> list[Path]:
    subcarpetas = sorted(
        (carpeta for carpeta in carpeta_principal.iterdir() if carpeta.is_dir()),
        key=lambda carpeta: carpeta.name.lower(),
    )
    return subcarpetas or [carpeta_principal]


def main() -> None:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"No existe la carpeta: {DATASET_ROOT}")

    lineas: list[str] = []
    carpetas_principales = sorted(
        (carpeta for carpeta in DATASET_ROOT.iterdir() if carpeta.is_dir()),
        key=lambda carpeta: carpeta.name.lower(),
    )

    for carpeta_principal in carpetas_principales:
        conteo = Counter()
        for carpeta_especie in carpetas_de_especies(carpeta_principal):
            cantidad = contar_imagenes(carpeta_especie)
            if cantidad:
                conteo[normalizar_especie(carpeta_especie.name)] += cantidad

        lineas.append(f"=== {carpeta_principal.name} ===")
        lineas.extend(
            f"{especie}: {cantidad}"
            for especie, cantidad in sorted(
                conteo.items(), key=lambda item: (-item[1], item[0].lower())
            )
        )
        lineas.append(f"Total: {sum(conteo.values())}")
        lineas.append("")

    SALIDA.write_text("\n".join(lineas), encoding="utf-8")
    print("\n".join(lineas))
    print(f"Resultado guardado en: {SALIDA}")


if __name__ == "__main__":
    main()
