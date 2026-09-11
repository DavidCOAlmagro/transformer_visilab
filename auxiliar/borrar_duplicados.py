# auxiliar/eliminar_duplicados.py
"""
Elimina las copias exactas duplicadas (mismo contenido) entre las carpetas
de dataset_aq_dbo5_agrupado, Common_species y Unique_species, quedándose
con UNA copia por cada imagen repetida.

Criterio de qué copia conservar (en este orden):
  1. Si el grupo incluye una copia en dataset_aq_dbo5_agrupado, se conserva esa.
  2. Si no, se conserva la primera por orden alfabético de ruta.

Por defecto corre en modo SIMULACIÓN (no borra nada, solo dice qué borraría).
Cambia DRY_RUN a False para borrar de verdad.
"""
import hashlib
from collections import defaultdict
from pathlib import Path

DRY_RUN = False  # <-- en False para borrar de verdad

extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

carpetas_a_comparar = [
    Path(__file__).resolve().parent.parent / "data" / "imagenes_visilab(raw)" / "Common_species",
    Path(__file__).resolve().parent.parent / "data" / "imagenes_visilab(raw)" / "dataset_aq_dbo5_agrupado",
    Path(__file__).resolve().parent.parent / "data" / "imagenes_visilab(raw)" / "Unique_species",
    Path(__file__).resolve().parent.parent / "data" / "imagenes_visilab(raw)" / "dataset_aq_dbo5_last_agrupado",
    
]   

CARPETA_PREFERIDA = "dataset_aq_dbo5_agrupado"


def calcular_hash(ruta: Path, tamano_bloque: int = 65536) -> str:
    hasher = hashlib.md5()
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(tamano_bloque), b""):
            hasher.update(bloque)
    return hasher.hexdigest()


def recorrer_imagenes(carpeta: Path) -> list[Path]:
    return [
        archivo for archivo in carpeta.rglob("*")
        if archivo.is_file() and archivo.suffix.lower() in extensiones
    ]


def elegir_a_conservar(rutas: list[Path]) -> Path:
    """De un grupo de rutas duplicadas, decide cuál se conserva."""
    preferidas = [r for r in rutas if CARPETA_PREFERIDA in r.parts]
    if preferidas:
        return sorted(preferidas)[0]
    return sorted(rutas)[0]


def main() -> None:
    hash_a_rutas: dict[str, list[Path]] = defaultdict(list)

    total_imagenes = 0
    for carpeta in carpetas_a_comparar:
        if not carpeta.exists():
            print(f"[AVISO] No existe: {carpeta}")
            continue
        imagenes = recorrer_imagenes(carpeta)
        print(f"Procesando {carpeta.name}: {len(imagenes)} imágenes...")
        total_imagenes += len(imagenes)
        for ruta in imagenes:
            hash_a_rutas[calcular_hash(ruta)].append(ruta)

    duplicados = {h: rutas for h, rutas in hash_a_rutas.items() if len(rutas) > 1}

    lineas: list[str] = []
    a_borrar: list[Path] = []

    for h, rutas in duplicados.items():
        conservar = elegir_a_conservar(rutas)
        borrar = [r for r in rutas if r != conservar]
        a_borrar.extend(borrar)

        lineas.append(f"Hash {h}:")
        lineas.append(f"  CONSERVAR: {conservar}")
        for r in borrar:
            lineas.append(f"  borrar:    {r}")
        lineas.append("")

    print(f"\nTotal imágenes procesadas: {total_imagenes}")
    print(f"Grupos duplicados: {len(duplicados)}")
    print(f"Archivos a eliminar: {len(a_borrar)}")

    if DRY_RUN:
        print("\n[MODO SIMULACIÓN] No se ha borrado nada. Revisa el log y pon DRY_RUN = False para borrar de verdad.")
    else:
        for ruta in a_borrar:
            ruta.unlink()
        print(f"\n{len(a_borrar)} archivos eliminados.")

    salida = Path(__file__).with_name("log_eliminacion_duplicados.txt")
    salida.write_text("\n".join(lineas), encoding="utf-8")
    print(f"Log detallado guardado en: {salida}")


if __name__ == "__main__":
    main()