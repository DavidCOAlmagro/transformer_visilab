# auxiliar/detectar_duplicados.py
"""
Detecta imágenes duplicadas (por contenido, no solo por nombre) entre
varias carpetas de dataset, para comprobar si el dataset que te ha
pasado alguien se solapa con el que ya tienes.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

base_dir = Path(__file__).resolve().parent.parent
dataset_root = base_dir / "data" / "imagenes_visilab(raw)"
cache_hashes = Path(__file__).with_name("hashes_imagenes.json")


def calcular_hash(ruta: Path, tamano_bloque: int = 65536) -> str:
    """Calcula el hash MD5 del CONTENIDO del archivo (no del nombre),
    para detectar duplicados reales aunque tengan nombres distintos."""
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


def cargar_cache() -> dict[str, dict[str, int | str]]:
    if not cache_hashes.exists():
        return {}
    try:
        return json.loads(cache_hashes.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(
            f"[AVISO] No se pudo leer el caché; se recalcularán los hashes: {cache_hashes}")
        return {}


def obtener_hash(ruta: Path, cache: dict[str, dict[str, int | str]]) -> str:
    clave = str(ruta.resolve())
    estado = ruta.stat()
    entrada = cache.get(clave)
    if entrada and entrada.get("tamano") == estado.st_size and entrada.get("modificado_ns") == estado.st_mtime_ns:
        return str(entrada["hash"])

    hash_archivo = calcular_hash(ruta)
    cache[clave] = {
        "tamano": estado.st_size,
        "modificado_ns": estado.st_mtime_ns,
        "hash": hash_archivo,
    }
    return hash_archivo


def main() -> None:
    hash_a_rutas: dict[str, list[Path]] = defaultdict(list)
    nombre_a_rutas: dict[str, list[Path]] = defaultdict(list)
    cache = cargar_cache()

    if not dataset_root.exists():
        raise FileNotFoundError(f"No existe la carpeta: {dataset_root}")
    carpetas_a_comparar = sorted(
        (carpeta for carpeta in dataset_root.iterdir() if carpeta.is_dir()),
        key=lambda carpeta: carpeta.name.lower(),
    )

    total_imagenes = 0
    for carpeta in carpetas_a_comparar:
        if not carpeta.exists():
            print(f"[AVISO] No existe: {carpeta}")
            continue
        imagenes = recorrer_imagenes(carpeta)
        print(f"Procesando {carpeta.name}: {len(imagenes)} imágenes...")
        total_imagenes += len(imagenes)

        for ruta in imagenes:
            nombre_a_rutas[ruta.name].append(ruta)
            hash_a_rutas[obtener_hash(ruta, cache)].append(ruta)

    cache_hashes.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    duplicados_contenido = {h: rutas for h,
                            rutas in hash_a_rutas.items() if len(rutas) > 1}
    duplicados_nombre = {n: rutas for n,
                         rutas in nombre_a_rutas.items() if len(rutas) > 1}

    lineas: list[str] = []
    lineas.append(f"Total imágenes procesadas: {total_imagenes}")
    lineas.append(
        f"Duplicados EXACTOS por contenido: {len(duplicados_contenido)} grupos")
    lineas.append(
        f"Coincidencias de NOMBRE (pueden ser o no la misma imagen): {len(duplicados_nombre)} grupos")
    lineas.append("")

    lineas.append("=== DUPLICADOS POR CONTENIDO (mismo archivo, fiable) ===")
    for h, rutas in duplicados_contenido.items():
        lineas.append(f"\nHash {h}:")
        for ruta in rutas:
            lineas.append(f"  {ruta}")

    lineas.append(
        "\n\n=== COINCIDENCIAS DE NOMBRE (revisar a mano, puede ser falso positivo) ===")
    for nombre, rutas in duplicados_nombre.items():
        lineas.append(f"\n{nombre}:")
        for ruta in rutas:
            lineas.append(f"  {ruta}")

    salida = Path(__file__).with_name("duplicados_detectados.txt")
    salida.write_text("\n".join(lineas), encoding="utf-8")

    print(f"\n{lineas[1]}")
    print(f"{lineas[2]}")
    print(f"Detalle guardado en: {salida}")


if __name__ == "__main__":
    main()
