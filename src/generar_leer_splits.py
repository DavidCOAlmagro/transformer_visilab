"""
--------------------------------------
Genera y guarda en disco un split fijo de imágenes en train/val/test,
estratificado por especie, para que sea reproducible y no cambie
cada vez que se ejecute el programa.
--------------------------------------
"""
from pathlib import Path
from sklearn.model_selection import train_test_split
from preparar_datos import rutas_imagenes,obtener_especies_activas
from collections import Counter
from constantes import VARIABLES_GLOBALES

def generar_split() -> None:
    """
    Divide las imágenes en train/val/test (70/15/15), estratificado por
    especie, y guarda las rutas resultantes en tres archivos de texto
    dentro de data/splits/.
    """
    imagenes: list[tuple[Path, str]] = rutas_imagenes()

    rutas: list[str] = [str(ruta) for ruta, especie in imagenes]
    especies: list[str] = [especie for ruta, especie in imagenes]
    conteo_por_especie = Counter(especies)

    especies_faltantes = sorted(obtener_especies_activas() - set(conteo_por_especie))
    
    if especies_faltantes:
        print(f"Advertencia: las siguientes especies filtradas no tienen imágenes: {especies_faltantes}")
   
    for especie, cantidad in sorted(conteo_por_especie.items()):
        if cantidad < VARIABLES_GLOBALES["MINIMO_IMAGENES_POR_ESPECIE"]:
            print(
                f"Advertencia: {especie} solo tiene {cantidad} imágenes "
                f"(mínimo recomendado: {VARIABLES_GLOBALES['MINIMO_IMAGENES_POR_ESPECIE']})."
            )
    rutas_train, rutas_temp, _, especies_temp = train_test_split(
        rutas, especies,
        test_size=0.30,
        stratify=especies,
        random_state=42
    )

    rutas_val, rutas_test, _, _ = train_test_split(
        rutas_temp, especies_temp,
        test_size=0.50,
        stratify=especies_temp,
        random_state=42
    )

    ruta_splits = VARIABLES_GLOBALES["RUTA_SPLITS"]
    ruta_splits.mkdir(parents=True, exist_ok=True)

    guardar_split(rutas_train, ruta_splits / "train.txt")
    guardar_split(rutas_val, ruta_splits / "val.txt")
    guardar_split(rutas_test, ruta_splits / "test.txt")

    print(f"Train: {len(rutas_train)} imágenes")
    print(f"Val:   {len(rutas_val)} imágenes")
    print(f"Test:  {len(rutas_test)} imágenes")


def guardar_split(rutas: list[str], ruta_archivo: Path) -> None:
    """Guarda una lista de rutas de imagen en un archivo de texto, una por línea."""
    with open(ruta_archivo, "w", encoding="utf-8") as archivo:
        for ruta in rutas:
            archivo.write(ruta + "\n")

def leer_split(ruta_archivo: Path) -> list[tuple[str, str]]:
    """
    Lee un archivo de split (train.txt, val.txt o test.txt) y devuelve
    una lista de tuplas (ruta_imagen, especie), reconstruyendo la especie
    a partir del nombre de la carpeta padre de cada ruta.
    """
    rutas_especies: list[tuple[str, str]] = []

    with open(ruta_archivo, "r", encoding="utf-8") as archivo:
        # Lee cada linea(Cada ruta), sin espacios
        for linea in archivo:
            ruta = linea.strip()
            # Si no está vacía, obtiene la especie a partir del nombre de la carpeta padre
            if ruta:
                especie = Path(ruta).parent.name
                rutas_especies.append((ruta, especie))

    return rutas_especies
if __name__ == "__main__":
    generar_split()
