"""
--------------------------------------
En este archivo se encuentran las funciones que preparan los datos para el clasificador,
como pasar las etiquetas de texto a números, filtrar las clases que se van a usar y
dividir los datos en conjuntos de entrenamiento, validación y prueba.
--------------------------------------
"""

from tqdm import tqdm
import torch
from constantes import VARIABLES_GLOBALES


def get_datos(nombre_split: str) -> dict[str, torch.Tensor]:
    """
    Devuelve los embeddings y las etiquetas de las imágenes,
    según si es de entrenamiento, val o test.
    """
    ruta_embeddings = VARIABLES_GLOBALES["RUTA_EMBEDDINGS"]/ f"embeddings_{nombre_split}.pt"
    datos: dict[str, torch.Tensor] = torch.load(ruta_embeddings)

    return datos


def codificacion(
        datos: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """
    Convierte las etiquetas de texto de las especies filtradas en números
    correlativos (0, 1, 2, ...). Devuelve los embeddings, las etiquetas numéricas y especie -> num.
    """
    # Mapeo especie -> número
    especies_ordenadas: list[str] = sorted(VARIABLES_GLOBALES["ESPECIES_FILTRADAS"])
    numero_especie: dict[str, int] = {}
    for i, especie in enumerate(especies_ordenadas):
        numero_especie[especie] = i

    embeddings_filtrados = []
    especies_numericas = []

    # Filtrar embeddings y etiquetas simultáneamente
    for emb, especie in zip(datos["embeddings"], datos["etiquetas"]):
        if especie in numero_especie:
            embeddings_filtrados.append(emb)
            especies_numericas.append(numero_especie[especie])

    # Convertir a tensores
    embeddings_tensor = torch.stack(embeddings_filtrados)
    etiquetas_tensor = torch.tensor(especies_numericas, dtype=torch.long)

    return embeddings_tensor, etiquetas_tensor, numero_especie

def rutas_imagenes() -> list[tuple[str, str]]:
    """
    Recorre las carpetas con las imágenes y devuelve una lista de tuplas (ruta_completa, especie).
    Las funciones usadas en este metodo son de pathlib.Path
    """
    imagenes: list[tuple[str, str]] = []
    grupos = ["Common_Species","Unique_Species","Seleccion_5_especies_por_especie","UDE_Diatoms_84k_normalizadas_reinhard"]

    for grupo in grupos:
        ruta_grupo = VARIABLES_GLOBALES["RUTA_BASE"] / "imagenes_visilab(raw)" /grupo
        print(f"Recorriendo {ruta_grupo}...")
        if ruta_grupo.exists():

            especies = [ruta for ruta in ruta_grupo.iterdir() if ruta.is_dir()
                        and ruta.name in VARIABLES_GLOBALES["ESPECIES_FILTRADAS"]]

            # tqdm es una librería que muestra una barra de progreso en la consola
            for especie in tqdm(especies, desc=f"Recorriendo {grupo}"):
                for archivo in especie.iterdir():
                    # La funcion suffix() devuelve la extensión del archivo, incluyendo el punto
                    if archivo.suffix.lower() in VARIABLES_GLOBALES["EXTENSIONES_VALIDAS"]:
                        imagenes.append((archivo, especie.name))

    return imagenes
def contar_clases_train(et_train: torch.Tensor, numero_especie: dict[str, int]) -> None:
    """
    Cuenta cuántas imágenes de train hay por cada especie de
    ESPECIES_FILTRADAS, para detectar si alguna se ha quedado
    con 0 muestras (lo que provocaría nan en la loss por división
    entre cero al calcular los pesos de clase).
    """

    especie_numero = {numero: especie for especie, numero in numero_especie.items()}
    conteo = torch.bincount(et_train, minlength=len(numero_especie))

    for numero, cantidad in enumerate(conteo):
        especie = especie_numero[numero]
        print(f"{especie:40s} {cantidad.item()} imágenes en train")