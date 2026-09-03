"""
--------------------------------------
En este archivo se encuentran las funciones que preparan los datos para el clasificador,
como pasar las etiquetas de texto a números, filtrar las clases que se van a usar y
dividir los datos en conjuntos de entrenamiento, validación y prueba.
--------------------------------------
"""

import statistics
import random
import numpy as np
from tqdm import tqdm
import torch
from constantes import VARIABLES_GLOBALES
from pathlib import Path
import json
from datetime import datetime

def get_datos(nombre_split: str) -> dict[str, torch.Tensor]:
    """
    Devuelve los embeddings y las etiquetas de las imágenes,
    según si es de entrenamiento, val o test.
    """
    ruta_embeddings = VARIABLES_GLOBALES["RUTA_EMBEDDINGS"] / \
        f"embeddings_{nombre_split}.pt"
    datos: dict[str, torch.Tensor] = torch.load(ruta_embeddings, 
                                                weights_only=True)

    return datos


def codificacion(
        datos: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """
    Convierte las etiquetas de texto de las especies filtradas en números
    correlativos (0, 1, 2, ...). Devuelve los embeddings, las etiquetas numéricas y especie -> num.
    """
    # Mapeo especie -> número
    especies_ordenadas: list[str] = sorted(
        VARIABLES_GLOBALES["ESPECIES_FILTRADAS"])
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


def rutas_imagenes(sin_filtro: bool = False) -> list[tuple[str, str]]:
    """
    Recorre las carpetas con las imágenes y devuelve una lista de tuplas (ruta_completa, especie).
    Por defecto, solo incluye las especies listadas en ESPECIES_FILTRADAS (comportamiento
    original). Si sin_filtro=True, se ignora ESPECIES_FILTRADAS y se recorren TODAS las
    carpetas de especie encontradas, útil para contar imágenes antes de decidir qué
    especies incluir en un experimento.
    """
    imagenes: list[tuple[str, str]] = []
    grupos = VARIABLES_GLOBALES["GRUPOS_DATOS"]

    for grupo in grupos:
        ruta_grupo = VARIABLES_GLOBALES["RUTA_BASE"] / "imagenes_visilab(raw)" / grupo
            
        print(f"Recorriendo {ruta_grupo}...")
        if ruta_grupo.exists():

            if sin_filtro:
                especies = [ruta for ruta in ruta_grupo.iterdir() if ruta.is_dir()]
            else:
                especies = [ruta for ruta in ruta_grupo.iterdir() if ruta.is_dir()
                            and ruta.name in VARIABLES_GLOBALES["ESPECIES_FILTRADAS"]]

            for especie in tqdm(especies, desc=f"Recorriendo {grupo}"):
                for archivo in especie.iterdir():
                    if archivo.suffix.lower() in VARIABLES_GLOBALES["EXTENSIONES_VALIDAS"]:
                        imagenes.append((archivo, especie.name))

    return imagenes

def contar_especies_disponibles() -> dict[str, int]:
    """
    Cuenta cuántas imágenes hay de cada especie, sin aplicar ESPECIES_FILTRADAS,
    recorriendo todas las fuentes disponibles (Visilab + UDE). Sirve de base
    para decidir, desde main.py, qué especies incluir en un experimento.
    """
    imagenes = rutas_imagenes(sin_filtro=True)
    return calcular_conteo_por_especie(imagenes)

def contar_clases_train(et_train: torch.Tensor, numero_especie: dict[str, int]) -> None:
    """
    Cuenta cuántas imágenes de train hay por cada especie de
    ESPECIES_FILTRADAS, para detectar si alguna se ha quedado
    con 0 muestras (lo que provocaría nan en la loss por división
    entre cero al calcular los pesos de clase).
    """

    especie_numero: dict[int, str] = {}
    for especie, numero in numero_especie.items():
        especie_numero[numero] = especie
    # torch.bincount() devuelve un tensor con el conteo de cada número en et_train.
    conteo = torch.bincount(et_train, minlength=len(numero_especie))

    for numero, cantidad in enumerate(conteo):
        especie = especie_numero[numero]
        print(f"{especie:40s} {cantidad.item()} imágenes en train")
        if cantidad.item() == 0:
            raise ValueError(
                f"La especie '{especie}' tiene 0 imágenes en train. "
                "Revisa que el nombre coincide exactamente con el de la carpeta.")


def calcular_conteo_por_especie(imagenes: list[tuple[Path, str]]) -> dict[str, int]:
    """
    Cuenta cuántas imágenes originales hay de cada especie en la lista dada
    (antes de aplicar ningún augmentation).
    """
    conteo_por_especie: dict[str, int] = {}
    for _, especie in imagenes:
        conteo_por_especie[especie] = conteo_por_especie.get(especie, 0) + 1
    return conteo_por_especie


def calcular_copias_extra_por_especie(
        conteo_por_especie: dict[str, int], max_copias: int = 5) -> dict[str, int]:
    """
    Calcula cuántas copias extra de augmentation le corresponden a cada especie,
    de forma continua según su frecuencia relativa respecto la mediana de todas las
    especies. Las especies con tantas imágenes como la mediana o más no reciben
    copias extra. Las más minoritarias reciben más copias, hasta max_copias.
    """
    mediana: float = statistics.median(conteo_por_especie.values())

    copias_por_especie: dict[str, int] = {}
    for especie, conteo in conteo_por_especie.items():
        copias_ideales: float = (mediana / conteo) - 1
        copias: int = max(0, min(max_copias, round(copias_ideales)))
        copias_por_especie[especie] = copias

    return copias_por_especie


def obtener_genero(especie: str) -> str:
    """Extrae el género de UNA SOLA especie a partir de la primera palabra antes del '_'."""
    genero: str = especie.split("_")[0]
    return genero


def construir_numero_genero(especies_filtradas: set[str]) -> dict[str, int]:
    """Mapea cada género presente en ESPECIES_FILTRADAS a un índice numérico."""
    generos_desordenados: list[str] = []
    for especie in especies_filtradas:
        genero: str = obtener_genero(especie)
        generos_desordenados.append(genero)
    generos_ordenados = sorted(set(generos_desordenados))
    mapeado: dict[str, int] = {}
    for i, genero in enumerate(generos_ordenados):
        mapeado[genero] = i
    return mapeado


def etiquetas_a_generos(
        etiquetas_especie: torch.Tensor, numero_especie: dict[str, int],
        numero_genero: dict[str, int]) -> torch.Tensor:
    """
    Convierte un tensor de etiquetas de especie en un tensor de etiquetas de
    género, manteniendo el mismo orden de las muestras.Invierte numero_especie 
    para pasar de número->especie en vez de especie->número. Finalmente,
    convierte el género a su índice numérico 
    """
    especie_numero: dict[int, str] = {}

    for especie, numero in numero_especie.items():
        especie_numero[numero] = especie   # aquí SÍ rellenamos el diccionario

    etiquetas_genero: list[int] = []

    for idx in etiquetas_especie:
        numero = idx.item()
        especie = especie_numero[numero]
        genero = obtener_genero(especie)
        numero_del_genero = numero_genero[genero]
        etiquetas_genero.append(numero_del_genero)

    etiquetas_genero_tensor = torch.tensor(etiquetas_genero, dtype=torch.long)

    return etiquetas_genero_tensor

def construir_especies_por_genero(
        numero_especie: dict[str, int], numero_genero: dict[str, int]) -> dict[int, list[int]]:
    """
    Mapea cada índice de género al listado de índices de especie (globales)
    que le pertenecen. Necesario para que el clasificador sepa qué cabeza
    usar y cómo traducir sus salidas locales a índices globales de especie.
    Se deriva automáticamente del nombre de cada especie.
    Devuelve un diccionario: índice de género -> lista de índices
    (Achnanthidium: Achnanthidium_minutissimum, Achnanthidium_parvulum, ...)
    """
    especies_por_genero: dict[int, list[int]] = {}
    for especie, indice_especie in numero_especie.items():
        genero: str = obtener_genero(especie)
        indice_genero: int = numero_genero[genero]
        # Si clave no existe, se crea la lista y se guarda en el diccionario. Y luego se 
        # añade el índice de especie a la lista correspondiente.
        especies_por_genero.setdefault(indice_genero, []).append(indice_especie)
    return especies_por_genero

def fijar_semilla(semilla: int) -> None:
    """
    Configura una semilla para que los resultados no cambien.
    """
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    # Evita que la inicialización de pesos y el shuffle del dataloader sean aleatorios
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def guardar_resumen_entrenamiento(ruta_modelo: Path,historial_macro_f1_val: list[float],
        metricas_test: dict[str, float]) -> None:
    """Guarda las métricas principales del último entrenamiento."""
    if not historial_macro_f1_val:
        raise ValueError("El historial de macro F1 de validación está vacío. No se puede guardar el resumen.")

    indice_mejor = max(range(len(historial_macro_f1_val)),key=lambda indice: historial_macro_f1_val[indice])

    resumen = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "mejor_epoca": indice_mejor + 1,
        "mejor_macro_f1_validacion": historial_macro_f1_val[indice_mejor],
        **metricas_test
    }

    ruta_resumen = ruta_modelo.parent / "resumen_entrenamiento.json"
    if ruta_resumen.is_file():
        with open(ruta_resumen, "r", encoding="utf-8") as archivo:
            contenido = json.load(archivo)
        historial_resumenes = contenido if isinstance(contenido, list) else [contenido]
    else:
        historial_resumenes = []

    historial_resumenes.append(resumen)

    with open(ruta_resumen, "w", encoding="utf-8") as archivo:
        json.dump(historial_resumenes, archivo, indent=4, ensure_ascii=False)

    print(f"Resumen guardado en: {ruta_resumen}")