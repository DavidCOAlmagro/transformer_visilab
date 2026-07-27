"""
--------------------------------------
Lista las imágenes del conjunto de test que el modelo clasifica mal,
junto con su ruta, la especie real y la especie predicha.
Sirve para revisar a mano si el error es del modelo o de un
etiquetado incorrecto en el dataset original (recortes con varias
diatomeas donde se asumió una única especie).
--------------------------------------
"""

from pathlib import Path
import torch

from constantes import VARIABLES_GLOBALES
from clasificador import ClasificadorDiatomeas
from preparar_datos import get_datos, codificacion
from generar_leer_splits import leer_split


def obtener_rutas_test() -> list[str]:
    """
    Devuelve la lista de rutas de imagen de test, en el mismo orden
    en que se calcularon sus embeddings (test no tiene augmentation,
    así que el orden se conserva 1 a 1 respecto a leer_split).
    """
    ruta_test_txt = VARIABLES_GLOBALES["RUTA_SPLITS"] / "test.txt"
    rutas_especies = leer_split(ruta_test_txt)
    rutas = [ruta for ruta, _especie in rutas_especies]
    return rutas


def comprobar_orden(rutas: list[str], etiquetas_guardadas: list[str]) -> None:
    """
    Verifica que el orden de las rutas leídas de test.txt coincide con
    el orden de las etiquetas guardadas en embeddings_test.pt. Si no
    coincide, el mapeo ruta -> predicción no sería fiable.
    """
    especies_desde_rutas = [Path(ruta).parent.name for ruta in rutas]
    if especies_desde_rutas != etiquetas_guardadas:
        raise ValueError(
            "El orden de test.txt no coincide con el de embeddings_test.pt. "
            "Regenera los embeddings de test antes de usar este script.")


def cargar_modelo() -> ClasificadorDiatomeas:
    """Carga el modelo entrenado con los pesos guardados en disco."""
    num_clases = len(VARIABLES_GLOBALES["ESPECIES_FILTRADAS"])
    numero_genero = construir_numero_genero(VARIABLES_GLOBALES["ESPECIES_FILTRADAS"])
    num_generos = len(numero_genero)
    modelo = ClasificadorDiatomeas(num_clases, num_generos).to(VARIABLES_GLOBALES["DEVICE"])

    ruta_pesos=VARIABLES_GLOBALES["RUTA_MODELOS"]/VARIABLES_GLOBALES["PRUEBA"] / "mejor_modelo.pth"
    pesos = torch.load(ruta_pesos, map_location=VARIABLES_GLOBALES["DEVICE"],weights_only=True)
    modelo.load_state_dict(pesos)
    modelo.eval()

    return modelo


@torch.no_grad()
def predecir(modelo: ClasificadorDiatomeas, embeddings: torch.Tensor) -> list[int]:
    """Pasa todos los embeddings por el modelo y devuelve el índice predicho de cada uno."""
    embeddings = embeddings.to(next(modelo.parameters()).device)
    logits_especie, logits_genero = modelo(embeddings)
    _, indices_predichos = torch.max(logits_especie, 1)
    return indices_predichos.cpu().tolist()


def listar_errores(
        rutas: list[str], y_true: list[int], y_pred: list[int],
        numero_especie: dict[str, int]) -> list[tuple[str, str, str]]:
    """
    Compara predicciones con etiquetas reales y devuelve, para cada
    imagen mal clasificada, una tupla (ruta, especie_real, especie_predicha).
    """
    especie_numero = {numero: especie for especie, numero in numero_especie.items()}

    errores: list[tuple[str, str, str]] = []
    for ruta, real, predicha in zip(rutas, y_true, y_pred):
        if real != predicha:
            errores.append((ruta, especie_numero[real], especie_numero[predicha]))

    return errores


def main() -> None:
    """
    Carga test, ejecuta el modelo y muestra por consola cada imagen mal
    clasificada junto con su especie real y la predicha por el modelo.
    """
    rutas = obtener_rutas_test()
    datos_test = get_datos("test")

    comprobar_orden(rutas, datos_test["etiquetas"])

    emb_test, et_test, numero_especie = codificacion(datos_test)

    modelo = cargar_modelo()
    y_pred = predecir(modelo, emb_test)
    y_true = et_test.tolist()

    errores = listar_errores(rutas, y_true, y_pred, numero_especie)

    print(f"Total de errores en test: {len(errores)}\n")
    for ruta, especie_real, especie_predicha in errores:
        print(f"Real: {especie_real:35s} -> Predicha: {especie_predicha:35s} | {ruta}")


if __name__ == "__main__":
    main()
