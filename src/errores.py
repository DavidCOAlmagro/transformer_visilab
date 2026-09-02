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
import torch.nn.functional as F
from constantes import VARIABLES_GLOBALES
from clasificador import ClasificadorDiatomeas
from preparar_datos import get_datos, codificacion, construir_numero_genero,etiquetas_a_generos
from generar_leer_splits import leer_split
from modelo import cargar_modelo_entrenado

PARES_A_REVISAR: set[frozenset[str]] = {
    frozenset({"Nitzschia_inconspicua", "Nitzschia_sp"}),
    frozenset({"Achnanthidium_pyrenaicum", "Achnanthidium_sp"}),
    frozenset({"Achnanthidium_rivulare", "Achnanthidium_sp"}),
    frozenset({"Fistulifera_saprophila", "Mayamaea_permitis"}),
    frozenset({"Navicula_sp", "Navicula_cryptotenella"}),
}

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


@torch.no_grad()
def predecir(modelo: ClasificadorDiatomeas, embeddings: torch.Tensor) -> tuple[list[int], list[float], list[int]]:
    """Pasa todos los embeddings por el modelo y devuelve el índice predicho de cada uno.
    Además, devuelve el indice del genero predicho y la probabilidad softmax(0-1)"""
    embeddings = embeddings.to(next(modelo.parameters()).device)
    logits_especie, logits_genero = modelo(embeddings)
    _, indices_predichos_genero = torch.max(logits_genero, dim=1)
    # Calcula la probabilidad softmax de cada clase y obtiene la predicción
    probs_especie = F.softmax(logits_especie, dim=1)
    # Obtiene la confianza de la predicción
    confianza, indices_predichos = torch.max(probs_especie, 1)
    return indices_predichos.cpu().tolist(),confianza.cpu().tolist(),indices_predichos_genero.cpu().tolist()


def listar_errores(
        rutas: list[str], y_true: list[int], y_pred: list[int],
        numero_especie: dict[str, int], confianzas: list[float],
        y_true_genero: list[int], y_pred_genero: list[int],
        pares_a_revisar: set[frozenset[str]]) -> list[tuple[str, str, str]]:
    """
    Para cada imagen mal clasificada, devuelve:
    (ruta, especie_real, especie_predicha, confianza, genero_coincide).
    Si se pasa pares_a_revisar, filtra solo esos pares. Ordena de mayor a
    menor confianza.
    """
    especie_numero = {numero: especie for especie, numero in numero_especie.items()}

    errores: list[tuple[str, str, str]] = []
    for ruta, real, predicha,confianza,genero_real,genero_predicho in zip(
        rutas, y_true, y_pred, confianzas, y_true_genero, y_pred_genero):
        if real != predicha:
            especie_real = especie_numero[real]
            especie_predicha = especie_numero[predicha]
            if pares_a_revisar:
                par = frozenset({especie_real, especie_predicha})
                if par not in pares_a_revisar:
                    continue
            genero_coincide = genero_real == genero_predicho
            errores.append((ruta, especie_real, especie_predicha, confianza, genero_coincide))
    errores.sort(key=lambda x: x[3], reverse=True)  # Ordena por confianza descendente

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
    numero_genero = construir_numero_genero(VARIABLES_GLOBALES["ESPECIES_FILTRADAS"])
    genero_true = etiquetas_a_generos(et_test, numero_especie, numero_genero).tolist()
    
    modelo, _ = cargar_modelo_entrenado()
    y_pred, confianzas, y_pred_genero = predecir(modelo, emb_test)
    y_true = et_test.tolist()

    errores = listar_errores(rutas, y_true, y_pred, numero_especie, confianzas, genero_true, y_pred_genero,pares_a_revisar=PARES_A_REVISAR)

# Ruta de salida del .txt, dentro de la carpeta del experimento actual
    ruta_salida = VARIABLES_GLOBALES["RUTA_MODELOS"] / VARIABLES_GLOBALES["PRUEBA"] / "errores_a_revisar.txt"
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    with open(ruta_salida, "w", encoding="utf-8") as archivo:
        for ruta, especie_real, especie_predicha, confianza, genero_coincide in errores:
            marca_genero = "GEN OK" if genero_coincide else "GEN MAL"
            linea = (f"[{confianza:6.1%}] [{marca_genero}] "
                     f"Real: {especie_real:35s} -> Predicha: {especie_predicha:35s} | {ruta}")
            print(linea)
            archivo.write(linea + "\n")

    print(f"\nGuardado en: {ruta_salida}")


if __name__ == "__main__":
    main()
