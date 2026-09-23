"""
--------------------------------------
Evalúa el mecanismo de detección de especies desconocidas desde varios
ángulos, sin necesidad de reentrenar la red:

1. Rechazo actual: con el umbral de centroides YA calibrado en el .pth,
   qué proporción de imágenes desconocidas (especies fuera de
   ESPECIES_FILTRADAS) se rechazan, y qué proporción de imágenes de test
   (conocidas) se rechazan por error.
2. Calibración de confianza: hueco entre la confianza media del softmax
   en aciertos vs en fallos sobre test. Un hueco pequeño (como el 0.004
   que tenía DINOv2 sin calibrar) significa que la confianza no sirve
   para distinguir aciertos de fallos.
3. Barrido de percentiles: recalcula el umbral de centroides con varios
   percentiles (reutilizando los centros ya calibrados, sin tocar los
   pesos de la red) y mide el trade-off entre falsos rechazos de
   conocidas (en test) y desconocidas pilladas de verdad (en unknown).
   El umbral se calibra siempre sobre val y se mide sobre test/unknown,
   para no medir sobre los mismos datos con los que se calibra.

Al final se restaura el umbral que tenía el modelo al empezar: este
script es solo de diagnóstico, no modifica el .pth.
--------------------------------------
"""
from pathlib import Path

import torch

from constantes import VARIABLES_GLOBALES
from modelo import cargar_modelo_entrenado
from preparar_datos import get_datos, codificacion
from calibracion_confianza import calibrar_centroides

PERCENTILES_A_PROBAR: list[float] = [99, 97.5, 95, 90, 85, 80, 70, 60, 50, 40]


@torch.no_grad()
def predicciones_y_tronco(modelo: torch.nn.Module,
                          embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pasa embeddings por el modelo en batches y devuelve (clase predicha
    por el softmax, embedding del tronco), ambos en CPU. Se calcula una
    sola vez y se reutiliza en todo el script para no repetir pasadas
    por el modelo.
    """
    device = next(modelo.parameters()).device
    logits_lista: list[torch.Tensor] = []
    tronco_lista: list[torch.Tensor] = []
    for lote in embeddings.split(VARIABLES_GLOBALES["BATCH_SIZE"]):
        logits, _, tronco = modelo(lote.to(device))
        logits_lista.append(logits.cpu())
        tronco_lista.append(tronco.cpu())
    logits = torch.cat(logits_lista)
    return logits.argmax(dim=1), torch.cat(tronco_lista)


def medir_rechazo(modelo: torch.nn.Module, pred: torch.Tensor, tronco: torch.Tensor) -> float:
    """Proporción de muestras rechazadas como desconocidas, con el umbral actual del modelo."""
    with torch.no_grad():
        return modelo.es_desconocida(tronco, pred).float().mean().item()


def medir_confianza(modelo: torch.nn.Module, embeddings: torch.Tensor,
                    etiquetas: torch.Tensor) -> tuple[float, float]:
    """Confianza media del softmax en aciertos vs en fallos, sobre el conjunto dado."""
    device = next(modelo.parameters()).device
    with torch.no_grad():
        logits, _, _ = modelo(embeddings.to(device))
        probs = torch.softmax(logits, dim=1)
        confianza, pred = probs.max(dim=1)
    aciertos = pred.cpu() == etiquetas
    conf_aciertos = confianza.cpu()[aciertos].mean().item() if aciertos.any() else float("nan")
    conf_fallos = confianza.cpu()[~aciertos].mean().item() if (~aciertos).any() else float("nan")
    return conf_aciertos, conf_fallos


def barrer_percentiles(
        modelo: torch.nn.Module, centros: torch.Tensor,
        tronco_calibracion: torch.Tensor, et_calibracion: torch.Tensor,
        pred_test: torch.Tensor, tronco_test: torch.Tensor,
        pred_unknown: torch.Tensor, tronco_unknown: torch.Tensor) -> list[str]:
    """
    Para cada percentil de PERCENTILES_A_PROBAR, recalibra el umbral de
    centroides usando (tronco_calibracion, et_calibracion) -normalmente
    val- y mide el resultado sobre test (conocidas) y unknown (especies
    nunca entrenadas). Modifica modelo.umbral_distancia en el proceso;
    quien llame a esta función es responsable de restaurarlo si hace
    falta seguir usando el modelo después.
    """
    cabecera = f"{'percentil':>10} | {'falsos rechazos (conocidas, test)':>34} | {'desconocidas pilladas (unknown)':>32}"
    lineas = [cabecera]
    print(cabecera)
    for percentil in PERCENTILES_A_PROBAR:
        calibrar_centroides(modelo, centros, tronco_calibracion, et_calibracion, percentil=percentil)
        falsos_rechazos = medir_rechazo(modelo, pred_test, tronco_test)
        pilladas = medir_rechazo(modelo, pred_unknown, tronco_unknown)
        linea = f"{percentil:>10} | {falsos_rechazos:>33.2%} | {pilladas:>31.2%}"
        print(linea)
        lineas.append(linea)
    return lineas


def main() -> None:
    modelo, _ = cargar_modelo_entrenado()

    datos_val = get_datos("val")
    emb_val, et_val, _ = codificacion(datos_val)
    datos_test = get_datos("test")
    emb_test, et_test, _ = codificacion(datos_test)
    datos_unknown = get_datos("unknown")

    pred_val, tronco_val = predicciones_y_tronco(modelo, emb_val)
    pred_test, tronco_test = predicciones_y_tronco(modelo, emb_test)
    pred_unknown, tronco_unknown = predicciones_y_tronco(modelo, datos_unknown["embeddings"])

    # --- 1. Estado actual: rechazo con el umbral YA calibrado en el .pth ---
    rechazo_unknown_actual = medir_rechazo(modelo, pred_unknown, tronco_unknown)
    rechazo_conocidas_actual = medir_rechazo(modelo, pred_test, tronco_test)

    # --- 2. Calibración de confianza (softmax), sobre test ---
    conf_aciertos, conf_fallos = medir_confianza(modelo, emb_test, et_test)
    hueco_confianza = conf_aciertos - conf_fallos

    # --- 3. Barrido de percentiles del umbral de centroides, sin reentrenar ---
    umbral_original = modelo.umbral_distancia.clone()
    centros = modelo.centros.clone()
    lineas_barrido = barrer_percentiles(
        modelo, centros, tronco_val, et_val,
        pred_test, tronco_test, pred_unknown, tronco_unknown)
    # Restauramos el umbral con el que llegó el modelo: este script es de
    # diagnóstico, no debe dejar el modelo en memoria con otro percentil.
    with torch.no_grad():
        modelo.umbral_distancia.copy_(umbral_original)

    # --- Informe ---
    ruta_reporte = (
        VARIABLES_GLOBALES["RUTA_MODELOS"] /
        VARIABLES_GLOBALES["PRUEBA"] / "reporte_desconocidas.txt"
    )
    lineas = [
        "Evaluación de rechazo de especies desconocidas",
        "",
        "--- Con el umbral calibrado actualmente en el .pth ---",
        f"Imágenes desconocidas: {len(datos_unknown['etiquetas'])}",
        f"Desconocidas rechazadas: {rechazo_unknown_actual:.2%}",
        f"Imágenes conocidas de test rechazadas: {rechazo_conocidas_actual:.2%}",
        "",
        "--- Calibración de confianza (softmax) en test ---",
        f"Confianza media en aciertos: {conf_aciertos:.4f}",
        f"Confianza media en fallos:   {conf_fallos:.4f}",
        f"Hueco (cuanto mayor, más honesta la confianza): {hueco_confianza:.4f}",
        "",
        "--- Barrido de percentiles del umbral de centroides (sin reentrenar) ---",
        "El umbral de cada fila se calibra sobre val y se mide sobre test/unknown.",
        *lineas_barrido,
    ]
    ruta_reporte.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    print(f"\nDesconocidas rechazadas (umbral actual): {rechazo_unknown_actual:.2%}")
    print(f"Conocidas rechazadas por error (umbral actual): {rechazo_conocidas_actual:.2%}")
    print(f"Hueco de confianza (aciertos - fallos): {hueco_confianza:.4f}")
    print(f"Informe guardado en: {ruta_reporte}")


if __name__ == "__main__":
    main()