"""Evalúa el rechazo de especies que no pertenecen al experimento, la
calibración de la confianza y cómo cambia el rechazo según el percentil
del umbral de centroides. No reentrena ni toca el .pth, solo diagnóstico."""

from pathlib import Path

import torch

from constantes import VARIABLES_GLOBALES
from modelo import cargar_modelo_entrenado
from preparar_datos import get_datos, codificacion
from calibracion_confianza import calibrar_centroides

PERCENTILES = [99, 97.5, 95, 90, 85, 80, 70, 60, 50, 40]


@torch.no_grad()
def predecir(modelo: torch.nn.Module, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pasa embeddings por el modelo y devuelve (clase predicha, embedding del tronco), en CPU."""
    device = next(modelo.parameters()).device
    logits_lista, tronco_lista = [], []
    for lote in embeddings.split(VARIABLES_GLOBALES["BATCH_SIZE"]):
        logits, _, tronco = modelo(lote.to(device))
        logits_lista.append(logits.cpu())
        tronco_lista.append(tronco.cpu())
    return torch.cat(logits_lista).argmax(dim=1), torch.cat(tronco_lista)


def medir_rechazo(modelo: torch.nn.Module, pred: torch.Tensor, tronco: torch.Tensor) -> float:
    """Proporción de embeddings rechazados como desconocidos, con el umbral actual del modelo."""
    return modelo.es_desconocida(tronco, pred).float().mean().item()


def medir_hueco_confianza(modelo: torch.nn.Module, embeddings: torch.Tensor,
                          etiquetas: torch.Tensor) -> tuple[float, float]:
    """Confianza media del softmax en aciertos y en fallos. Si el hueco entre ambas
    es pequeño (como el 0.004 que daba DINOv2 sin calibrar), la confianza no sirve
    para distinguir un acierto de un fallo."""
    device = next(modelo.parameters()).device
    with torch.no_grad():
        logits, _, _ = modelo(embeddings.to(device))
        confianza, pred = torch.softmax(logits, dim=1).max(dim=1)
    aciertos = pred.cpu() == etiquetas
    return confianza.cpu()[aciertos].mean().item(), confianza.cpu()[~aciertos].mean().item()


def main() -> None:
    modelo, _ = cargar_modelo_entrenado()

    datos_val = get_datos("val")
    emb_val, et_val, _ = codificacion(datos_val)
    datos_test = get_datos("test")
    emb_test, et_test, _ = codificacion(datos_test)
    datos_unknown = get_datos("unknown")

    pred_val, tronco_val = predecir(modelo, emb_val)
    pred_test, tronco_test = predecir(modelo, emb_test)
    pred_unknown, tronco_unknown = predecir(modelo, datos_unknown["embeddings"])

    rechazo_unknown = medir_rechazo(modelo, pred_unknown, tronco_unknown)
    rechazo_conocidas = medir_rechazo(modelo, pred_test, tronco_test)
    conf_aciertos, conf_fallos = medir_hueco_confianza(modelo, emb_test, et_test)

    print(f"Desconocidas rechazadas: {rechazo_unknown:.2%}")
    print(f"Conocidas rechazadas por error: {rechazo_conocidas:.2%}")
    print(f"Confianza media en aciertos/fallos: {conf_aciertos:.4f} / {conf_fallos:.4f} "
          f"(hueco: {conf_aciertos - conf_fallos:.4f})")

    # Barrido de percentiles: recalcula el umbral con cada percentil (calibrando
    # sobre val, midiendo sobre test/unknown) sin tocar los pesos de la red.
    # Al final se deja el modelo con el umbral que tenía al empezar.
    umbral_original = modelo.umbral_distancia.clone()
    centros = modelo.centros.clone()

    print(f"\n{'percentil':>10} | {'falsos rechazos (conocidas)':>28} | {'desconocidas pilladas':>22}")
    lineas_barrido = []
    for percentil in PERCENTILES:
        calibrar_centroides(modelo, centros, tronco_val, et_val, percentil=percentil)
        falsos = medir_rechazo(modelo, pred_test, tronco_test)
        pilladas = medir_rechazo(modelo, pred_unknown, tronco_unknown)
        linea = f"{percentil:>10} | {falsos:>27.2%} | {pilladas:>21.2%}"
        print(linea)
        lineas_barrido.append(linea)

    modelo.umbral_distancia.copy_(umbral_original)

    ruta_reporte = VARIABLES_GLOBALES["RUTA_MODELOS"] / VARIABLES_GLOBALES["PRUEBA"] / "reporte_desconocidas.txt"
    ruta_reporte.write_text(
        "Evaluación de rechazo de especies\n"
        f"Imágenes desconocidas: {len(datos_unknown['etiquetas'])}\n"
        f"Desconocidas rechazadas: {rechazo_unknown:.2%}\n"
        f"Imágenes conocidas de test rechazadas: {rechazo_conocidas:.2%}\n"
        f"Confianza media en aciertos: {conf_aciertos:.4f}\n"
        f"Confianza media en fallos: {conf_fallos:.4f}\n\n"
        "Barrido de percentiles (umbral calibrado en val, medido en test/unknown):\n"
        + "\n".join(lineas_barrido) + "\n",
        encoding="utf-8",
    )
    print(f"\nInforme guardado en: {ruta_reporte}")


if __name__ == "__main__":
    main()