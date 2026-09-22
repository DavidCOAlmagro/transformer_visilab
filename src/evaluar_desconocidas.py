"""Evalúa el rechazo de especies que no pertenecen al experimento."""

from pathlib import Path

import torch

from constantes import VARIABLES_GLOBALES
from modelo import cargar_modelo_entrenado
from preparar_datos import get_datos


@torch.no_grad()
def medir_rechazo(modelo: torch.nn.Module, embeddings: torch.Tensor) -> float:
    """Devuelve la proporción de embeddings rechazados como desconocidos."""
    device = next(modelo.parameters()).device
    embeddings = embeddings.to(device)
    rechazos: list[torch.Tensor] = []

    for lote in embeddings.split(VARIABLES_GLOBALES["BATCH_SIZE"]):
        logits, _, embedding_tronco = modelo(lote)
        indice_predicho = logits.argmax(dim=1)
        rechazos.append(modelo.es_desconocida(
            embedding_tronco, indice_predicho).cpu())

    return torch.cat(rechazos).float().mean().item()


def main() -> None:
    modelo, _ = cargar_modelo_entrenado()
    datos_unknown = get_datos("unknown")
    datos_test = get_datos("test")

    rechazo_unknown = medir_rechazo(modelo, datos_unknown["embeddings"])
    rechazo_conocidas = medir_rechazo(modelo, datos_test["embeddings"])

    ruta_reporte = (
        VARIABLES_GLOBALES["RUTA_MODELOS"] /
        VARIABLES_GLOBALES["PRUEBA"] / "reporte_desconocidas.txt"
    )
    ruta_reporte.write_text(
        "Evaluación de rechazo de especies\n"
        f"Imágenes desconocidas: {len(datos_unknown['etiquetas'])}\n"
        f"Desconocidas rechazadas: {rechazo_unknown:.2%}\n"
        f"Imágenes conocidas de test rechazadas: {rechazo_conocidas:.2%}\n",
        encoding="utf-8",
    )
    print(f"Desconocidas rechazadas: {rechazo_unknown:.2%}")
    print(f"Conocidas rechazadas por error: {rechazo_conocidas:.2%}")
    print(f"Informe guardado en: {ruta_reporte}")


if __name__ == "__main__":
    main()
