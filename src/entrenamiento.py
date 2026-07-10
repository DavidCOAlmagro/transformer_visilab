"""
--------------------------------------
Script que entrena el modelo por épocas
con todo el dataset filtrado.
--------------------------------------
"""
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def entrenar_epoca(modelo: nn.Module,dataloader: DataLoader,func_loss: nn.Module,
    optimizador: torch.optim.Optimizer) -> float:
    """
    Entrena el modelo durante una época completa (una pasada por todo el
    dataloader). Devuelve la pérdida media de la época.
    """
    modelo.train()
    perdida_acumulada: float = 0.0

    # Itera sobre todos los batches del dataloader
    for batch_embeddings, batch_etiquetas in tqdm(dataloader, desc="Entrenando",leave=False):
        # Mueve los tensores a la GPU si el modelo está en GPU
        batch_embeddings = batch_embeddings.to(modelo.clasificador.weight.device)
        batch_etiquetas = batch_etiquetas.to(modelo.clasificador.weight.device)

        # Limpia los gradientes de la GPU para que no se acumulen de un batch a otro.
        optimizador.zero_grad()

        # Hace forward, intenta adivinar la especie
        salida: torch.Tensor = modelo(batch_embeddings)
        perdida: torch.Tensor = func_loss(salida, batch_etiquetas)

        # Calcula los gradientes de la pérdida con respecto a los pesos(como mejorar)
        perdida.backward()
        # Gradient clipping: limita la los gradientes para que no desvarien mucho
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
        # Actualiza los pesos de la red neuronal según los gradientes calculados en backward.
        optimizador.step()

        perdida_acumulada += perdida.item()

    perdida_media: float = perdida_acumulada / len(dataloader)
    return perdida_media

@torch.no_grad()
def validacion(modelo: nn.Module,dataloader: DataLoader,
               func_loss: nn.Module) -> tuple[float, float]:
    """
    Trabaja con los datos no utilizados en entrenamiento para ver si
    el modelo realmente está aprendiendo y no memorizando.
    """
    modelo.eval()
    perdida_acumulada: float = 0.0
    aciertos: int = 0
    total: int = 0
    for batch_embeddings, batch_etiquetas in tqdm(dataloader, desc="Validando",leave=False):
        batch_embeddings = batch_embeddings.to(modelo.clasificador.weight.device)
        batch_etiquetas = batch_etiquetas.to(modelo.clasificador.weight.device)

        salida: torch.Tensor = modelo(batch_embeddings)
        perdida: torch.Tensor = func_loss(salida, batch_etiquetas)
        perdida_acumulada += perdida.item()

        # Calcula el número de aciertos en este batch
        _, indice_predicciones = torch.max(salida, 1) # devuelve valor max e indice(especie)
        # Compara dos tensores,si son iguales devuelve True, lo mismo que sumar 1
        comparacion = torch.eq(indice_predicciones, batch_etiquetas)
        aciertos += comparacion.sum().item()

        total += len(batch_embeddings) # Numero de imagenes de cada batch

    perdida_media: float = perdida_acumulada / len(dataloader)
    precision: float = aciertos / total

    return perdida_media, precision
def entrenar_modelo(
        modelo: nn.Module, dataloader_train: DataLoader, dataloader_val: DataLoader,
        func_loss: nn.Module, optimizador: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler, ruta_mejor_modelo: Path,
        num_epocas: int, paciencia: int = 5
        ) -> tuple[list[float], list[float], list[float]]:
    """
    Bucle completo de entrenamiento por épocas: entrena, valida, actualiza
    el learning rate con el scheduler, aplica early stopping y guarda el
    mejor modelo según la pérdida de validación.
    Devuelve el historial de pérdida train/val y precisión val, uno por
    cada época realmente entrenada (para poder graficarlo después).
    """
    historial_perdida_train: list[float] = []
    historial_perdida_val: list[float] = []
    historial_precision_val: list[float] = []

    # Si no mejora en 'paciencia' épocas consecutivas, se detiene el entrenamiento (early stopping)
    mejor_perdida_val = float("inf")
    contador_no_mejora = 0
    # Crea directorio si no existe
    ruta_mejor_modelo.parent.mkdir(parents=True, exist_ok=True)

    for epoca in range(num_epocas):
        if contador_no_mejora < paciencia:
            perdida_train = entrenar_epoca(modelo, dataloader_train, func_loss, optimizador)
            perdida_val, precision_val = validacion(modelo, dataloader_val, func_loss)

            scheduler.step()  # avanza el learning rate según el schedule

            # Guardamos los valores de esta época en el historial
            historial_perdida_train.append(perdida_train)
            historial_perdida_val.append(perdida_val)
            historial_precision_val.append(precision_val)

            print(f"Época {epoca + 1}/{num_epocas} — "
                f"loss train: {perdida_train:.4f} — "
                f"loss val: {perdida_val:.4f} — "
                f"precisión val: {precision_val:.2%} — "
                f"lr: {scheduler.get_last_lr()[0]:.6f}")

            if perdida_val < mejor_perdida_val:
                mejor_perdida_val = perdida_val
                contador_no_mejora = 0
                torch.save(modelo.state_dict(), ruta_mejor_modelo)

            else:
                contador_no_mejora += 1
                print(f"No mejora ({contador_no_mejora}/{paciencia})")
        else:
            print("Early stopping: no hay mejora en validación.")

    return historial_perdida_train, historial_perdida_val, historial_precision_val
