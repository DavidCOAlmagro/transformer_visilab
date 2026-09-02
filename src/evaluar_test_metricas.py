"""
--------------------------------------
Métricas avanzadas de evaluación: matriz de confusión, precisión, recall,
F1 y accuracy por clase, para el modelo ya entrenado.
--------------------------------------
"""
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report,accuracy_score,f1_score,precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
from constantes import VARIABLES_GLOBALES
from modelo import cargar_modelo_entrenado
from preparar_datos import get_datos, codificacion, construir_numero_genero, etiquetas_a_generos
from dataset import MyDataset

@torch.no_grad()
def obtener_predicciones(
        modelo: nn.Module, dataloader: DataLoader) -> tuple[list[int], list[int]]:
    """
    Pasa todos los batches de un dataloader por el modelo y devuelve tres listas
    paralelas: las etiquetas verdaderas de especie (y_true), las predichas de
    especie (y_pred), y las predichas de género (genero_pred). Esto último sirve
    para medir si los errores de especie vienen de que el género se predice mal
    (en cuyo caso el error de especie es inevitable, ya que la cabeza de ese
    género ni siquiera tiene la especie correcta entre sus opciones).
    """
    modelo.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    genero_pred: list[int] = []

    for batch_embeddings, batch_etiquetas in tqdm(dataloader, desc="Evaluando test"):
        batch_embeddings = batch_embeddings.to(next(modelo.parameters()).device)


        # Para cada embedding, el modelo devuelve un vector con las probabilidades/logits.
        logits_especie, logits_genero = modelo(batch_embeddings)
        # Predicción final, devuelve el mayor logit y el indice
        _, indice_predicciones = torch.max(logits_especie, 1)
        _, indice_predicciones_genero = torch.max(logits_genero, 1)
        # Añade las etiquetas verdaderas y predichas a las listas correspondientes
        y_true.extend(batch_etiquetas.tolist())
        y_pred.extend(indice_predicciones.cpu().tolist())
        genero_pred.extend(indice_predicciones_genero.cpu().tolist())

    return y_true, y_pred,genero_pred


def matriz_confusion(y_true: list[int], y_pred: list[int],
                     nombres_clases: list[str], ruta_guardado: Path) -> None:
    """
    Calcula la matriz de confusión a partir de las predicciones y la guarda
    como imagen PNG. Cada fila es la clase real, cada columna la clase predicha.
    La diagonal principal son los aciertos; fuera de la diagonal, los errores.
    """
    matrix: np.ndarray = confusion_matrix(y_true, y_pred)
    displayer: ConfusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix=matrix,
                                                               display_labels=nombres_clases)
    # figsize se adapta un poco al número de clases para que no se amontonen
    fig, eje = plt.subplots(figsize=(max(6, len(nombres_clases) * 0.8),
                                     max(6, len(nombres_clases) * 0.8)))
    displayer.plot(cmap=plt.get_cmap("Blues"), xticks_rotation=45, ax=eje)
     # Alinea el texto rotado a la derecha, para que no se corte al final
    plt.setp(eje.get_xticklabels(), ha="right")

    plt.title("Matriz de confusión")
    # Ajusta los márgenes
    plt.tight_layout()
    ruta_guardado.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta_guardado, dpi=300)
    plt.close(fig)

    print(f"Matriz de confusión guardada en: {ruta_guardado}")


def generar_reporte_clasificacion(
        y_true: list[int], y_pred: list[int], nombres_clases: list[str]) -> str:
    """
    Genera un texto con precision, recall, F1-score y accuracy por cada clase,
    además de los promedios globales (macro y weighted).
    """
    reporte: str = classification_report(
        y_true, y_pred,
        target_names=nombres_clases,
        zero_division=0  # evita warnings si alguna clase no tiene predicciones
    )
    return reporte

def calcular_accuracy_genero(y_true_especie: list[int], gen_pred: list[int],
                             numero_especie: dict[str, int],
                             numero_genero: dict[str, int]) -> float:
    """
    Calcula qué porcentaje de muestras de test tienen el género predicho
    correctamente. Sirve para diagnosticar si los errores de especie del
    clasificador jerárquico vienen de que el género se predice mal (en
    cuyo caso el error de especie es inevitable, no un fallo del
    clasificador de especie en sí).
    """
    # Convertimos las etiquetas verdaderas de especie a género real
    et_true_tensor : torch.Tensor = torch.tensor(y_true_especie, dtype=torch.long)
    genero_true_tensor : torch.Tensor =  etiquetas_a_generos(et_true_tensor, numero_especie, numero_genero)
    genero_true : list[int] = genero_true_tensor.tolist()
    
    aciertos : int = sum(1 for true, pred in zip(genero_true, gen_pred) if true == pred)
    accuracy : float = aciertos / len(genero_true) 
    return accuracy

def main() -> dict[str, float]:                            
    """
    Carga el modelo entrenado, evalúa el conjunto de test y genera
    la matriz de confusión + el reporte de clasificación.
    """
    datos_test = get_datos("test")
    emb_test, et_test, numero_especie = codificacion(datos_test)
    dataset_test = MyDataset(emb_test, et_test)

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=VARIABLES_GLOBALES["BATCH_SIZE"],
        shuffle=False,
        num_workers=VARIABLES_GLOBALES["NUM_WORKERS"],
        pin_memory=VARIABLES_GLOBALES["PIN_MEMORY"]
    )

    numero_genero = construir_numero_genero(VARIABLES_GLOBALES["ESPECIES_FILTRADAS"])
    modelo, _ = cargar_modelo_entrenado()

    # Carga los pesos del mejor modelo entrenado
    ruta_pesos=VARIABLES_GLOBALES["RUTA_MODELOS"]/VARIABLES_GLOBALES["PRUEBA"] / "mejor_modelo.pth"
    pesos = torch.load(ruta_pesos, map_location=VARIABLES_GLOBALES["DEVICE"],weights_only=True)
    
    if not ruta_pesos.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado en: {ruta_pesos}\n"
            "Entrena el modelo antes de ejecutar la evaluación."
        )
    # Carga los pesos en el modelo
    modelo.load_state_dict(pesos)
    modelo.eval()

    y_true, y_pred, y_pred_genero = obtener_predicciones(modelo, dataloader_test)
    accuracy_genero = calcular_accuracy_genero(y_true, y_pred_genero, numero_especie, numero_genero)
    print(f"\nAccuracy del clasificador de género en test: {accuracy_genero:.2%}\n")
    # La lista de nombres de clases se ordena según el índice de especie
    # para que coincida con las etiquetas
    nombres_clases = sorted(numero_especie, key=numero_especie.get)


    ruta_matriz = VARIABLES_GLOBALES["RUTA_MODELOS"] / VARIABLES_GLOBALES["PRUEBA"] / \
        "matriz_confusion_test.png"
    matriz_confusion(y_true, y_pred, nombres_clases, ruta_matriz)
    ruta_metricas_especie = VARIABLES_GLOBALES["RUTA_MODELOS"] / VARIABLES_GLOBALES["PRUEBA"] / \
    "metricas_por_especie.png"
    graficar_metricas_por_especie(y_true, y_pred, nombres_clases, ruta_metricas_especie)
    
    reporte = generar_reporte_clasificacion(y_true, y_pred, nombres_clases)
    print("\nReporte de clasificación (test):\n")
    print(reporte)

    ruta_reporte=VARIABLES_GLOBALES["RUTA_MODELOS"]/VARIABLES_GLOBALES["PRUEBA"]/"reporte_test.txt"
    with open(ruta_reporte, "w", encoding="utf-8") as archivo:
        archivo.write(reporte)
    print(f"Reporte guardado en: {ruta_reporte}")
    
    return {
    "accuracy_test": accuracy_score(y_true, y_pred),
    "macro_f1_test": f1_score(y_true,y_pred,average="macro",zero_division=0),
    "accuracy_genero_test": accuracy_genero
    }

def graficar_curvas_entrenamiento( perdida_train: list[float], perdida_val: list[float],
        precision_val: list[float], macro_f1_val: list[float], ruta_guardado: Path) -> None:
    """
    Dibuja dos gráficas una al lado de la otra: la evolución de la pérdida
    (train vs val) y la evolución de la precisión de validación, por época.
    Sirve para ver de un vistazo si el modelo mejora, se estanca, o empieza
    a sobreajustar (overfitting).
    """
    num_epocas = len(perdida_train)
    epocas = range(1, num_epocas + 1)

    fig, (eje_perdida, eje_precision) = plt.subplots(1, 2, figsize=(12, 5))

    # Gráfica de la izquierda: pérdida
    eje_perdida.plot(epocas, perdida_train, label="Train")
    eje_perdida.plot(epocas, perdida_val, label="Validación")
    eje_perdida.set_title("Pérdida por época")
    eje_perdida.set_xlabel("Época")
    eje_perdida.set_ylabel("Pérdida")
    eje_perdida.legend()

    # Gráfica de la derecha: precisión de validación
    eje_precision.plot(epocas, precision_val, label="Validación", color="green")
    eje_precision.set_title("Precisión de validación por época")
    eje_precision.set_xlabel("Época")
    eje_precision.set_ylabel("Precisión")
    eje_precision.legend()

    # Gráfica de la derecha: macro F1 de validación
    eje_macro_f1 = eje_precision.twinx()
    eje_macro_f1.plot(epocas, macro_f1_val, label="Macro F1", color="blue")
    eje_macro_f1.set_ylabel("Macro F1")
    eje_macro_f1.legend(loc="upper right")

    plt.tight_layout()

    ruta_guardado.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta_guardado, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Curvas de entrenamiento guardadas en: {ruta_guardado}")
def graficar_metricas_por_especie(
        y_true: list[int], y_pred: list[int],
        nombres_clases: list[str], ruta_guardado: Path) -> None:
    """
    Dibuja un gráfico de barras agrupadas con precision, recall y F1
    de cada especie en test, ordenadas de menor a mayor F1. Sirve para
    ver de un vistazo qué especies le cuestan más al modelo, útil para
    informes y presentaciones.
    """
    precision, recall, f1, _soporte = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(nombres_clases)), zero_division=0)

    # Ordena los índices de especie de menor a mayor F1
    orden = sorted(range(len(nombres_clases)), key=lambda i: f1[i])
    especies_ordenadas = [nombres_clases[i] for i in orden]
    precision_ordenada = [precision[i] for i in orden]
    recall_ordenado = [recall[i] for i in orden]
    f1_ordenado = [f1[i] for i in orden]

    posiciones = np.arange(len(especies_ordenadas))
    ancho_barra = 0.25

    fig, eje = plt.subplots(figsize=(max(10, len(especies_ordenadas) * 0.5), 6))
    eje.bar(posiciones - ancho_barra, precision_ordenada, ancho_barra, label="Precision")
    eje.bar(posiciones, recall_ordenado, ancho_barra, label="Recall")
    eje.bar(posiciones + ancho_barra, f1_ordenado, ancho_barra, label="F1")

    eje.set_xticks(posiciones)
    eje.set_xticklabels(especies_ordenadas, rotation=45, ha="right")
    eje.set_ylim(0, 1.05)
    eje.set_ylabel("Puntuación")
    eje.set_title("Precision / Recall / F1 por especie (test), ordenado de peor a mejor F1")
    eje.legend()

    plt.tight_layout()
    ruta_guardado.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta_guardado, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Gráfico de métricas por especie guardado en: {ruta_guardado}")

if __name__ == "__main__":
    main()
