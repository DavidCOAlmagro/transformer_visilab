"""
--------------------------------------
Transformer DINOV2
David Calzado Olmo
--------------------------------------
"""

import os
import math
from pathlib import Path
import torch
from torch import nn
from constantes import VARIABLES_GLOBALES
from preparar_datos import get_datos, codificacion, contar_clases_train
from generar_leer_splits import leer_split, generar_split
from embeddings import inicializar_dinov2, calcular_embeddings
from clasificador import ClasificadorDiatomeas
from dataloader import crear_dataloaders,calcular_pesos_clases
from entrenamiento import entrenar_modelo
from evaluar_test_metricas import main as evaluar_test, graficar_curvas_entrenamiento

def limpiar_pantalla() -> None:
    """Limpia la consola de forma compatible con Windows y Unix."""
    os.system("cls" if os.name == "nt" else "clear")


def lr_lambda(epoca_actual: int) -> float:
    """
    Scheduler con warmup + descenso:
    Durante las primeras x épocas, el learning rate sube
    linealmente desde casi 0 hasta el valor normal (evita que los
    pesos, recién inicializados, den un "giro brusco" grande al principio).
    Después, desciende suavemente hasta el final.
    """
    epocas_warmup = VARIABLES_GLOBALES["EPOCAS_WARMUP"]
    num_epocas_total = VARIABLES_GLOBALES["num_epocas"]
    if epoca_actual < epocas_warmup:
        # Warmup lineal: 1/epocas_warmup, 2/epocas_warmup, ... hasta 1.0
        return (epoca_actual + 1) / epocas_warmup
    # Descenso coseno desde 1.0 hasta ~0.0 en las épocas restantes
    progreso = (epoca_actual - epocas_warmup) / \
        max(1, num_epocas_total - epocas_warmup)
    return 0.5 * (1 + math.cos(math.pi * progreso))


def main() -> None:
    """
    Función trabajo principal
    """
    limpiar_pantalla()

    # Semilla fija para que la inicialización de pesos, el shuffle del
    # dataloader y el data augmentation sean reproducibles entre ejecuciones.
    # Así, si el resultado cambia, sabemos que es por un cambio real y no por azar.

    SEMILLA = 42
    torch.manual_seed(SEMILLA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEMILLA)

    ruta_mejor_modelo = VARIABLES_GLOBALES["RUTA_MODELOS"] / VARIABLES_GLOBALES["PRUEBA"] / "mejor_modelo.pth"

    # Si ya existe un modelo entrenado, preguntamos si se quiere reentrenar
    # o usar directamente el que ya está guardado en disco
    entrenar_de_nuevo = True
    if ruta_mejor_modelo.exists():
        respuesta = input(
            "Ya existe un modelo entrenado (mejor_modelo.pth).\n"
            "¿Quieres entrenar uno nuevo? (s/n): ").strip().lower()
        entrenar_de_nuevo = respuesta == "s"

    if entrenar_de_nuevo:
        print("Quieres regenerar splits de train/val/test? (s/n): ")
        resp_split = input().strip().lower()

        if resp_split == "s":
            print("Regenerando splits...")
            generar_split()
        else:
            print("Usando splits ya existentes.")

        print("¿Quieres recalcular embeddings? (s/n): ")
        resp_emb = input().strip().lower()

        if resp_emb == "s":
            print("Recalculando embeddings...")
            preparar_embeddings_splits()
        else:
            print("Usando embeddings ya existentes.")

        print("Cargando datos...")
        datos_train = get_datos("train")
        datos_val = get_datos("val")
        _ = get_datos("test")

        print("Renumerando etiquetas...")
        emb_train, et_train, numero_especie = codificacion(datos_train)
        emb_val, et_val, _ = codificacion(datos_val)
        contar_clases_train(et_train, numero_especie)
        num_clases = len(VARIABLES_GLOBALES["ESPECIES_FILTRADAS"])
        print("Creando modelo...")
        modelo = ClasificadorDiatomeas(num_clases).to(
            VARIABLES_GLOBALES["DEVICE"])

        # El optimizador es el encargado de actualizar los pesos de la red neuronal para que aprenda
        # modelo.parameters() devuelve los pesos y sesgos de la red neuronal entrenables.
        # Learning rate 0.0003 es un valor pequeño para no oscilar demasiado.
        # AdamW con weight decay 0.0001 ayuda a regularizar el modelo y evitar overfitting.
        optimizador = torch.optim.AdamW(
            modelo.parameters(), lr=VARIABLES_GLOBALES["LEARNING_RATE"], 
            weight_decay=VARIABLES_GLOBALES["WEIGHT_DECAY"])

        num_epocas_total = VARIABLES_GLOBALES["num_epocas"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizador, lr_lambda)

        print("Creando dataloaders...")
        dataloader_train, dataloader_val = crear_dataloaders(
            emb_train, et_train, emb_val, et_val)

        pesos_clase = calcular_pesos_clases(et_train)
        # Función que devuelve la puntuación(logits) de cada clase para cada imagen.
        func_loss = nn.CrossEntropyLoss(label_smoothing=VARIABLES_GLOBALES["LABEL_SMOOTHING"],
                                        weight=pesos_clase.to(VARIABLES_GLOBALES["DEVICE"]))
        print("Iniciando entrenamiento...")

        historial_perdida_train, historial_perdida_val, historial_precision_val, historial_macro_f1_val = entrenar_modelo(
            modelo, dataloader_train, dataloader_val, func_loss, optimizador, scheduler,
            ruta_mejor_modelo, num_epocas_total, paciencia=VARIABLES_GLOBALES["PACIENCIA"])

        # Graficamos la evolución de pérdida y precisión de todas las épocas entrenadas
        ruta_curvas = VARIABLES_GLOBALES["RUTA_MODELOS"] / \
            "curvas_entrenamiento.png"
        graficar_curvas_entrenamiento(
            historial_perdida_train, historial_perdida_val,
            historial_precision_val, historial_macro_f1_val, ruta_curvas)
    else:
        print("Usando el modelo ya entrenado, sin reentrenar.")

    # Al terminar el entrenamiento (o si se ha saltado), evaluamos automáticamente en test ---
    print("\nEvaluando en el conjunto de test...")
    evaluar_test()


def preparar_embeddings_splits() -> None:
    """
    Calcula y guarda los embeddings de train/val/test a partir de los splits
    guardados en data/splits/. Aplica augmentation solo al conjunto de train.
    Si los embeddings ya existen en disco, no se recalculan.
    """
    ruta_splits: Path = VARIABLES_GLOBALES["RUTA_SPLITS"]
    ruta_embeddings: Path = VARIABLES_GLOBALES["RUTA_EMBEDDINGS"]
    # Solo train tiene augmentation:
    configuracion: dict[str, bool] = {
        "train": True,
        "val": False,
        "test": False,
    }

    # Comprobamos si YA existen los 3 archivos de embeddings antes de nada
    rutas_destino = {
        nombre_split: ruta_embeddings / f"embeddings_{nombre_split}.pt"
        for nombre_split in configuracion
    }
    todos_existen = True
    for ruta in rutas_destino.values():
        if not ruta.exists():
            todos_existen = False

    regenerar_todo = False

    if todos_existen:
        respuesta = input(
            "Ya existen embeddings de train, val y test.\n"
            "¿Quieres regenerarlos los 3 desde cero? (s/n): ").strip().lower()
        regenerar_todo = respuesta == "s"

    processor, model, device, augmentation = inicializar_dinov2()
    # Bucle sobre cada split (train, val, test) y calcula los embeddings si no existen
    for nombre_split, is_train in configuracion.items():
        ruta_destino = rutas_destino[nombre_split]

        if not ruta_destino.exists() or regenerar_todo:
            ruta_split_txt = ruta_splits / f"{nombre_split}.txt"
            imagenes = leer_split(ruta_split_txt)
            print(
                f"Calculando embeddings de {nombre_split} ({len(imagenes)} imágenes)...")
            datos = calcular_embeddings(imagenes, processor, model, device,
                                        augmentation, is_train=is_train)
            torch.save(datos, ruta_destino)
            print(f"Guardado en {ruta_destino}")


if __name__ == "__main__":
    main()
