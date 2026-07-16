"""
DataLoader es el indicado en como servir los datos al modelo, agrupa en batches,
mezcla,itera...
"""


import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import MyDataset
from constantes import VARIABLES_GLOBALES


def crear_dataloaders(
        emb_train: torch.Tensor, et_train: torch.Tensor, emb_val: torch.Tensor,
        et_val: torch.Tensor) -> tuple[DataLoader, DataLoader]:
    """
    Crea los DataLoaders para entrenamiento y validación.
    """

    dataset_train = MyDataset(emb_train, et_train)
    dataset_val = MyDataset(emb_val, et_val)
    
    # WeightedRandomSampler balancea las clases minoritarias
    pesos_muestras = calcular_pesos_muestras(et_train)
    sampler_train = WeightedRandomSampler(
    weights=pesos_muestras,
    num_samples=len(pesos_muestras),
    replacement=True
    )

    dataloader_train = DataLoader(dataset_train,
                                  batch_size=VARIABLES_GLOBALES["BATCH_SIZE"],
                                  sampler=sampler_train,
                                  num_workers=VARIABLES_GLOBALES["NUM_WORKERS"],
                                  pin_memory=VARIABLES_GLOBALES["PIN_MEMORY"],
                                  persistent_workers=VARIABLES_GLOBALES["PERSISTENT_WORKERS"])
    dataloader_val = DataLoader(dataset_val,
                                batch_size=VARIABLES_GLOBALES["BATCH_SIZE"],
                                num_workers=VARIABLES_GLOBALES["NUM_WORKERS"],
                                pin_memory=VARIABLES_GLOBALES["PIN_MEMORY"],
                                persistent_workers=VARIABLES_GLOBALES["PERSISTENT_WORKERS"])

    return dataloader_train, dataloader_val

def calcular_pesos_muestras(etiquetas: torch.Tensor) -> torch.Tensor:
    """
    Calcula un peso por cada muestra de train, inversamente proporcional
    a la frecuencia de su clase. Las clases con pocas imágenes obtienen
    un peso mayor, para que el WeightedRandomSampler las muestree con
    más frecuencia y compensar así el desbalance entre especies.
    """
    # Cuenta cuántas muestras hay de cada clase (0, 1, 2, ...)
    conteo_por_clase: torch.Tensor = torch.bincount(etiquetas)

    # Peso de cada clase = 1 / número de muestras de esa clase.
    peso_por_clase: torch.Tensor = 1.0 / torch.sqrt(conteo_por_clase.float())

    # Le asigna a cada muestra el peso de la clase a la que pertenece
    pesos_muestras: torch.Tensor = peso_por_clase[etiquetas]
    return pesos_muestras

def calcular_pesos_clases(etiquetas: torch.Tensor) -> torch.Tensor:
    """
    Calcula un peso por cada clase (no por muestra), inversamente
    proporcional a su frecuencia en train. Se usa en CrossEntropyLoss
    para que los errores en clases minoritarias penalicen más.
    """
    conteo_por_clase: torch.Tensor = torch.bincount(etiquetas)
    peso_por_clase: torch.Tensor = 1.0 / torch.sqrt(conteo_por_clase.float())

    return peso_por_clase.to(VARIABLES_GLOBALES["DEVICE"])
