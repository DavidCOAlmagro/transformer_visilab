"""
DataLoader es el indicado en como servir los datos al modelo, agrupa en batches,
mezcla,itera...
"""


import torch
from torch.utils.data import DataLoader
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

    dataloader_train = DataLoader(dataset_train,
                                  batch_size=VARIABLES_GLOBALES["BATCH_SIZE"], shuffle=True,
                                  num_workers=VARIABLES_GLOBALES["NUM_WORKERS"],
                                  pin_memory=VARIABLES_GLOBALES["PIN_MEMORY"],
                                  persistent_workers=VARIABLES_GLOBALES["PERSISTENT_WORKERS"])
    dataloader_val = DataLoader(dataset_val,
                                batch_size=VARIABLES_GLOBALES["BATCH_SIZE"], shuffle=False,
                                num_workers=VARIABLES_GLOBALES["NUM_WORKERS"],
                                pin_memory=VARIABLES_GLOBALES["PIN_MEMORY"],
                                persistent_workers=VARIABLES_GLOBALES["PERSISTENT_WORKERS"])

    return dataloader_train, dataloader_val
