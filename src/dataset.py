
"""
Un dataset de PyTorch es una clase que define que cualquier clase
que herede de ella puedes obtener el numero de objetos(len) y acceder a cada objeto
por su índice (getitem). Sin importar el tipo del que venga(imagenes,csv...)
"""
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """
    Clase que hereda de Dataset de PyTorch para saber cuándo termina una época (len(dataset))
    Pedir muestras una a una o en batches (dataset[i])
    Usarlo con un DataLoader para mezclar (shuffle), dividir en batches, cargar en paralelo, etc.
    """
    # Embeddings y etiquetas por separado,porque __getitem__ necesita devolver el
    # par (embedding, etiqueta) para que el entrenamiento pueda calcular la
    # pérdida (loss) comparando la predicción con la etiqueta real.
    def __init__(self, embeddings, etiquetas):
        self.embeddings = embeddings
        self.etiquetas = etiquetas
    def __len__(self):
        return len(self.embeddings)
    def __getitem__(self, index):
        return self.embeddings[index], self.etiquetas[index]
