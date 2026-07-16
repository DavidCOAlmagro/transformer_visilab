
"""
Un dataset de PyTorch es una clase que define que cualquier clase
que herede de ella puedes obtener el numero de objetos(len) y acceder a cada objeto
por su índice (getitem). Sin importar el data type del que venga
"""
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """
    Envuelve embeddings y etiquetas ya calculados para servírselos al DataLoader.
    """

    def __init__(self, embeddings, etiquetas):
        self.embeddings = embeddings
        self.etiquetas = etiquetas
    def __len__(self):
        return len(self.embeddings)
    def __getitem__(self, index):
        return self.embeddings[index], self.etiquetas[index]
