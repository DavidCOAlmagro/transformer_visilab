
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

    def __init__(self, embeddings, etiquetas_especie, etiquetas_genero=None):
        self.embeddings = embeddings
        self.etiquetas_especie = etiquetas_especie
        self.etiquetas_genero = etiquetas_genero

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, index):
        """
        Según si es genero o especie, devuelve un tuple con (embedding, etiqueta_especie) 
        o (embedding, etiqueta_especie, etiqueta_genero)"""
        
        if self.etiquetas_genero is None:
            return self.embeddings[index], self.etiquetas_especie[index]
        return (self.embeddings[index], self.etiquetas_especie[index],
                self.etiquetas_genero[index])
