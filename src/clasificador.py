"""
--------------------------------------
Red neuronal clasificadora que opera sobre embeddings de DINOv2.
Entrada: embedding de 768 dimensiones por imagen.
Salida: una puntuación por cada clase (especie).
--------------------------------------
"""

import torch
from torch import nn
from constantes import VARIABLES_GLOBALES


class ClasificadorDiatomeas(nn.Module):
    """
    El problema es que pesos y sesgo son números
    que cambian durante el entrenamiento necesitan vivir en algún sitio,
    recordarse entre llamadas, guardarse en disco, moverse a GPU... Por
    eso los encapsulamos en un clasificador que hereda de nn.Module. (Neural Network Module)
    """

    def __init__(self, num_clases: int, num_generos: int) -> None:
        """
        En un inicio probé con una sola capa lineal (fully connected) que transformaba el embedding
        en logits de clase. Pero el modelo no era suficientemente potente para aprender a clasificar
        bien, así que añadí capas intermedias con activaciones ReLU y Dropout para regularizar
        y evitar overfitting. Transforma el embedding en logits de clase.
        También crea los sesgos (bias) que se suman a los logits.
        """
        super().__init__()
        self.tronco = nn.Sequential(
            # Capa 1: Coge 768 valores de embedding y los transforma en 512 valores intermedios
            nn.Linear(VARIABLES_GLOBALES["DIM_EMBEDDING"], 512),
            # Función de activación ReLU (Rectified Linear Unit) que introduce no linealidad.
            nn.ReLU(),
            # Desactiva aleatoriamente un porcentaje de neuronas durante el entrenamiento.
            nn.Dropout(0.3),

            # Capa 2: Coge 512 valores de la capa previa y los transforma en 256 valores intermedios
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # Capa final: Coge 256 valores de la capa previa y
        # los transforma en num_clases/num_generos valores de salida
        self.cabeza_especie = nn.Linear(256, num_clases)
        self.cabeza_genero = nn.Linear(256, num_generos)
        
        # Recorre recursivamente todos los submódulos del modelo y, para cada
        # capa nn.Linear, aplica Xavier en los pesos y bias a cero. Así evitamos
        # inicializaciones manuales capa por capa.
        self.apply(self._inicializar_capa_lineal)

    @staticmethod
    def _inicializar_capa_lineal(capa: nn.Module) -> None:
        """Aplica Xavier + bias a cero solo a capas nn.Linear."""
        if isinstance(capa, nn.Linear):
            nn.init.xavier_uniform_(capa.weight)
            nn.init.zeros_(capa.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define que ocurre cuando le das datos al modelo.
        x: [batch(n), 768]
        """
        # Embedding transformado por las capas intermedias
        embedding = self.tronco(x)
        logits_especie = self.cabeza_especie(embedding)
        logits_genero = self.cabeza_genero(embedding)
        return logits_especie, logits_genero
