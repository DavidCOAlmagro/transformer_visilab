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
    Clase clasificador.El problema es que pesos y sesgo son números
    que cambian durante el entrenamiento necesitan vivir en algún sitio,
    recordarse entre llamadas, guardarse en disco, moverse a GPU...
    Por eso en lugar de funciones usamos una clase que hereda de nn.Module,
    que es la clase base de todos los modelos de PyTorch.
    nn es la neural network library de PyTorch, contiene capas, funciones de activación,
    optimizadores...Donde se guardan los pesos y sesgos de la red neuronal.
    768 entradas → n clases (especies).
    """

    def __init__(self, num_clases: int):
        """
        En un inicio usé una sola capa lineal (fully connected) que transformaba el embedding
        en logits de clase. Pero el modelo no era suficientemente potente para aprender a clasificar bien, así que añadí
        capas intermedias con activaciones ReLU y Dropout para regularizar y evitar overfitting.
        """
        super().__init__()
        # Crea y registra los pesos y sesgos de la capa lineal (fully connected)
        # que transforma el embedding en logits de clase. [2.1, -0.5, 3.7, 0.2, -1.3]
        # También crea los sesgos (bias) que se suman a los logits. [0.1, -0.2, 0.3, 0.4, -0.5]
        self.clasificador = nn.Sequential(
            nn.Linear(VARIABLES_GLOBALES["DIM_EMBEDDING"], 512),
            nn.ReLU(), # Función de activación ReLU (Rectified Linear Unit) que introduce no linealidad.
            nn.Dropout(0.3), # Desactiva aleatoriamente un porcentaje de neuronas durante el entrenamiento.
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_clases)
)
        # Recorremos todas las capas y solo inicializamos las lineales
        # (ReLU y Dropout no tienen pesos que inicializar)
        for capa in self.clasificador:
            if isinstance(capa, nn.Linear):
                nn.init.xavier_uniform_(capa.weight)
                nn.init.zeros_(capa.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define que ocurre cuando le das datos al modelo.
        x: [batch(n), 768]
        """
        return self.clasificador(x)
