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
        super().__init__()
        # Crea y registra los pesos y sesgos de la capa lineal (fully connected)
        # que transforma el embedding en logits de clase. [2.1, -0.5, 3.7, 0.2, -1.3]
        # También crea los sesgos (bias) que se suman a los logits. [0.1, -0.2, 0.3, 0.4, -0.5]
        self.clasificador = nn.Linear(
            VARIABLES_GLOBALES["DIM_EMBEDDING"],
            num_clases
        )
        # Xavier controla los pesos iniciales para que no dependa de pytorch
        nn.init.xavier_uniform_(self.clasificador.weight)
        nn.init.zeros_(self.clasificador.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define que ocurre cuando le das datos al modelo.
        x: [batch(n), 768]
        """
        return self.clasificador(x)
