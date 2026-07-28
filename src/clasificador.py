"""
Red neuronal clasificadora que opera sobre embeddings de DINOv2.
"""

import torch
from torch import nn

from constantes import VARIABLES_GLOBALES


class ClasificadorDiatomeas(nn.Module):
    """Clasifica especie y, como tarea auxiliar, género."""

    def __init__(self, num_clases: int, num_generos: int) -> None:
        super().__init__()

        self.tronco = nn.Sequential(
            nn.Linear(VARIABLES_GLOBALES["DIM_EMBEDDING"], 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # La especie es la tarea principal. La cabeza de género regulariza el
        # tronco compartido, pero no condiciona la predicción de especie.
        self.cabeza_especie = nn.Linear(256, num_clases)
        self.cabeza_genero = nn.Linear(256, num_generos)
        self.apply(self._inicializar_capa_lineal)

    @staticmethod
    def _inicializar_capa_lineal(capa: nn.Module) -> None:
        if isinstance(capa, nn.Linear):
            nn.init.xavier_uniform_(capa.weight)
            nn.init.zeros_(capa.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Devuelve logits globales de especie y logits de género."""
        embedding = self.tronco(x)
        return self.cabeza_especie(embedding), self.cabeza_genero(embedding)
