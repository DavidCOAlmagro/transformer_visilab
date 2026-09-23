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
            nn.Linear(VARIABLES_GLOBALES["DIM_EMBEDDING"],
                      VARIABLES_GLOBALES["DIM_CAPA_1"]),
            nn.ReLU(),
            nn.Dropout(VARIABLES_GLOBALES["DROPOUT_CAPA_1"]),
            nn.Linear(VARIABLES_GLOBALES["DIM_CAPA_1"],
                      VARIABLES_GLOBALES["DIM_CAPA_2"]),
            nn.ReLU(),
            nn.Dropout(VARIABLES_GLOBALES["DROPOUT_CAPA_2"]),
        )

        # La especie es la tarea principal. La cabeza de género regulariza el
        # tronco compartido, pero no condiciona la predicción de especie.
        self.cabeza_especie = nn.Linear(
            VARIABLES_GLOBALES["DIM_CAPA_2"], num_clases)
        self.cabeza_genero = nn.Linear(
            VARIABLES_GLOBALES["DIM_CAPA_2"], num_generos)
        self.apply(self._inicializar_capa_lineal)
        self.register_buffer(
            "centros", torch.zeros(num_clases, VARIABLES_GLOBALES["DIM_CAPA_2"]))
        self.register_buffer(
            "umbral_distancia", torch.full((num_clases,), float("inf")))

    @staticmethod
    def _inicializar_capa_lineal(capa: nn.Module) -> None:
        if isinstance(capa, nn.Linear):
            nn.init.xavier_uniform_(capa.weight)
            nn.init.zeros_(capa.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Devuelve logits globales de especie, logits de género y el embedding
        del tronco para center loss."""
        embedding = self.tronco(x)
        return self.cabeza_especie(embedding), self.cabeza_genero(embedding), embedding

    def distancia_a_centro(self, embedding: torch.Tensor,
                           indice_clase: torch.Tensor) -> torch.Tensor:
        """Devuelve la distancia al centro de la clase predicha para cada embedding."""
        indice_clase = indice_clase.to(self.centros.device)
        centros_correspondientes = self.centros[indice_clase]
        return (embedding.to(self.centros.device) - centros_correspondientes).pow(2).sum(dim=1)

    def es_desconocida(self, embedding: torch.Tensor,
                       indice_clase_predicha: torch.Tensor) -> torch.Tensor:
         """Devuelve un tensor de booleanos indicando qué embeddings son desconocidos."""
         indice_clase_predicha = indice_clase_predicha.to(self.umbral_distancia.device)
         distancia = self.distancia_a_centro(embedding, indice_clase_predicha)
         umbral = self.umbral_distancia[indice_clase_predicha]
         return distancia > umbral
