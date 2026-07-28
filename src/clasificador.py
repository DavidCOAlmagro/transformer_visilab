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
from preparar_datos import construir_numero_genero, etiquetas_a_generos

class ClasificadorDiatomeas(nn.Module):
    """
    El problema es que pesos y sesgo son números
    que cambian durante el entrenamiento necesitan vivir en algún sitio,
    recordarse entre llamadas, guardarse en disco, moverse a GPU... Por
    eso los encapsulamos en un clasificador que hereda de nn.Module. (Neural Network Module)
    """

    def __init__(self, num_clases: int, num_generos: int, especies_por_genero: dict[int, list[int]]) -> None:
        """
        En un inicio probé con una sola capa lineal (fully connected) que transformaba el embedding
        en logits de clase. Pero el modelo no era suficientemente potente para aprender a clasificar
        bien, así que añadí capas intermedias con activaciones ReLU y Dropout para regularizar
        y evitar overfitting. Transforma el embedding en logits de clase.
        También crea los sesgos (bias) que se suman a los logits. También añadí cabezas por cada género.                
        """
        super().__init__()
        self.num_clases = num_clases
        self.especies_por_genero = especies_por_genero
        
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
        self.cabeza_genero = nn.Linear(256, num_generos)
        # Capa final: Coge 256 valores de la capa previa y
        # Una por cada genero
        self.cabezas_especie = nn.ModuleDict({str(genero): nn.Linear(256, len(indices_especie)) 
                                             for genero, indices_especie in especies_por_genero.items()})

        
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

    def forward(self, x: torch.Tensor,
                genero_target: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Define que ocurre cuando le das datos al modelo.
        x: [batch(n), 768]
        genero_target: [batch] con el género REAL de cada muestra, solo
        disponible en entrenamiento/validación. Si es
        None (test/inferencia), se predice el género primero con
        logits_genero y se usa la cabeza de ESE género predicho (cascada real).
        """
        # Embedding transformado por las capas intermedias
        embedding = self.tronco(x)
        logits_genero = self.cabeza_genero(embedding)

        if genero_target is None:
            # Para cada muestra, cogemos el género con mayor logit
            genero_target = torch.argmax(logits_genero, dim=1)

        batch_size: int = x.size(0)
        # Creamos un tensor de logits de especie inicializado a -inf para que
        # cualquier logit que no se actualice sea descartado por softmax.
        VALOR_IMPOSIBLE: float = -0.0001
        logits_especie = torch.full((batch_size, self.num_clases), VALOR_IMPOSIBLE,
                                    device=x.device)

        # Recorremos el batch muestra a muestra 
        for i in range(batch_size): 
            genero_de_esta_muestra: int = genero_target[i].item()

            # Cogemos SOLO el embedding de esta muestra (mantiene la dimensión
            # de batch=1 con unsqueeze, porque la capa espera [batch, 256])
            embedding_muestra = embedding[i].unsqueeze(0)  # [1, 256]

            cabeza = self.cabezas_especie[str(genero_de_esta_muestra)]
            logits_local = cabeza(embedding_muestra)  # [1, n_especies_de_este_genero]

            # Traducimos cada logit local a su posición global de especie
            indices_globales = self.especies_por_genero[genero_de_esta_muestra]
            for pos_local, indice_global in enumerate(indices_globales):
                logits_especie[i, indice_global] = logits_local[0, pos_local]

        return logits_especie, logits_genero