"""
Center Loss fuerza a que los embeddings de una misma clase se agrupen
en un centro aprendido, compacta las clases en un centro,(intra-clase)
complementando a CrossEntropy, que separa las clases entre si (inter-clase).
"""

import torch
from torch import nn

class center_loss(nn.Module):
    def __init__(self,num_clases:int,dim_embedding:int,device:torch.device) -> None:
        super().__init__()
        # Un centro por clase, de tamaño dim_embedding:
        self.centros = nn.Parameter(torch.randn(num_clases,dim_embedding, device=device))
        
    def forward(self,embeddings:torch.Tensor,etiquetas:torch.Tensor) -> torch.Tensor:
        """
        Devuelve la distancia media entre cada embedding y el
        centro de su clase correspondiente.Los centros son parámetros 
        aprendibles normales (nn.Parameter), así que el propio optimizador.step() 
        los actualiza junto con el resto de pesos
        """
        
        # Obtener los centros correspondientes a cada etiqueta
        centros_correspondientes = self.centros[etiquetas]
        # Calcular la distancia entre los embeddings y sus centros correspondientes
        distancia = (embeddings - centros_correspondientes).pow(2).sum(dim=1)
        return distancia.mean()