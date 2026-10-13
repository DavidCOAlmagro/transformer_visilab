"""
--------------------------------------
Calibración de confianza (temperature scaling).

Busca una temperatura T tal que softmax(logits / T) se parezca a la
probabilidad real de acierto en validación. El entrenamiento la divide en los
pesos de la cabeza de especie, así el .pth ya sale calibrado.
--------------------------------------
"""

import torch
from torch import nn


def ajustar_temperatura(logits: torch.Tensor, etiquetas: torch.Tensor,
                        pasos: int = 100, lr: float = 0.01) -> float:
    """Temperatura positiva que minimiza la CrossEntropy en validación."""
    log_temperatura = torch.nn.Parameter(torch.zeros(1))
    optimizador = torch.optim.LBFGS([log_temperatura], lr=lr, max_iter=pasos)
    func_perdida = nn.CrossEntropyLoss()

    def paso_cierre() -> torch.Tensor:
        optimizador.zero_grad()
        perdida = func_perdida(logits / torch.exp(log_temperatura), etiquetas)
        perdida.backward()
        return perdida

    optimizador.step(paso_cierre)
    return round(torch.exp(log_temperatura).item(), 4)
