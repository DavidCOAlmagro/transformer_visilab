"""
--------------------------------------
Calibración de confianza (temperature scaling).

El modelo, tal como está, da una confianza casi siempre pegada a 0.99,
acierte o falle. Se busca una temperatura T tal que, dividiendo los
logits entre T antes del softmax, la confianza se parezca a la
probabilidad real de acierto.

En vez de guardar T en un archivo aparte, se aplica directamente a los
pesos de la cabeza de especie y se sobrescribe el .pth. Así, quien
cargue el modelo (incluso sin conocer nada de calibración) ya recibe
probabilidades calibradas de serie, sin pasos extra.
--------------------------------------
"""

from pathlib import Path
import torch
from torch import nn

from preparar_datos import get_datos, codificacion


@torch.no_grad()
def obtener_datos_val(modelo: nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Obtiene logits, etiquetas y embeddings de validación."""
    datos_val = get_datos("val")
    emb_val, et_val, _ = codificacion(datos_val)
    device = next(modelo.parameters()).device
    emb_val = emb_val.to(device)
    logits_especie, _, tronco_embedding = modelo(emb_val)
    return logits_especie.cpu(), et_val, tronco_embedding.cpu()


def ajustar_temperatura(logits: torch.Tensor, etiquetas: torch.Tensor,
                        pasos: int = 100, lr: float = 0.01) -> float:
    """
    Ajusta una temperatura positiva para calibrar la confianza del modelo,
    minimizando la CrossEntropy en validación respecto a T.
    """
    log_temperatura = torch.nn.Parameter(torch.zeros(1))
    optimizador = torch.optim.LBFGS([log_temperatura], lr=lr, max_iter=pasos)
    func_perdida = nn.CrossEntropyLoss()

    def paso_cierre() -> torch.Tensor:
        temperatura = torch.exp(log_temperatura)
        optimizador.zero_grad()
        perdida = func_perdida(logits / temperatura, etiquetas)
        perdida.backward()
        return perdida

    optimizador.step(paso_cierre)
    temperatura = torch.exp(log_temperatura)
    return round(temperatura.item(), 4)


def hornear_temperatura_en_pesos(modelo: nn.Module, temperatura: float) -> None:
    """
    Aplica la temperatura directamente a los pesos de la cabeza de especie,
    in-place. Matemáticamente equivalente a dividir los logits entre T en
    cada inferencia (softmax(Wx+b) / T == softmax((W/T)x + b/T)), pero sin
    depender de ningún archivo extra ni de que quien use el modelo sepa
    nada de calibración: el .pth ya sale calibrado.
    """
    with torch.no_grad():
        modelo.cabeza_especie.weight.div_(temperatura)
        modelo.cabeza_especie.bias.div_(temperatura)


def calibrar_centroides(modelo: nn.Module, centros: torch.Tensor,
                        embeddings: torch.Tensor, etiquetas: torch.Tensor,
                        percentil: float = 95.0) -> None:
    """Guarda los centros y calcula un umbral de distancia por clase."""
    device = next(modelo.parameters()).device
    centros = centros.to(device=device, dtype=modelo.centros.dtype)
    embeddings = embeddings.to(device)
    etiquetas = etiquetas.to(device)

    modelo.centros.copy_(centros)
    umbrales = torch.full(
        (centros.shape[0],), float("inf"), device=device, dtype=centros.dtype)
    distancias = modelo.distancia_a_centro(embeddings, etiquetas)
    for indice_clase in range(centros.shape[0]):
        distancias_clase = distancias[etiquetas == indice_clase]
        if distancias_clase.numel():
            umbrales[indice_clase] = torch.quantile(
                distancias_clase, percentil / 100.0)
    modelo.umbral_distancia.copy_(umbrales)


def calibrar_pesos(modelo: nn.Module, centros: torch.Tensor | None,
                   ruta_modelo: Path) -> float:
    """
    Calibra la confianza del modelo sobre validación, hornea la
    temperatura resultante en los pesos de la cabeza de especie y
    sobrescribe ruta_modelo con el modelo ya calibrado.

    No cambia ninguna predicción ni el accuracy/macro F1: solo aplana
    las probabilidades para que la confianza reportada sea más honesta.
    Devuelve la temperatura usada, solo para dejar constancia en consola.
    """
    logits_val, et_val, embeddings_val = obtener_datos_val(modelo)
    temperatura = ajustar_temperatura(logits_val, et_val)

    print(f"Temperatura calibrada: {temperatura}")
    if temperatura > 1.5:
        print("T > 1.5: el modelo estaba sobreconfiado; se corrige en los pesos.")
    elif temperatura < 1.0:
        print("T < 1.0: el modelo estaba infraconfiado (poco habitual).")

    hornear_temperatura_en_pesos(modelo, temperatura)
    if centros is not None:
        calibrar_centroides(modelo, centros, embeddings_val, et_val)
    torch.save(modelo.state_dict(), ruta_modelo)
    print(f"Pesos calibrados guardados en: {ruta_modelo}")

    return temperatura
