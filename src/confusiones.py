"""
--------------------------------------
Muestra qué especies se confunden con cuáles en el conjunto de test.
--------------------------------------
"""
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from constantes import VARIABLES_GLOBALES
from clasificador import ClasificadorDiatomeas
from preparar_datos import get_datos, codificacion, construir_numero_genero
from dataset import MyDataset


@torch.no_grad()
def obtener_predicciones(modelo, dataloader) -> tuple[list[int], list[int]]:
    """Pasa todo el dataloader por el modelo y devuelve (y_true, y_pred)."""
    modelo.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch_embeddings, batch_etiquetas in dataloader:
        batch_embeddings = batch_embeddings.to(next(modelo.parameters()).device)
        logits_especie, _ = modelo(batch_embeddings)
        _, indice_predicciones = torch.max(logits_especie, 1)
        y_true.extend(batch_etiquetas.tolist())
        y_pred.extend(indice_predicciones.cpu().tolist())

    return y_true, y_pred


def cargar_modelo(num_clases: int, num_generos: int) -> ClasificadorDiatomeas:
    """Carga el modelo entrenado con los pesos guardados en disco."""
    modelo = ClasificadorDiatomeas(num_clases, num_generos).to(VARIABLES_GLOBALES["DEVICE"])
    ruta_pesos = VARIABLES_GLOBALES["RUTA_MODELOS"] / VARIABLES_GLOBALES["PRUEBA"] / "mejor_modelo.pth"
    pesos = torch.load(ruta_pesos, map_location=VARIABLES_GLOBALES["DEVICE"], weights_only=True)
    modelo.load_state_dict(pesos)
    modelo.eval()
    return modelo


def main() -> None:
    """Carga el modelo, evalúa en test e imprime los pares de especies confundidas."""
    datos_test = get_datos("test")
    emb_test, et_test, numero_especie = codificacion(datos_test)
    dataset_test = MyDataset(emb_test, et_test)
    dataloader_test = DataLoader(dataset_test, batch_size=VARIABLES_GLOBALES["BATCH_SIZE"])

    num_clases = len(VARIABLES_GLOBALES["ESPECIES_FILTRADAS"])
    numero_genero = construir_numero_genero(VARIABLES_GLOBALES["ESPECIES_FILTRADAS"])
    num_generos = len(numero_genero)

    modelo = cargar_modelo(num_clases, num_generos)
    y_true, y_pred = obtener_predicciones(modelo, dataloader_test)

    nombres_clases = sorted(numero_especie, key=numero_especie.get)
    matriz = confusion_matrix(y_true, y_pred)

    print("\n=== Especies que se confunden entre sí ===\n")
    for i, especie_real in enumerate(nombres_clases):
        for j, especie_predicha in enumerate(nombres_clases):
            if i != j and matriz[i, j] > 0:
                print(f"{especie_real:35s} -> {especie_predicha:35s} : {matriz[i, j]} veces")


if __name__ == "__main__":
    main()