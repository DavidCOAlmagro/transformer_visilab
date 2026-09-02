"""
Módulo para cargar y utilizar el modelo dinoV2 entrenado.
"""


import torch

from clasificador import ClasificadorDiatomeas
from constantes import VARIABLES_GLOBALES
from preparar_datos import construir_numero_genero


def cargar_modelo_entrenado() -> tuple[ClasificadorDiatomeas, list[str]]:
    """
    Carga el mejor modelo del experimento actual y devuelve también
    las especies ordenadas según sus índices de salida.
    """
    especies_ordenadas = sorted(VARIABLES_GLOBALES["ESPECIES_FILTRADAS"])
    numero_genero = construir_numero_genero(
 VARIABLES_GLOBALES["ESPECIES_FILTRADAS"])

    modelo = ClasificadorDiatomeas(num_clases=len(especies_ordenadas),
        num_generos=len(numero_genero)).to(VARIABLES_GLOBALES["DEVICE"])

    ruta_pesos = (
        VARIABLES_GLOBALES["RUTA_MODELOS"]
        / VARIABLES_GLOBALES["PRUEBA"]
        / "mejor_modelo.pth"
    )

    if not ruta_pesos.is_file():
        raise FileNotFoundError(f"No se encontró el modelo entrenado en: {ruta_pesos}\n"
            "Entrena el modelo antes de continuar."
        )

    pesos = torch.load(ruta_pesos,
        map_location=VARIABLES_GLOBALES["DEVICE"], weights_only=True)
    modelo.load_state_dict(pesos)
    modelo.eval()

    return modelo, especies_ordenadas