"""
Guarda en un .json la configuración del modelo y los 
hiperparámetros utilizados en el entrenamiento.
"""
import json
from constantes import VARIABLES_GLOBALES

def exportar_config() -> None:
    ruta_carpeta = VARIABLES_GLOBALES["RUTA_MODELOS"] / VARIABLES_GLOBALES["PRUEBA"]
    ruta_metadatos = ruta_carpeta / "metadatos_modelo.json"

    with open(ruta_metadatos, "r", encoding="utf-8") as f:
        especies = json.load(f)["especies_filtradas"]

    config = {
        "modelo_base": "facebook/dinov2-base",
        "tamano_imagen_entrada": "224x224 (fijado por AutoImageProcessor de dinov2-base)",
        "dim_embedding": VARIABLES_GLOBALES["DIM_EMBEDDING"],
        "arquitectura_clasificador": "768 -> 512 -> 256 (ReLU + Dropout) -> cabeza especie / cabeza genero",
        "dim_capa_1": VARIABLES_GLOBALES["DIM_CAPA_1"],
        "dim_capa_2": VARIABLES_GLOBALES["DIM_CAPA_2"],
        "dropout_capa_1": VARIABLES_GLOBALES["DROPOUT_CAPA_1"],
        "dropout_capa_2": VARIABLES_GLOBALES["DROPOUT_CAPA_2"],
        "batch_size": VARIABLES_GLOBALES["BATCH_SIZE"],
        "learning_rate": VARIABLES_GLOBALES["LEARNING_RATE"],
        "weight_decay": VARIABLES_GLOBALES["WEIGHT_DECAY"],
        "label_smoothing": VARIABLES_GLOBALES["LABEL_SMOOTHING"],
        "peso_genero": VARIABLES_GLOBALES["PESO_GENERO"],
        "exponente_peso_clase": VARIABLES_GLOBALES["EXPONENTE_PESO_CLASE"],
        "lambda_center_loss": VARIABLES_GLOBALES["LAMBDA_CENTER_LOSS"],
        "umbral_confianza": VARIABLES_GLOBALES["UMBRAL_CONF"],
        "num_especies": len(especies),
        "especies": sorted(especies),
    }

    ruta_salida = ruta_carpeta / "config_modelo.json"
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Config exportada en: {ruta_salida}")

if __name__ == "__main__":
    exportar_config()