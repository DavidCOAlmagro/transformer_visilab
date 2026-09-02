"""
Constantes de todo el programa.
"""

import os
from pathlib import Path
import torch

VARIABLES_GLOBALES: dict[str, object] = {

    # -------------------------
    # Rutas principales
    # -------------------------
    "RUTA_BASE": Path(__file__).resolve().parent.parent / "data",
    "RUTA_SPLITS": Path(__file__).resolve().parent.parent / "data" / "splits",
    "RUTA_EMBEDDINGS": Path(__file__).resolve().parent.parent / "data" / "embeddings_procesado",
    "RUTA_MODELOS": Path(__file__).resolve().parent.parent / "modelos",

    # -------------------------
    # Configuración de imágenes
    # -------------------------
    "EXTENSIONES_VALIDAS": {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"},

    # -------------------------
    # Token HuggingFace
    # -------------------------
    "HF_TOKEN": os.environ.get("HF_TOKEN", "") or None,

# -------------------------
    # Especies filtradas
    # -------------------------
    "ESPECIES_FILTRADAS": {
        "Achnanthidium_minutissimum",
        "Nitzschia_inconspicua",
        "Cocconeis_placentula_var_euglypta",
        "Achnanthidium_pyrenaicum",
        "Fistulifera_saprophila",
        "Achnanthidium_sp",
        "Nitzschia_sp",
        "Navicula_caterva",
        "Achnanthidium_rivulare",
        "Gomphonema_pumilum",
        "Seminavis_strigosa",
        "Achnanthidium_rostropyrenaicum",
        "Fragilaria_sp",
        "Gomphonema_rhombicum",
        "Nitzschia_palea_var_palea",
        "Halamphora_sp",
        "Navicula_sp",
        "Navicula_cryptotenella",
        "Epithemia_adnata",
        "Mayamaea_permitis"
    },
 # De las 20 especies más enviadas en el excel:
 # - Eunotia exigua solo 6 imágenes en dataset, insuficiente para entrenar.
 # - Fragilaria radians: solo 3 imágenes, mismo problema.
 # - Gomphonema pumilum var. rigidum: ninguna, existe pumilum a secas
 # - Achnanthidium delmontii: daba malos resultados
 # - Achnanthidium rostropyrenaicum: muy pocas imágenes
    # -------------------------
    # Parámetros
    # -------------------------
    "BATCH_SIZE": 32,
    "DIM_EMBEDDING": 768,
    "num_epocas": 40,
    "NUM_WORKERS": 4,
    "PIN_MEMORY": True,
    "PERSISTENT_WORKERS": True,
    "EPOCAS_WARMUP": 3,
    "PACIENCIA": 7,
    "UMBRAL_CONF": 0.80, # Comprobado con validación, 0.80 es un buen valor para filtrar predicciones poco confiables.
    "LEARNING_RATE": 0.0003,
    "WEIGHT_DECAY": 0.0001,
    "LABEL_SMOOTHING": 0.05, # No confia mucho en sus predicciones.
    "PRUEBA": "20_especies",
    "PESO_GENERO": 0.3, # Rango típico 0.1-0.5. Cuanto más alto, más importancia a la pérdida de género.
    "MINIMO_IMAGENES_POR_ESPECIE": 5, 
    # -------------------------
    # Clasificador
    # -------------------------
    "DIM_CAPA_1": 512,
    "DIM_CAPA_2": 256,
    "DROPOUT_CAPA_1": 0.3,
    "DROPOUT_CAPA_2": 0.2,
    
    # -------------------------
    # CARPETAS CON IMAGENES
    # -------------------------
    "GRUPOS_DATOS": [
    "Common_Species",
    "Unique_Species",
    "Seleccion_5_especies_por_especie",
    "UDE_Diatoms_84k_normalizadas_reinhard",
    ],
    # -------------------------
    # Device
    # -------------------------
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}
