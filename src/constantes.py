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
    "HF_TOKEN": os.environ.get("HF_TOKEN", ""),

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
        "Gomphonema_pumilum"
    },
    "ESPECIES_MINORITARIAS": {
        "Navicula_caterva",
        "Cocconeis_placentula_var_euglypta",
        "Gomphonema_pumilum"
    },
    "ESPECIES_MUY_MINORITARIAS": {
        
    },

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
    "PACIENCIA": 5,
    # -------------------------
    # Numero de pruebas
    # -------------------------
    "PRUEBA" : "1",

    # -------------------------
    # Device
    # -------------------------
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}
