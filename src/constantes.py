"""
Constantes de todo el programa.
"""

import os
from pathlib import Path
import torch

# -------------------------
# NOMBRE DE PRUEBA
# -------------------------
PRUEBA: str = "UMBRAL"

VARIABLES_GLOBALES: dict[str, object] = {
    

    "PRUEBA": PRUEBA,
    # -------------------------
    # Rutas principales
    # -------------------------
    "RUTA_BASE": Path(__file__).resolve().parent.parent / "data",
    "RUTA_SPLITS": Path(__file__).resolve().parent.parent / "data" / "splits" / PRUEBA,
    "RUTA_EMBEDDINGS": Path(__file__).resolve().parent.parent / "data" / "embeddings_procesado" / PRUEBA,
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
    "Achnanthidium_atomoides",
    "Achnanthidium_jackii",
    "Achnanthidium_pyrenaicum",
    "Achnanthidium_rivulare",
    "Achnanthidium_sp",
    "Amphora_indistincta",
    "Amphora_pediculus",
    "Aulacoseira_granulata",
    "Cocconeis_placentula",
    "Craticula_sp",
    "Crenotia_rumrichorum",
    "Cyclotella_atomus",
    "Cyclotella_meduanae",
    "Cyclotella_meneghiniana",
    "Denticula_tenuis",
    "Diatoma_tenuis",
    "Discostella_pseudostelligera",
    "Encyonema_minutum",
    "Fragilaria_deformis",
    "Fragilaria_famelica",
    "Gomphonella_olivacea",
    "Gomphonema_micropus",
    "Gomphonema_parvulum",
    "Gyrosigma_acuminatum",
    "Humidophila_contemnata",
    "Luticola_frequentissima",
    "Mayamaea_permitis",
    "Melosira_varians",
    "Meridion_circulare",
    "Navicula_cryptotenella",
    "Navicula_germainii",
    "Navicula_gregaria",
    "Navicula_lanceolata",
    "Navicula_recens",
    "Navicula_sp",
    "Navicula_tripunctata",
    "Navicula_veneta",
    "Nitzschia_dissipata",
    "Nitzschia_inconspicua",
    "Nitzschia_palea",
    "Nitzschia_soratensis",
    "Nitzschia_sp",
    "Planothidium_frequentissimum",
    "Planothidium_lanceolatum",
    "Rhoicosphenia_abbreviata",
    "Sellaphora_nigri",
    "Seminavis_strigosa",
    "Stephanodiscus_lacustris",
    "Surirella_brebissonii"
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
    "PESO_GENERO": 0.3, # Rango típico 0.1-0.5. Cuanto más alto, más importancia a la pérdida de género.
    "MINIMO_IMAGENES_POR_ESPECIE": 5,
    "UMBRAL_IMAGENES": 300, 
    "EXPONENTE_PESO_CLASE": 0.5, # Cuanto más alto, más importancia a las clases minoritarias. Rango 0.3-1.0
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
