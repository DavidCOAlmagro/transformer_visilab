"""
Constantes de todo el programa.
"""

import os
from pathlib import Path
import torch

# -------------------------
# NOMBRE DE PRUEBA
# -------------------------
PRUEBA: str = "75_objetivo"  

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
    # Especies filtradas "Planothidium_fp" solo tiene 3 imágenes, insuficiente para entrenar. 
    # -------------------------
    "ESPECIES_FILTRADAS": {
     "Achnanthidium_atomoides",
    "Achnanthidium_caravelense",
    "Achnanthidium_catenatum",
    "Achnanthidium_druartii",
    "Achnanthidium_eutrophilum",
    "Achnanthidium_exile",
    "Achnanthidium_fp",
    "Achnanthidium_minutissimum",
    "Achnanthidium_pyrenaicum",
    "Achnanthidium_rivulare",
    "Amphora_pediculus",
    "Aulacoseira_granulata",
    "Cocconeis_lineata",
    "Cocconeis_pediculus",
    "Cocconeis_placentula_var_euglypta",
    "Craticula_accomoda",
    "Cyclotella_atomus",
    "Cymbella_excisiformis_var_excisiformis",
    "Cymbella_parva",
    "Debris",
    "Denticula_tenuis",
    "Diatoma_moniliformis",
    "Diatoma_vulgaris",
    "Discostella_pseudostelligera",
    "Encyonema_fp",
    "Encyonema_minutum",
    "Encyonema_reichardtii",
    "Encyonema_silesiacum",
    "Encyonema_ventricosum",
    "Encyonopsis_fp",
    "Encyonopsis_minuta",
    "Epithemia_adnata",
    "Epithemia_fp",
    "Fistulifera_saprophila",
    "Fragilaria_fp",
    "Fragilaria_pararumpens",
    "Fragilaria_perminuta",
    "Fragilaria_vaucheriae",
    "Fragments",
    "Gomphonema_angustivalva",
    "Gomphonema_fp",
    "Gomphonema_minusculum",
    "Gomphonema_minutum",
    "Gomphonema_parvulum_f_saprophilum",
    "Gomphonema_pumilum_var_elegans",
    "Gomphonema_rhombicum",
    "Humidophila_contenta",
    "Karayevia_clevei_var_clevei",
    "Mayamaea_permitis",
    "Melosira_fp",
    "Melosira_varians",
    "Navicula_capitatoradiata",
    "Navicula_cryptotenella",
    "Navicula_cryptotenelloides",
    "Navicula_gregaria",
    "Navicula_lanceolata",
    "Navicula_notha",
    "Navicula_tripunctata",
    "Nitzschia_amphibia",
    "Nitzschia_capitellata",
    "Nitzschia_desertorum",
    "Nitzschia_fonticola",
    "Nitzschia_fp",
    "Nitzschia_frustulum_var_frustulum",
    "Nitzschia_inconspicua",
    "Nitzschia_palea_var_palea",
    "Nitzschia_paleacea",
    "Nitzschia_soratensis",
    "Planothidium_frequentissimum",
    "Planothidium_lanceolatum",
    "Reimeria_sinuata",
    "Reimeria_sp",
    "Rhoicosphenia_abbreviata",
    "Staurosira_venter",
    "Stephanodiscus_hantzschii",
    "Tabellaria_flocculosa",
    "Tabellaria_fp"
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
    "PESO_GENERO": 0.4, # Rango típico 0.1-0.5. Cuanto más alto, más importancia a la pérdida de género.
    "MINIMO_IMAGENES_POR_ESPECIE": 5,
    "EXPONENTE_PESO_CLASE": 0.3, # Cuanto más alto, más importancia a las clases minoritarias. Rango 0.3-1.0
    "LAMBDA_CENTER_LOSS": 0.01, # Cuanto más alto, más importancia a la pérdida de center loss. Rango 0.001-0.1
    # -------------------------
    # Clasificador
    # -------------------------
    "DIM_CAPA_1": 512,
    "DIM_CAPA_2": 256,
    "DROPOUT_CAPA_1": 0.3,
    "DROPOUT_CAPA_2": 0.2,
    # -------------------------
    # Device
    # -------------------------
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}
