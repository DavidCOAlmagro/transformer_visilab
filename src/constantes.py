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
# -------------------------
    # Especies agrupadas por tanda de dificultad
    # (según F1 medio del género en la prueba de 92 especies)
    # -------------------------
    "TANDAS_ESPECIES": {
        1: {  # Fáciles (F1 medio de género >= 0.70)
            "Aulacoseira_granulata", "Cocconeis_pediculus",
            "Cocconeis_placentula_var_euglypta", "Cocconeis_sp",
            "Cymbella_excisiformis_var_excisiformis", "Cymbella_parva",
            "Denticula_tenuis", "Diatoma_moniliformis", "Diatoma_tenuis",
            "Diatoma_vulgaris", "Discostella_pseudostelligera",
            "Encyonopsis_minuta", "Fragilaria_deformis", "Fragilaria_vaucheriae",
            "Gomphonella_olivacea", "Humidophila_contenta", "Melosira_varians",
            "Meridion_circulare", "Rhoicosphenia_abbreviata", "Seminavis_strigosa",
            "Staurosira_venter", "Surirella_brebissonii", "Tabularia_fasciculata",
            "Ulnaria_ulna",
        },
        2: {  # Medias (F1 medio de género 0.55-0.70)
            "Achnanthidium_atomoides", "Achnanthidium_catenatum",
            "Achnanthidium_delmontii", "Achnanthidium_druartii",
            "Achnanthidium_eutrophilum", "Achnanthidium_lineare",
            "Achnanthidium_minutissimum", "Achnanthidium_pyrenaicum",
            "Achnanthidium_rivulare", "Achnanthidium_sp",
            "Achnanthidium_straubianum", "Achnanthidium_subatomus",
            "Amphora_inariensis", "Amphora_pediculus", "Caloneis_lancettula",
            "Encyonema_minutum", "Encyonema_reichardtii", "Encyonema_silesiacum",
            "Encyonema_sp", "Encyonema_ventricosum", "Gomphonema_minusculum",
            "Gomphonema_minutum", "Gomphonema_parvulum", "Gomphonema_pumilum",
            "Gomphonema_pumilum_var_elegans", "Gomphonema_sp", "Halamphora_sp",
            "Mayamaea_permitis", "Nitzschia_amphibia", "Nitzschia_capitellata",
            "Nitzschia_dissipata", "Nitzschia_fonticola", "Nitzschia_frustulum",
            "Nitzschia_frustulum_var_frustulum", "Nitzschia_inconspicua",
            "Nitzschia_palea", "Nitzschia_paleacea", "Nitzschia_sociabilis",
            "Nitzschia_soratensis", "Nitzschia_sp", "Nitzschia_supralitorea",
            "Planothidium_frequentissimum", "Planothidium_lanceolatum",
            "Planothidium_sp", "Reimeria_sinuata", "Reimeria_uniseriata",
        },
        3: {  # Difíciles (F1 medio de género < 0.55)
            "Conticribra_weissflogii", "Cyclotella_atomus", "Fistulifera_saprophila",
            "Navicula_antonii", "Navicula_caterva", "Navicula_cryptocephala",
            "Navicula_cryptotenella", "Navicula_cryptotenelloides",
            "Navicula_erifuga", "Navicula_germainii", "Navicula_gregaria",
            "Navicula_lanceolata", "Navicula_metareichardtiana", "Navicula_recens",
            "Navicula_rostellata", "Navicula_simulata", "Navicula_sp",
            "Navicula_veneta", "Sellaphora_crassulexigua", "Sellaphora_nigri",
            "Sellaphora_pupula", "Sellaphora_sp", "Stephanodiscus_hantzschii",
        },
    },
    "TANDA_ACTUAL": 3,  # 1, 2 o 3 — cambia esto para elegir qué tanda entrenar


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
    "UMBRAL_CONF": 0.60,
    "LEARNING_RATE": 0.0003,
    "WEIGHT_DECAY": 0.0001,
    "LABEL_SMOOTHING": 0.05, # No confia mucho en sus predicciones.

    # -------------------------
    # Device
    # -------------------------
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# Especies de la tanda activa: esto es lo que usa el entrenamiento/evaluación
# Usar las claves dentro del dict para evitar referencias indefinidas dentro del literal
VARIABLES_GLOBALES["ESPECIES_FILTRADAS"] = VARIABLES_GLOBALES["TANDAS_ESPECIES"][VARIABLES_GLOBALES["TANDA_ACTUAL"]]

# Carpeta de resultados separada por tanda (modelos/tanda_1/, tanda_2/, tanda_3/)
VARIABLES_GLOBALES["PRUEBA"] = f"tanda_{VARIABLES_GLOBALES['TANDA_ACTUAL']}"
