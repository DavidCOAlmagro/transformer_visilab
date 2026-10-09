"""
--------------------------------------
Carga de DINOv2 y aumentación de datos para el entrenamiento.
--------------------------------------
"""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import try_to_load_from_cache
from torchvision import transforms


def resolver_modelo_dinov2() -> tuple[str, bool]:
    """Devuelve una ruta local utilizable (y si es solo local) o el identificador remoto del modelo."""
    modelo_id = "facebook/dinov2-base"
    ruta_configurada = os.environ.get("DINOV2_MODEL_PATH")
    if ruta_configurada:
        ruta = Path(ruta_configurada).expanduser()
        if not ruta.is_dir():
            raise FileNotFoundError(f"DINOV2_MODEL_PATH no es una carpeta válida: {ruta}")
        return str(ruta), True

    archivos_requeridos = ("config.json", "preprocessor_config.json", "model.safetensors")
    rutas_cache = [try_to_load_from_cache(modelo_id, archivo) for archivo in archivos_requeridos]
    if all(isinstance(ruta, str) for ruta in rutas_cache):
        carpetas_cache = {str(Path(ruta).parent) for ruta in rutas_cache}
        if len(carpetas_cache) == 1:
            return next(iter(carpetas_cache)), True
    return modelo_id, False


def crear_augmentation() -> transforms.Compose:
    """Giros, rotación, pequeño desplazamiento y escala, brillo/contraste y desenfoque leve."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15, fill=(128, 128, 128)),
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.97, 1.03), fill=(128, 128, 128)),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
    ])
