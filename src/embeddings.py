"""
--------------------------------------
Extracción de embeddings con DINOv2 para imágenes de diatomeas.

Este script recorre las carpetas de imágenes del proyecto, procesa cada
archivo con el AutoImageProcessor y obtiene un embedding de 768 valores
usando DINOv2. Los embeddings y sus etiquetas se devuelven listos para
guardarse y usarse para clasificación.

He utilizado embeddings ya que es mucho más barato computacionalmente que entrenar un modelo 
desde cero, o hacer fine tuning de DINOv2, pasar imagenes por un modelo tan grande es carisimo
y no aprende ya que tiene pesos fijos por lo que habria que hacerse constantemente. Por eso
uso: Transfer Learning + Feature extraction(extraccion embeddings) + 
entrenamiento de un clasificador supervisado (MLP) sobre esos embeddings congelados.
--------------------------------------
"""
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms
from tqdm import tqdm
from constantes import VARIABLES_GLOBALES


def inicializar_dinov2() -> tuple[AutoImageProcessor, AutoModel, torch.device, transforms.Compose]:
    """
    Inicializa el procesador de imágenes(imagen a tensor), el modelo DINOv2
    y el dispositivo (CPU o GPU).
    """
    processor: AutoImageProcessor = AutoImageProcessor.from_pretrained(
        'facebook/dinov2-base')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: AutoModel = AutoModel.from_pretrained(
        'facebook/dinov2-base', token=VARIABLES_GLOBALES["HF_TOKEN"])
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    augmentation: transforms.Compose = crear_augmentation()

    return processor, model, device, augmentation


def crear_augmentation() -> transforms.Compose:
    """
    Crea la augmentation del tipo giro horizontal, vertical, rotación,blur y color.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(360),
        # Rotación pequeña de hasta 5 grados, escalado y traslación
        transforms.RandomAffine(
            degrees=0,
            translate=(0.02, 0.02),
            scale=(0.97, 1.03)
        ),
        # Ajuste aleatorio de brillo y contraste
        transforms.ColorJitter(
            brightness=0.08,
            contrast=0.08
        ),
        # Aplicación de un desenfoque gaussiano aleatorio
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 1.0)
        )
    ])

# Decorador @torch.inference_mode() indica que no se calcularán gradientes, es mas rapido
# que @torch.no_grad()
@torch.inference_mode()
def get_embedding(ruta_imagen: str, processor: AutoImageProcessor, model: AutoModel,
                  device: torch.device, augmentation: transforms.Compose,
                  is_train: bool) -> torch.Tensor:
    """
    Dada la ruta de una imagen, devuelve su embedding de 768 valores
    calculado por DINOv2. Aplica augmentation si is_train=True.
    "pixel_values": tensor [1, 3, 224, 224] 1 imagen, 3 canales, 224x224 píxeles
    """

    imagen: Image = Image.open(ruta_imagen).convert("RGB")
    if is_train:
        imagen = augmentation(imagen)

    # El procesador: Redimensiona(224x224), Normaliza([-1, 1]) y convierte a tensor.
    inputs: dict[str, torch.Tensor] = processor(images=imagen, return_tensors="pt")

    # Se mueven los tensores al mismo dispositivo que el modelo (CPU o GPU)
    for i in inputs:
        inputs[i] = inputs[i].to(device)

    # Se pasa la imagen por dinov2 y se obtienen los nuevos tensores
    outputs: dict[str, torch.Tensor] = model(pixel_values=inputs["pixel_values"])
    
    # Usamos pooler_output: el token CLS, recomendado para clasificación
    
    embedding = outputs.pooler_output  # forma: [1, 768]
    del outputs, inputs, imagen  # Liberamos memoria de GPU
    
    # Asegura float32 para compatibilidad con el clasificador, que espera float32
    embedding = embedding.float()
    # Normaliza el embedding para que tenga norma 1, lo que ayuda a la estabilidad del entrenamiento
    embedding = embedding / embedding.norm(dim=1, keepdim=True)
    embedding = embedding.cpu()

    return embedding


def calcular_embeddings(imagenes: list[tuple[Path, str]], processor: AutoImageProcessor,
                        model: AutoModel,
                        device: torch.device, augmentation: transforms.Compose,
                        is_train: bool, copias_por_especie: dict[str, int]
                        ) -> dict[str, torch.Tensor | list[str]]:
    """
    Calcula el embedding de cada imagen de la lista y los guarda todos juntos
    con sus etiquetas correspondientes.
    copias_por_especie indica, para train, cuántas copias de augmentation
    recibe cada especie, según su desbalance.
    """

    if copias_por_especie is None:
        copias_por_especie = {}
    # En vez de tuplas, usamos dos listas ya que torch.cat requiere una lista de tensores solamente
    lista_embeddings: list[torch.Tensor] = []
    lista_etiquetas: list[str] = []

    for ruta, especie in tqdm(imagenes, desc="Calculando embeddings"):
        valid: bool = True
        try:
            embedding: torch.Tensor = get_embedding(
                ruta, processor, model, device,
                augmentation, is_train=False
            )
        except FileNotFoundError:
            print(f"Imagen no encontrada, se omite: {ruta}")
            valid = False
        if valid:
            lista_embeddings.append(embedding.cpu())
            lista_etiquetas.append(especie)
            # Si es train, añadimos ADEMÁS una augmentation de la imagen
            if is_train:
                embedding_aumentado: torch.Tensor = get_embedding(
                    ruta, processor, model, device, augmentation, is_train=True)
                lista_embeddings.append(embedding_aumentado.cpu())
                lista_etiquetas.append(especie)
                
            # Copias extra según el desbalance de la especie (0 si está
            # por encima de la mediana, hasta el tope para las más raras)
            copias_extra: int = copias_por_especie.get(especie, 0)
            for _ in range(copias_extra):
                embedding_extra = get_embedding(
                    ruta, processor, model, device,
                    augmentation, is_train=True
                )
                lista_embeddings.append(embedding_extra.cpu())
                lista_etiquetas.append(especie)


    # Apilamos todos los embeddings individuales [1, 768] en uno solo [N, 768]
    embeddings_finales: torch.Tensor = torch.cat(lista_embeddings, dim=0)

    dict_embeddings: dict[str, torch.Tensor | list[str]] = {
        "embeddings": embeddings_finales,
        "etiquetas": lista_etiquetas
    }

    return dict_embeddings
