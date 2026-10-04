import os
import re
import cv2
from ultralytics import YOLO
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import pandas as pd
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

ROOT = Path(__file__).resolve().parents[1]
YOLO_WEIGHTS = ROOT / 'Inferir' / 'yolo_dinov2' / 'yolo_best.pt'
RESNET_WEIGHTS = ROOT / 'Inferir' / 'yolo_dinov2' / 'resnet50_checkpoint_epoch50.pth'
DINO_WEIGHTS = ROOT / 'modelos' / '75_objetivo' / 'modelo_75_objetivo.pth'
CLASS_FILES = {
    'resnet': ROOT / 'Inferir' / 'txt_classes' / 'classes_78(resnet).txt',
    'dinov2': ROOT / 'Inferir' / 'txt_classes' / 'classes_77(dino).txt',
}
IMAGES_DIR = ROOT / 'Inferir' / 'imagenes_inferencia'
OUTPUT_BASE_DIR = ROOT / 'Inferir' / 'resultados_inferencia'


def extract_original_class(image_path):
    """Obtiene la clase original usando las dos primeras partes del nombre."""
    stem = Path(image_path).stem
    parts = [part for part in re.split(r"[_\s]+", stem) if part]
    return " ".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "")


class DinoV2CheckpointClassifier(torch.nn.Module):
    """Clasificador DINOv2 para las especies."""

    def __init__(self, backbone_name, species_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size
        self.tronco = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
        )
        self.cabeza_especie = torch.nn.Linear(256, species_classes)
        self.num_classes = species_classes

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.pooler_output
        cls_token = cls_token/cls_token.norm(dim=-1, keepdim=True)
        features = self.tronco(cls_token)
        return self.cabeza_especie(features)


def load_dinov2_checkpoint(checkpoint_path, device):
    """Carga el checkpoint .pth de DINOv2 con una única cabeza de especies."""
    backbone_name = 'facebook/dinov2-base'
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
    else:
        raise TypeError('El checkpoint DINOv2 debe contener un state_dict de PyTorch.')

    state_dict = {
        key[len('module.'):] if key.startswith('module.') else key: value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }
    species_weight = state_dict.get('cabeza_especie.weight')
    if species_weight is None:
        raise ValueError(
            'El checkpoint no contiene cabeza_especie.weight. '
            'Este script espera el checkpoint DINOv2 de una sola cabeza.'
        )

    classifier = DinoV2CheckpointClassifier(
        backbone_name,
        species_classes=species_weight.shape[0],
    )
    model_state = classifier.state_dict()
    compatible_state = {
        key: value for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    classifier.load_state_dict(compatible_state, strict=False)
    return classifier


def load_classifier(classifier_name, device):
    """Carga el clasificador seleccionado y su procesador de imagen."""
    class_path = CLASS_FILES[classifier_name]
    with open(class_path, 'r', encoding='utf-8') as classes_file:
        class_names = [line.strip() for line in classes_file if line.strip()]

    if classifier_name == 'resnet':
        classifier = models.resnet50(weights=None)
        num_features = classifier.fc.in_features
        classifier.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(num_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(256, len(class_names))
        )
        checkpoint = torch.load(
            RESNET_WEIGHTS,
            map_location=device
        )
        classifier.load_state_dict(checkpoint)
        classifier.eval().to(device)

        classifier_processor = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        dino_model_path = DINO_WEIGHTS
        if os.path.isfile(dino_model_path) and dino_model_path.lower().endswith(('.pth', '.pt')):
            classifier_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            classifier = load_dinov2_checkpoint(dino_model_path, device)
        else:
            classifier_processor = AutoImageProcessor.from_pretrained(dino_model_path)
            classifier = AutoModelForImageClassification.from_pretrained(dino_model_path)
        classifier.eval().to(device)

        model_num_labels = (
            classifier.config.num_labels
            if hasattr(classifier, 'config')
            else classifier.num_classes
        )
        if model_num_labels != len(class_names):
            raise ValueError(
                f'El modelo DINOv2 tiene {model_num_labels} clases, '
                f'pero el archivo de clases contiene {len(class_names)}.'
            )

    return classifier, classifier_processor, class_names


model = YOLO(str(YOLO_WEIGHTS))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier_name = input("Introduce el clasificador (resnet/dinov2): ").strip().lower()
if classifier_name not in ['resnet', 'dinov2']:
    print("Clasificador inválido. Usando 'resnet' por defecto.")
    classifier_name = 'resnet'

# Algunas instalaciones CUDA fallan con los kernels de DINOv2. Mantener
# DINOv2 en CPU permite usar YOLO/ResNet en GPU sin bloquear la inferencia.
classifier_device = torch.device('cpu') if classifier_name == 'dinov2' else device
if classifier_name == 'dinov2' and device.type == 'cuda':
    print('DINOv2 se ejecutará en CPU para evitar errores de kernels CUDA.')

classifier, classifier_processor, class_names = load_classifier(classifier_name, classifier_device)

images = IMAGES_DIR
output_base_dir = OUTPUT_BASE_DIR

# Introducir sufijo al inicio
suffix = input("Introduce el sufijo (original/normalizada): ").strip().lower()
if suffix not in ['original', 'normalizada']:
    print("Sufijo inválido. Usando 'original' por defecto.")
    suffix = 'original'

# Crear carpeta única con el nombre del sufijo
roi_output_dir = os.path.join(output_base_dir, classifier_name.capitalize(), suffix.capitalize())
crops_output_dir = os.path.join(roi_output_dir, 'crops')
bbox_output_dir = os.path.join(roi_output_dir, 'bbox')
os.makedirs(crops_output_dir, exist_ok=True)
os.makedirs(bbox_output_dir, exist_ok=True)

images_dir = Path(images)
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
images_path = [
    str(image_path)
    for image_path in images_dir.rglob('*')
    if image_path.is_file() and image_path.suffix.lower() in image_extensions
]
print(f'Imágenes encontradas (incluyendo subcarpetas): {len(images_path)}')

global_roi_count = 0
metrics_rows = []

for image_path in images_path:
    # Incorporar la ruta relativa para evitar colisiones entre subcarpetas.
    relative_image_path = Path(image_path).relative_to(images_dir).with_suffix('')
    image_name = '_'.join(relative_image_path.parts)
    original_class = extract_original_class(image_path)
    
    # Realizar predicción YOLO
    results = model.predict(
        image_path,
        save_txt=False,
        save=False,
        conf=0.3,
        imgsz=1024
    )
    
    # Procesar resultados
    if not results:
        print(f"No se pudo procesar la imagen con YOLO: {image_path}")
        continue
    result = results[0]
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print(f"No se pudo leer la imagen: {image_path}")
        continue
    height, width = image_cv.shape[:2]
    
    # Primero: clasificar todos los ROI con el modelo seleccionado.
    predictions = []
    if result.boxes is not None:
        for idx, box in enumerate(result.boxes):
            # Obtener coordenadas del box YOLO (solo usamos la detección, no la clasificación)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Asegurar que están dentro de los límites
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            # Recortar ROI
            roi = image_cv[y1:y2, x1:x2]
            
            roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            if classifier_name == 'resnet':
                model_inputs = classifier_processor(roi_pil).unsqueeze(0).to(classifier_device)
            else:
                model_inputs = {
                    key: value.to(classifier_device)
                    for key, value in classifier_processor(images=roi_pil, return_tensors='pt').items()
                }

            with torch.no_grad():
                model_output = classifier(model_inputs) if classifier_name == 'resnet' else classifier(**model_inputs)
                logits = model_output if isinstance(model_output, torch.Tensor) else model_output.logits
                prediction = torch.nn.functional.softmax(logits, dim=1)
                top_scores, top_class_ids = torch.topk(
                    prediction,
                    k=min(3, prediction.shape[1]),
                    dim=1
                )

            top_classes = [
                f"{class_names[class_id]} ({score:.4f})"
                for class_id, score in zip(
                    top_class_ids[0].tolist(),
                    top_scores[0].tolist()
                )
            ]
            class_id = top_class_ids[0, 0].item()
            confidence = top_scores[0, 0].item()
            
            class_name = class_names[class_id]
            
            predictions.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'top_classes': top_classes,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'roi': roi,
                'image_name': image_name,
                'original_class': original_class,
                'item': idx + 1
            })
    
    # Segundo: dibujar la clasificación del modelo seleccionado.
    result_image = image_cv.copy()
    if predictions:
        for pred in predictions:
            x1, y1, x2, y2 = pred['x1'], pred['y1'], pred['x2'], pred['y2']
            label = f"{pred['class_name']} ({pred['confidence']:.2f})"
            
            # Dibujar bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Dibujar etiqueta de ResNet
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_color = (0, 0, 255)  # Rojo en BGR
            bg_color = (255, 255, 255)  # Blanco
            thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            
            # Fondo para el texto
            cv2.rectangle(result_image, (x1, y1 - text_size[1] - 10), 
                          (x1 + text_size[0] + 5, y1), bg_color, -1)
            # Texto
            cv2.putText(result_image, label, (x1 + 2, y1 - 5), 
                        font, font_scale, font_color, thickness)
    
    # Guardar imagen con anotaciones en la carpeta única
    result_image_path = os.path.join(bbox_output_dir, f"{image_name}_{suffix}_annotated.png")
    cv2.imwrite(result_image_path, result_image)
    
    # Guardar anotaciones de texto con la clasificación seleccionada.
    txt_file_path = os.path.join(bbox_output_dir, f"{image_name}_{suffix}.txt")
    with open(txt_file_path, 'w') as txt_file:
        for pred in predictions:
            x1, y1, x2, y2 = pred['x1'], pred['y1'], pred['x2'], pred['y2']
            class_id = pred['class_id']
            
            # Calcular centro y dimensiones normalizadas YOLO
            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            
            # Formato YOLO normalizado
            txt_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    
    # Tercero: recortar y guardar los bounding boxes clasificados.
    for pred in predictions:
        global_roi_count += 1
        roi = pred['roi']
        class_name = pred['class_name']
        confidence = pred['confidence']
        source_image = pred['image_name']
        
        # Nombre del archivo ROI con clasificación ResNet e información de la imagen origen
        roi_filename = f"{source_image}_{class_name}_{confidence:.4f}_roi_{global_roi_count}.png"
        roi_path = os.path.join(crops_output_dir, roi_filename)
        
        # Guardar ROI
        cv2.imwrite(roi_path, roi)

        metrics_rows.append({
            'image': source_image,
            'item': pred['item'],
            'original_class': pred['original_class'],
            'classifier': classifier_name,
            'predicted_class_id': pred['class_id'],
            'predicted_class': class_name,
            'confidence': confidence,
            'top1': pred['top_classes'][0],
            'top2': pred['top_classes'][1] if len(pred['top_classes']) > 1 else '',
            'top3': pred['top_classes'][2] if len(pred['top_classes']) > 2 else '',
            'x1': pred['x1'],
            'y1': pred['y1'],
            'x2': pred['x2'],
            'y2': pred['y2'],
            'crop_file': roi_filename,
            'bbox_image_file': f"{image_name}_{suffix}_annotated.png"
        })
    
    print(f"✓ Procesada imagen: {image_name}")

metrics_path = os.path.join(roi_output_dir, f"classification_results_{suffix}.xlsx")
pd.DataFrame(metrics_rows).to_excel(metrics_path, index=False)

print(f"¡Inferencia completada! Total de ROIs guardados: {global_roi_count}")
print(f"Crops: {crops_output_dir}")
print(f"Imágenes con bbox: {bbox_output_dir}")
print(f"Resultados XLSX: {metrics_path}")
