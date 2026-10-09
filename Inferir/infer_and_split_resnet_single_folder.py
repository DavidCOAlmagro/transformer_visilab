"""Inferencia YOLO recursiva y clasificación DINOv2/ResNet50.

Los pesos no se distribuyen con el repositorio: ``*.pt`` y ``*.pth`` están
excluidos por ``.gitignore``. Este módulo no ejecuta inferencia al importarse;
use ``python Inferir/infer_and_split_resnet_single_folder.py --help``.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from embeddings import resolver_modelo_dinov2
from preprocesado import preparar_para_dinov2

DEFAULT_INPUT = ROOT / "Inferir" / "imagenes_inferencia"
DEFAULT_YOLO_WEIGHTS = ROOT / "Inferir" / "yolo_dinov2" / "yolo_best.pt"
DEFAULT_DINO_WEIGHTS = ROOT / "modelos" / "75_objetivo_ft" / "modelo_75_objetivo_ft.pth"
DEFAULT_RESNET_WEIGHTS = ROOT / "Inferir" / "yolo_dinov2" / "resnet50_checkpoint_epoch50.pth"
DEFAULT_DINO_CLASSES = ROOT / "Inferir" / "txt_classes" / "classes_77(dino).txt"
DEFAULT_RESNET_CLASSES = ROOT / "Inferir" / "txt_classes" / "classes_78(resnet).txt"
# Único sitio donde se decide dónde van los resultados: <RESULTS_ROOT>/<modelo>/<prueba>/
RESULTS_ROOT = ROOT / "Inferir" / "resultados_inferencia"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_LOAD_WORKERS = 3
IMAGE_PREFETCH = 6
IMAGE_SAVE_WORKERS = 4
MAX_PENDING_SAVES = 32
GENERATED_DIRS = {
    "bbox",
    "crops",
    "output",
    "outputs",
    "result",
    "results",
    "resultados",
    "resultados_inferencia",
    "runs",
}
_PIL_IMAGE_OPEN = Image.open


def extract_original_class(image_path: str | Path) -> str:
    """Obtiene la clase indicada por las dos primeras partes del nombre."""
    import re

    parts = [part for part in re.split(r"[_\s]+", Path(image_path).stem) if part]
    return " ".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "")


def load_torch_checkpoint(path: Path, device: torch.device) -> Any:
    """Carga un checkpoint usando weights_only cuando el runtime lo admite."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError as error:
        if "weights_only" not in str(error):
            raise
        return torch.load(path, map_location=device)


def read_classes(path: Path, expected: int, label: str) -> list[str]:
    """Lee una clase por línea, tolerando BOM, comentarios y líneas vacías."""
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el fichero de clases {label}: {path}")
    classes = [
        line.strip().lstrip("\ufeff")
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(classes) != expected:
        raise ValueError(
            f"{label} debe contener exactamente {expected} clases; contiene {len(classes)}: {path}"
        )
    if len(set(classes)) != len(classes):
        raise ValueError(f"{label} contiene clases duplicadas: {path}")
    return classes


def read_metadata_classes(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el metadata del modelo DINOv2: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Metadata DINOv2 inválido: {path}") from error
    classes = data.get("especies_filtradas") or data.get("especies")
    if not isinstance(classes, list) or not all(isinstance(item, str) for item in classes):
        raise ValueError(f"El metadata no contiene una lista de clases: {path}")
    if len(classes) != 77 or len(set(classes)) != 77:
        raise ValueError(f"El metadata DINOv2 debe contener exactamente 77 clases: {path}")
    return list(classes)


def resolve_dino_classes(weights: Path, classes_path: Path | None) -> list[str]:
    """Prioriza metadata y comprueba cualquier ``clases.txt`` disponible."""
    metadata_path = weights.parent / "metadatos_modelo.json"
    metadata = read_metadata_classes(metadata_path) if metadata_path.is_file() else None
    if classes_path is not None and not classes_path.is_file():
        raise FileNotFoundError(f"No se encontró el fichero de clases DINOv2: {classes_path}")
    candidates = [classes_path] if classes_path else []
    candidates.extend(
        path for path in (
            weights.parent / "clases.txt",
            weights.parent / "classes.txt",
            DEFAULT_DINO_CLASSES,
        ) if path not in candidates
    )
    class_file = next((path for path in candidates if path.is_file()), None)
    classes = read_classes(class_file, 77, "clases DINOv2") if class_file else None
    if metadata is not None and classes is not None and classes != metadata:
        raise ValueError(
            "El orden de clases DINOv2 no coincide entre metadatos_modelo.json y clases.txt."
        )
    if metadata is not None:
        return metadata
    if classes is not None:
        return classes
    raise FileNotFoundError(
        "No se encontró una lista DINOv2 de 77 clases ni metadatos_modelo.json."
    )


def top_predictions(logits: torch.Tensor, classes: list[str], threshold: float) -> list[dict[str, Any]]:
    if not 0 <= threshold <= 1:
        raise ValueError("El umbral de confianza debe estar entre 0 y 1.")
    if logits.ndim != 2 or logits.shape[1] != len(classes):
        raise ValueError(
            f"Dimensiones incompatibles: logits={tuple(logits.shape)}, clases={len(classes)}."
        )
    probabilities = torch.softmax(logits.float(), dim=1)
    values, indices = torch.topk(probabilities, k=min(3, len(classes)), dim=1)
    predictions = []
    for row_values, row_indices in zip(values, indices):
        top = [
            (classes[index], round(float(value) * 100, 2))
            for value, index in zip(row_values.tolist(), row_indices.tolist())
        ]
        predicted = top[0][0]
        low_confidence = top[0][1] < threshold * 100
        reasons = []
        if low_confidence:
            reasons.append("confianza_baja")
        predictions.append({
            # La clase ganadora siempre se conserva, incluso si el dataset la
            # denomina "Desconocida". No se inventa una clase ni se reemplaza
            # una predicción por vacío por pertenecer a una lista especial.
            "especie_predicha": predicted,
            "especie_mas_parecida": predicted,
            "confianza": top[0][1],
            "top": top,
            "revisar": low_confidence,
            "motivo_revision": ",".join(reasons),
        })
    return predictions


def _state_dict(checkpoint: Any, label: str) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Los pesos {label} no contienen un state_dict válido.")
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }


def backbone_state(state: dict[str, Any]) -> dict[str, Any]:
    """Extracts fine-tuned DINOv2 backbone weights (``backbone.*`` keys), if any.

    Checkpoints from the partial fine-tuning (spec 006) store the whole backbone
    under ``backbone.``; older checkpoints only contain the MLP and return {}.
    """
    return {
        key.removeprefix("backbone."): value
        for key, value in state.items()
        if key.startswith("backbone.")
    }


class ResNetClassifier:
    def __init__(self, weights: Path, classes: list[str], device: torch.device) -> None:
        from torchvision import models, transforms

        if not weights.is_file():
            raise FileNotFoundError(f"No se encontraron los pesos ResNet: {weights}")
        self.device = device
        self.classes = classes
        self.model = models.resnet50(weights=None)
        features = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 78),
        )
        state = _state_dict(load_torch_checkpoint(weights, device), "ResNet")
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError(
                "Los pesos ResNet no coinciden con ResNet50 de 78 clases."
            ) from error
        self.model.to(device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    @torch.inference_mode()
    def predict_batch(self, images: list[Image.Image], threshold: float) -> list[dict[str, Any]]:
        if not images:
            return []
        batch = torch.stack(
            [self.transform(image.convert("RGB")) for image in images]
        ).to(self.device)
        with torch.autocast(
            self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            logits = self.model(batch)
        return top_predictions(logits, self.classes, threshold)


class DinoClassifier:
    def __init__(self, weights: Path, classes: list[str], device: torch.device) -> None:
        from transformers import AutoImageProcessor, AutoModel

        if not weights.is_file():
            raise FileNotFoundError(f"No se encontraron los pesos DINOv2: {weights}")
        self.device = device
        self.classes = classes
        try:
            state = _state_dict(load_torch_checkpoint(weights, device), "DINOv2")
        except (OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(
                f"No se pudo cargar el checkpoint DINOv2 {weights}. "
                "Comprueba que sea un checkpoint PyTorch válido y no esté incompleto."
            ) from error
        # Igual que en el entrenamiento: caché local o DINOV2_MODEL_PATH, sin depender de la red
        model_path, local_only = resolver_modelo_dinov2()
        try:
            self.backbone = AutoModel.from_pretrained(model_path, local_files_only=local_only)
        except Exception as error:
            raise RuntimeError(
                "No se pudo cargar el backbone DINOv2 'facebook/dinov2-base'. "
                "Comprueba la caché/conectividad y que los pesos del checkpoint "
                "correspondan a ese backbone."
            ) from error
        tuned_backbone = backbone_state(state)
        if tuned_backbone:
            try:
                self.backbone.load_state_dict(tuned_backbone, strict=True)
            except RuntimeError as error:
                raise ValueError(
                    f"The fine-tuned DINOv2 backbone in {weights} does not match 'facebook/dinov2-base'."
                ) from error
        hidden = self.backbone.config.hidden_size
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(hidden, 512), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
        )
        self.head = torch.nn.Linear(256, 77)
        head_weight = state.get("cabeza_especie.weight")
        head_bias = state.get("cabeza_especie.bias")
        if (
            head_weight is None
            or head_bias is None
            or head_weight.ndim != 2
            or head_weight.shape != self.head.weight.shape
            or head_bias.shape != self.head.bias.shape
        ):
            raise ValueError(
                "El checkpoint DINOv2 debe contener cabeza_especie.weight y "
                "cabeza_especie.bias compatibles con exactamente 77 clases."
            )
        required = {
            "tronco.0.weight": self.trunk[0].weight,
            "tronco.0.bias": self.trunk[0].bias,
            "tronco.3.weight": self.trunk[3].weight,
            "tronco.3.bias": self.trunk[3].bias,
            "cabeza_especie.weight": self.head.weight,
            "cabeza_especie.bias": self.head.bias,
        }
        missing = [key for key in required if key not in state]
        incompatible = [
            key for key, value in required.items()
            if key in state and state[key].shape != value.shape
        ]
        if missing or incompatible:
            raise ValueError(
                f"Faltan pesos esenciales DINOv2: {missing}; dimensiones incompatibles: {incompatible}."
            )
        self.trunk.load_state_dict({
            key.removeprefix("tronco."): state[key] for key in required if key.startswith("tronco.")
        }, strict=True)
        self.head.load_state_dict({
            key.removeprefix("cabeza_especie."): state[key]
            for key in required if key.startswith("cabeza_especie.")
        }, strict=True)
        self.backbone.to(device).eval()
        self.trunk.to(device).eval()
        self.head.to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=local_only)

    @torch.inference_mode()
    def predict_batch(self, images: list[Image.Image], threshold: float) -> list[dict[str, Any]]:
        if not images:
            return []
        inputs = self.processor(
            images=[preparar_para_dinov2(image) for image in images],
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.autocast(
            self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            embedding = self.backbone(pixel_values=inputs["pixel_values"]).pooler_output
            logits = self.head(
                self.trunk(
                    embedding
                    / embedding.norm(dim=1, keepdim=True).clamp_min(1e-12)
                )
            )
        return top_predictions(logits, self.classes, threshold)


def results_dir(model_name: str, run_name: str | None = None, root: Path = RESULTS_ROOT) -> Path:
    """Carpeta de una ejecución nueva; si el nombre ya existe se añade _2, _3… para no sobrescribir."""
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / model_name / run_name
    suffix = 2
    while path.exists():
        path = root / model_name / f"{run_name}_{suffix}"
        suffix += 1
    return path


def model_name(args: argparse.Namespace) -> str:
    """Nombre del modelo para la carpeta de resultados: carpeta de los pesos DINOv2 y/o resnet50."""
    names = []
    if args.classifier in ("dinov2", "both"):
        names.append(args.dino_weights.expanduser().resolve().parent.name)
    if args.classifier in ("resnet", "both"):
        names.append("resnet50")
    return "+".join(names)


def discover_images(input_path: Path, excluded_paths: Iterable[Path] = ()) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTENSIONS:
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"No existe la entrada: {input_path}")
    excluded = {path.resolve() for path in excluded_paths}
    images = sorted(
        path for path in input_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.resolve().parent not in excluded
        and not any(
            part.casefold() in GENERATED_DIRS
            or part.casefold().endswith("_resultados")
            for part in path.relative_to(input_path).parts[:-1]
        )
    )
    if not images:
        raise ValueError(f"No se encontraron imágenes en: {input_path}")
    return images


def normalize_bbox(
    box: Iterable[float], width: int, height: int
) -> tuple[int, int, int, int] | None:
    """Devuelve una caja entera válida o None sin crear ROI artificiales."""
    values = list(box)
    if len(values) != 4 or not all(math.isfinite(float(value)) for value in values):
        return None
    raw_x1, raw_y1, raw_x2, raw_y2 = (float(value) for value in values)
    if raw_x2 <= raw_x1 or raw_y2 <= raw_y1 or width <= 0 or height <= 0:
        return None
    x1 = max(0, min(width, math.floor(raw_x1)))
    y1 = max(0, min(height, math.floor(raw_y1)))
    x2 = max(0, min(width, math.ceil(raw_x2)))
    y2 = max(0, min(height, math.ceil(raw_y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def load_rgb_image(image_path: Path) -> Image.Image | None:
    try:
        with _PIL_IMAGE_OPEN(image_path) as image:
            return image.convert("RGB")
    except (ModuleNotFoundError, OSError, ValueError) as error:
        print(f"Se omite la imagen ilegible {image_path}: {error}", file=sys.stderr)
        return None


def _prediction_columns(row: dict[str, Any], prefix: str, prediction: dict[str, Any]) -> None:
    row[f"{prefix}_especie_predicha"] = prediction["especie_predicha"]
    row[f"{prefix}_especie_mas_parecida"] = prediction["especie_mas_parecida"]
    row[f"{prefix}_confianza_%"] = prediction["confianza"]
    row[f"{prefix}_revisar"] = "Revision" if prediction["revisar"] else ""
    row[f"{prefix}_motivo_revision"] = prediction["motivo_revision"]
    for position in range(3):
        row[f"{prefix}_top{position + 1}"] = (
            prediction["top"][position][0] if position < len(prediction["top"]) else ""
        )
        row[f"{prefix}_top{position + 1}_confianza_%"] = (
            prediction["top"][position][1] if position < len(prediction["top"]) else ""
        )


def _output_stem(relative: Path) -> str:
    """Convierte una ruta relativa en un nombre estable y seguro para ficheros."""
    parts = [
        re.sub(r"[^0-9A-Za-zÀ-ÿ.-]+", "_", part).strip(" ._") or "imagen"
        for part in relative.with_suffix("").parts
    ]
    return "__".join(parts) or "imagen"


ANNOTATION_COLOR = (0, 200, 0)
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Ubuntu
    "DejaVuSans-Bold.ttf",
    "arialbd.ttf",  # Windows
    "arial.ttf",
)


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.ImageFont:
    """Fuente legible y escalable; nunca cae en la fuente bitmap diminuta de PIL."""
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except (OSError, TypeError):
            continue
    return ImageFont.load_default(size=size)  # Pillow >= 10.1: escalable


def _annotation_label(prediction: dict[str, Any]) -> str:
    """Solo el nombre de la especie (sin confianza ni marca de revisión)."""
    return (prediction["especie_mas_parecida"] or "sin clase").replace("_", " ")


def _annotate_image(
    original: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    predictions: list[dict[str, Any]],
    model_name: str,
) -> Image.Image:
    """Dibuja recuadros verdes con el nombre de la especie sobre una copia de la imagen.

    La confianza y la marca de revisión solo van al Excel. El tamaño de letra y el
    grosor del recuadro son proporcionales a la imagen para que se lean en cualquier resolución.
    """
    annotated = original.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    short_side = min(width, height)
    font = _font(max(24, short_side // 30))
    line_width = max(4, short_side // 200)
    padding = max(4, line_width)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle((x1, y1, x2, y2), outline=ANNOTATION_COLOR, width=line_width)
    placed: list[tuple[int, int, int, int]] = []
    for (x1, y1, x2, y2), prediction in zip(boxes, predictions):
        label = _annotation_label(prediction)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_w, label_h = right - left + 2 * padding, bottom - top + 2 * padding
        label_x = max(0, min(x1, width - label_w))  # que no se salga por la derecha
        # Encima del recuadro, debajo o dentro: la primera posición que quepa y no pise otra etiqueta
        candidates = [y1 - label_h, y2, y1]
        fits = [y for y in candidates if 0 <= y and y + label_h <= height]
        label_y = next(
            (y for y in fits if not any(
                label_x < px2 and px1 < label_x + label_w and y < py2 and py1 < y + label_h
                for px1, py1, px2, py2 in placed
            )),
            fits[0] if fits else max(0, y1),
        )
        rect = (label_x, label_y, label_x + label_w, label_y + label_h)
        placed.append(rect)
        draw.rectangle(rect, fill=ANNOTATION_COLOR)
        draw.text((label_x + padding - left, label_y + padding - top), label, fill=(0, 0, 0), font=font)
    return annotated


def _save_excel(rows: list[dict[str, Any]], path: Path, model: str | None = None) -> None:
    """Escribe resultados y ofrece un error accionable si falta el motor XLSX."""
    path.parent.mkdir(parents=True, exist_ok=True)
    output_rows = rows
    if model is not None:
        output_rows = [
            {
                key: value
                for key, value in row.items()
                if not key.startswith(("dinov2_", "resnet_")) or key.startswith(f"{model}_")
            }
            for row in rows
        ]
    try:
        pd.DataFrame(output_rows).to_excel(path, index=False, engine="xlsxwriter")
    except ImportError as error:
        if "xlsxwriter" in str(error).lower():
            raise RuntimeError(
                "No se puede escribir Excel: falta la dependencia 'xlsxwriter'. "
                "Instálala con 'python -m pip install xlsxwriter'."
            ) from error
        raise


def iter_images(
    paths: list[Path],
    workers: int = IMAGE_LOAD_WORKERS,
    ahead: int = IMAGE_PREFETCH,
) -> Iterable[tuple[Path, Image.Image | None]]:
    """Decodifica unas pocas imágenes por adelantado mientras trabaja la GPU."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator = iter(paths)
        queue: deque[tuple[Path, Future[Image.Image | None]]] = deque(
            (path, pool.submit(load_rgb_image, path))
            for path in itertools.islice(iterator, ahead)
        )
        while queue:
            path, future = queue.popleft()
            next_path = next(iterator, None)
            if next_path is not None:
                queue.append((next_path, pool.submit(load_rgb_image, next_path)))
            yield path, future.result()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--classifier",
        "--classifier-architecture",
        dest="classifier",
        choices=("dinov2", "resnet", "both"),
        default="both",
    )
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--dino-weights", type=Path, default=DEFAULT_DINO_WEIGHTS)
    parser.add_argument("--resnet-weights", type=Path, default=DEFAULT_RESNET_WEIGHTS)
    parser.add_argument("--dino-classes", type=Path, default=DEFAULT_DINO_CLASSES)
    parser.add_argument("--resnet-classes", type=Path, default=DEFAULT_RESNET_CLASSES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-name", help="Nombre de la prueba (por defecto, fecha y hora).")
    parser.add_argument("--output-dir", type=Path,
                        help="Solo si hace falta otra carpeta; por defecto Inferir/resultados_inferencia/<modelo>/<prueba>.")
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument(
        "--reinhard-reference",
        type=Path,
        help="Referencia requerida para Reinhard; no se aplica por defecto y aún no se implementa.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.conf <= 1 or not 0 <= args.threshold <= 1:
        raise SystemExit("--conf y --threshold deben estar entre 0 y 1.")
    if args.reinhard_reference is not None:
        raise NotImplementedError(
            "Reinhard requiere estadísticas de una imagen de referencia; "
            "no se aplica automáticamente."
        )

    input_path = args.input.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir is not None
                  else results_dir(model_name(args), args.run_name))
    if input_path.is_dir():
        try:
            output_dir.relative_to(input_path)
        except ValueError:
            pass
        else:
            raise ValueError(
                f"--output-dir no puede estar dentro de la entrada {input_path}: {output_dir}. "
                "Usa una carpeta de resultados hermana para no reingerir salidas."
            )
    images = discover_images(input_path, (output_dir,))
    yolo_path = args.yolo_weights.expanduser()
    if not yolo_path.is_file():
        raise FileNotFoundError(f"No se encontraron los pesos YOLO: {yolo_path}")
    from ultralytics import YOLO
    detector = YOLO(str(yolo_path))
    device = torch.device(args.device)

    classifiers: dict[str, Any] = {}
    if args.classifier in ("dinov2", "both"):
        dino_classes = resolve_dino_classes(args.dino_weights.expanduser(), args.dino_classes.expanduser())
        classifiers["dinov2"] = DinoClassifier(args.dino_weights.expanduser(), dino_classes, device)
    if args.classifier in ("resnet", "both"):
        resnet_classes = read_classes(args.resnet_classes.expanduser(), 78, "clases ResNet")
        classifiers["resnet"] = ResNetClassifier(args.resnet_weights.expanduser(), resnet_classes, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "crops"
    bbox_dir = output_dir / "bbox"
    crops_dir.mkdir(exist_ok=True)
    bbox_dir.mkdir(exist_ok=True)
    bbox_model_dirs = {name: bbox_dir / name for name in classifiers}
    for model_dir in bbox_model_dirs.values():
        model_dir.mkdir(exist_ok=True)
    output = args.output.expanduser() if args.output else output_dir / "classification_results.xlsx"
    model_outputs = {
        name: output.parent / f"{output.stem}_{name}{output.suffix}"
        for name in classifiers
    }
    rows: list[dict[str, Any]] = []
    roi_number = 0
    original_error: BaseException | None = None
    io_pool = ThreadPoolExecutor(max_workers=IMAGE_SAVE_WORKERS)
    pending_saves: deque[Future[Any]] = deque()

    def save_async(image: Image.Image, path: Path, **kwargs: Any) -> None:
        pending_saves.append(io_pool.submit(image.save, path, **kwargs))
        while len(pending_saves) > MAX_PENDING_SAVES:
            pending_saves.popleft().result()

    def drain_saves() -> None:
        io_pool.shutdown(wait=True)
        while pending_saves:
            pending_saves.popleft().result()

    try:
        progress = tqdm(
            iter_images(images),
            total=len(images),
            desc="Procesando imágenes",
            unit="imagen",
            dynamic_ncols=True,
        )
        used_stems: set[str] = set()
        for image_path, original in progress:
            relative = image_path.relative_to(input_path) if input_path.is_dir() else Path(image_path.name)
            stem = _output_stem(relative)
            if stem in used_stems:
                suffix = 2
                while f"{stem}_{suffix}" in used_stems:
                    suffix += 1
                stem = f"{stem}_{suffix}"
            used_stems.add(stem)
            progress.set_postfix_str(f"YOLO: {relative}", refresh=False)
            if original is None:
                continue
            result = detector.predict(
                source=original, conf=args.conf, imgsz=args.imgsz,
                save=False, save_txt=False, verbose=False,
                device=args.device, half=device.type == "cuda",
            )[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            crops: list[Image.Image] = []
            detections: list[tuple[int, float, tuple[int, int, int, int]]] = []
            width, height = original.size
            for index, (box, confidence) in enumerate(
                zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist()), start=1
            ):
                normalized = normalize_bbox(box, width, height)
                if normalized is None:
                    continue
                detections.append((index, float(confidence), normalized))
                crops.append(original.crop(normalized))
            if not crops:
                continue
            predictions: dict[str, list[dict[str, Any]]] = {}
            for name, classifier in classifiers.items():
                progress.set_postfix_str(
                    f"{name} ({len(crops)} ROI): {relative}",
                    refresh=False,
                )
                predictions[name] = classifier.predict_batch(crops, args.threshold)
                save_async(
                    _annotate_image(
                        original,
                        [box for _, _, box in detections],
                        predictions[name],
                        name,
                    ),
                    bbox_model_dirs[name] / f"{stem}.jpg",
                    quality=85,
                )
            for crop_index, (detection, yolo_confidence, box) in enumerate(detections):
                roi_number += 1
                x1, y1, x2, y2 = box
                roi_path = crops_dir / f"{stem}_roi_{roi_number}.png"
                save_async(crops[crop_index], roi_path, compress_level=1)
                row: dict[str, Any] = {
                    "image": str(relative),
                    "item": detection,
                    "yolo_confidence_%": round(yolo_confidence * 100, 2),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "crop_file": roi_path.name,
                }
                for name, batch in predictions.items():
                    _prediction_columns(row, name, batch[crop_index])
                rows.append(row)
    except BaseException as error:
        original_error = error
    finally:
        save_errors: list[BaseException] = []
        try:
            drain_saves()
        except BaseException as error:
            save_errors.append(error)
            print(f"No se pudieron guardar todas las imágenes: {error}", file=sys.stderr)
        for path, model in [(output, None), *[(path, name) for name, path in model_outputs.items()]]:
            try:
                _save_excel(rows, path, model)
            except BaseException as error:
                save_errors.append(error)
                print(f"No se pudo guardar {path}: {error}", file=sys.stderr)
        if original_error is not None:
            if save_errors:
                print(
                    f"Se conservaron {len(rows)} ROI, pero falló algún Excel parcial.",
                    file=sys.stderr,
                )
            raise original_error.with_traceback(original_error.__traceback__)
        if save_errors:
            raise RuntimeError(
                "La inferencia terminó, pero no se pudieron guardar todos los Excel."
            ) from save_errors[0]
    print(f"Inferencia completada: {len(rows)} ROI; resultados: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
