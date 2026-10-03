"""Inferencia YOLO recursiva con DINOv2, ResNet50 o ambos clasificadores."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms

from constantes import VARIABLES_GLOBALES
from embeddings import get_embedding, inicializar_dinov2
from modelo import cargar_modelo_entrenado

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "imagenes_inferencia"
DEFAULT_YOLO = ROOT / "yolo_dinov2" / "yolo_best.pt"
DEFAULT_DINO_WEIGHTS = ROOT / "modelos" / "75_objetivo" / "modelo_75_objetivo.pth"
DEFAULT_RESNET = ROOT / "yolo_dinov2" / "resnet50_checkpoint_epoch50.pth"
DEFAULT_CLASSES = {
    "dinov2": ROOT / "txt_classes" / "classes_77(dino).txt",
    "resnet": ROOT / "txt_classes" / "classes_78(resnet).txt",
}


def cargar_clases(ruta: Path, esperadas: int) -> list[str]:
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de clases: {ruta}")
    clases = [linea.strip() for linea in ruta.read_text(encoding="utf-8").splitlines()
              if linea.strip() and not linea.lstrip().startswith("#")]
    if len(clases) != esperadas or len(set(clases)) != len(clases):
        raise ValueError(f"{ruta} debe contener exactamente {esperadas} clases únicas; contiene {len(clases)}.")
    return clases


def cargar_yolo(ruta: Path) -> Any:
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontraron los pesos YOLO: {ruta}")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("YOLO es obligatorio: instala la dependencia ultralytics.") from error
    return YOLO(str(ruta))


def _top3(logits: torch.Tensor, clases: list[str], umbral: float) -> dict[str, Any]:
    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != len(clases):
        raise ValueError(f"Salida incompatible: {tuple(logits.shape)} para {len(clases)} clases.")
    valores, indices = torch.topk(torch.softmax(logits, dim=1), k=min(3, len(clases)), dim=1)
    top = [(clases[indice], round(float(valor) * 100, 2))
           for indice, valor in zip(indices[0].tolist(), valores[0].tolist())]
    revisar = top[0][1] < umbral * 100
    return {
        "especie_predicha": top[0][0],
        "especie_mas_parecida": top[0][0],
        "confianza": top[0][1],
        "top": top,
        "revisar": revisar,
        "motivo_revision": "confianza_baja" if revisar else "",
    }


class ClasificadorDino:
    def __init__(self, pesos: Path, clases_txt: Path, device: torch.device) -> None:
        clases = cargar_clases(clases_txt, 77)
        VARIABLES_GLOBALES["DEVICE"] = device
        self.modelo, metadatos = cargar_modelo_entrenado(ruta_pesos=pesos, clases=clases)
        if metadatos != clases:
            raise ValueError("classes_77(dino).txt no coincide exactamente con el orden de metadatos_modelo.json.")
        self.clases = metadatos
        self.device = device
        self.processor, self.backbone, _, self.augmentation = inicializar_dinov2(device)

    def predecir(self, imagen: Image.Image) -> dict[str, Any]:
        inputs = self.processor(images=imagen.convert("RGB"), return_tensors="pt")
        inputs = {clave: valor.to(self.device) for clave, valor in inputs.items()}
        with torch.inference_mode():
            embedding = self.backbone(pixel_values=inputs["pixel_values"]).pooler_output.float()
            logits, _, tronco = self.modelo(embedding)
            indices = torch.argmax(logits, dim=1)
            desconocida = bool(self.modelo.es_desconocida(tronco, indices).item())
        resultado = _top3(logits, self.clases, float(VARIABLES_GLOBALES["UMBRAL_CONF"]))
        if desconocida:
            resultado["especie_predicha"] = "Desconocida"
            resultado["revisar"] = True
            resultado["motivo_revision"] = "especie_desconocida"
        return resultado


class ClasificadorResnet:
    def __init__(self, pesos: Path, clases_txt: Path, device: torch.device) -> None:
        clases = cargar_clases(clases_txt, 78)
        if not pesos.is_file():
            raise FileNotFoundError(f"No se encontraron los pesos ResNet: {pesos}")
        self.modelo = models.resnet50(weights=None)
        self.modelo.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(self.modelo.fc.in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(256, 78),
        )
        estado = torch.load(pesos, map_location=device, weights_only=True)
        if isinstance(estado, dict) and "state_dict" in estado:
            estado = estado["state_dict"]
        if not isinstance(estado, dict):
            raise ValueError("Los pesos ResNet no contienen un state_dict válido.")
        estado = {clave.removeprefix("module."): valor for clave, valor in estado.items()}
        try:
            self.modelo.load_state_dict(estado, strict=True)
        except RuntimeError as error:
            raise ValueError("Los pesos ResNet no coinciden con la arquitectura de 78 clases.") from error
        self.modelo.to(device).eval()
        self.clases = clases
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def predecir(self, imagen: Image.Image) -> dict[str, Any]:
        with torch.inference_mode():
            logits = self.modelo(self.transform(imagen.convert("RGB")).unsqueeze(0).to(self.device))
        return _top3(logits, self.clases, float(VARIABLES_GLOBALES["UMBRAL_CONF"]))


def _imagenes(ruta: Path) -> list[Path]:
    if ruta.is_file():
        return [ruta]
    if not ruta.is_dir():
        raise FileNotFoundError(f"No existe la entrada: {ruta}")
    imagenes = sorted(archivo for archivo in ruta.rglob("*")
                      if archivo.is_file() and archivo.suffix.lower() in VARIABLES_GLOBALES["EXTENSIONES_VALIDAS"])
    if not imagenes:
        raise ValueError(f"No se encontraron imágenes en {ruta}")
    return imagenes


def _añadir_prediccion(fila: dict[str, Any], prefijo: str, resultado: dict[str, Any]) -> None:
    fila[f"{prefijo}_especie_predicha"] = resultado["especie_predicha"]
    fila[f"{prefijo}_especie_mas_parecida"] = resultado["especie_mas_parecida"]
    fila[f"{prefijo}_confianza_%"] = resultado["confianza"]
    fila[f"{prefijo}_revisar"] = "Revision" if resultado["revisar"] else ""
    fila[f"{prefijo}_motivo_revision"] = resultado["motivo_revision"]
    for numero in range(3):
        fila[f"{prefijo}_top{numero + 1}"] = (
            resultado["top"][numero][0] if numero < len(resultado["top"]) else ""
        )
        fila[f"{prefijo}_top{numero + 1}_confianza_%"] = (
            resultado["top"][numero][1] if numero < len(resultado["top"]) else ""
        )


def inferir(entrada: Path, salida: Path, detector: Any, clasificadores: dict[str, Any],
            confianza_yolo: float, device_yolo: str | int) -> Path:
    filas: list[dict[str, Any]] = []
    imagenes = _imagenes(entrada)
    raiz = entrada if entrada.is_dir() else entrada.parent
    for imagen in imagenes:
        detecciones = detector.predict(source=str(imagen), conf=confianza_yolo,
                                       device=device_yolo, verbose=False)[0]
        ruta_relativa = str(imagen.relative_to(raiz)) if entrada.is_dir() else imagen.name
        if detecciones.boxes is None or len(detecciones.boxes) == 0:
            fila = {"imagen": ruta_relativa, "deteccion": "", "confianza_yolo_%": "",
                    "revisar": "Revision", "motivo_revision": "sin_detecciones"}
            for prefijo in clasificadores:
                _añadir_prediccion(fila, prefijo, {"especie_predicha": "", "especie_mas_parecida": "",
                    "confianza": "", "top": [], "revisar": True, "motivo_revision": "sin_detecciones"})
            filas.append(fila)
            continue
        with Image.open(imagen) as original:
            original = original.convert("RGB")
            for numero, (caja, confianza) in enumerate(zip(
                    detecciones.boxes.xyxy.cpu().tolist(), detecciones.boxes.conf.cpu().tolist()), 1):
                x1, y1, x2, y2 = [max(0, round(valor)) for valor in caja]
                recorte = original.crop((x1, y1, x2, y2))
                fila = {"imagen": ruta_relativa, "deteccion": numero,
                        "confianza_yolo_%": round(float(confianza) * 100, 2),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2}
                for prefijo, clasificador in clasificadores.items():
                    _añadir_prediccion(fila, prefijo, clasificador.predecir(recorte))
                filas.append(fila)
    salida.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_excel(salida, index=False)
    return salida


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferencia recursiva YOLO + DINOv2/ResNet50.")
    parser.add_argument("entrada", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--classifier", choices=("dinov2", "resnet", "both"), default="both")
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--dino-weights", type=Path, default=DEFAULT_DINO_WEIGHTS)
    parser.add_argument("--resnet-weights", type=Path, default=DEFAULT_RESNET)
    parser.add_argument("--dino-classes", type=Path, default=DEFAULT_CLASSES["dinov2"])
    parser.add_argument("--resnet-classes", type=Path, default=DEFAULT_CLASSES["resnet"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=str(VARIABLES_GLOBALES["DEVICE"]))
    return parser.parse_args()


def main() -> None:
    args = argumentos()
    if not 0 <= args.conf <= 1:
        raise SystemExit("--conf debe estar entre 0 y 1.")
    entrada = args.entrada.expanduser()
    if not entrada.exists():
        raise FileNotFoundError(f"No existe la entrada: {entrada}")
    device = torch.device(args.device)
    detector = cargar_yolo(args.yolo_weights.expanduser())
    clasificadores: dict[str, Any] = {}
    if args.classifier in ("dinov2", "both"):
        clasificadores["dinov2"] = ClasificadorDino(args.dino_weights.expanduser(),
                                                    args.dino_classes.expanduser(), device)
    if args.classifier in ("resnet", "both"):
        clasificadores["resnet"] = ClasificadorResnet(args.resnet_weights.expanduser(),
                                                       args.resnet_classes.expanduser(), device)
    salida = args.output or (entrada / "predicciones.xlsx" if entrada.is_dir()
                             else entrada.parent / "predicciones.xlsx")
    ruta = inferir(entrada, salida, detector, clasificadores, args.conf,
                   int(args.device) if args.device.isdigit() else args.device)
    print(f"Excel guardado en: {ruta}")


if __name__ == "__main__":
    main()
