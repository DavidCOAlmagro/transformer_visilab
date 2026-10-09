"""
--------------------------------------
Spec 006 — Fine-tuning parcial de DINOv2 (últimos bloques + MLP).

A diferencia de main.py (DINOv2 congelado + embeddings precalculados), aquí
las imágenes pasan por DINOv2 en cada época: así se pueden entrenar sus
últimos bloques y aplicar aumentación online (una variante nueva por época).

- Mismas 77 clases y splits que 75_objetivo (lista relativa en la spec).
- MLP desde cero (Xavier) o, con --pesos-mlp, desde un modelo anterior.
- Reanudable: guarda <salida>/ultimo.pth al final de cada época.
- Freno térmico: pausa si la GPU llega a 85 ºC hasta que baje 10 ºC.
  `--temperatura-pausa 0` lo desactiva (la GPU mantiene su propia protección).

Uso (Ubuntu):
    python3 tareas.py entrenar          (o: make entrenar)
Prueba de humo (Windows):
    python src/entrenar.py \
        --max-imagenes 256 --epocas 1 --batch 8 --salida <carpeta_temporal>
--------------------------------------
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageFile
from sklearn.metrics import classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from calibracion_confianza import ajustar_temperatura  # noqa: E402
from clasificador import ClasificadorDiatomeas  # noqa: E402
from constantes import VARIABLES_GLOBALES  # noqa: E402
from embeddings import crear_augmentation, resolver_modelo_dinov2  # noqa: E402
from preparar_datos import construir_numero_genero, fijar_semilla, normalizar_nombre_especie  # noqa: E402
from preprocesado import CONFIGURACION_PREPROCESADO, preparar_para_dinov2  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True
PRUEBA = "75_objetivo_ft"


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning parcial de DINOv2 (spec 006).")
    parser.add_argument("--raiz-imagenes", type=Path, default=RAIZ / "data" / "imagenes_visilab(raw)")
    parser.add_argument("--splits", type=Path, default=RAIZ / "recursos" / "splits_75_objetivo_relativos.txt.gz")
    parser.add_argument("--metadatos", type=Path,
                        help="metadatos_modelo.json con las especies; por defecto, ESPECIES_FILTRADAS de constantes.py.")
    parser.add_argument("--pesos-mlp", type=Path,
                        help="Pesos iniciales del MLP (opcional); sin ellos el MLP empieza desde cero (Xavier).")
    parser.add_argument("--salida", type=Path, default=RAIZ / "modelos" / PRUEBA)
    parser.add_argument("--bloques", type=int, default=4, help="Últimos bloques de DINOv2 que se entrenan.")
    parser.add_argument("--epocas", type=int, default=15)
    parser.add_argument("--paciencia", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-cabeza", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0 if os.name == "nt" else 8)
    parser.add_argument("--temperatura-pausa", type=int, default=85,
                        help="ºC a los que se pausa el entrenamiento; 0 desactiva el freno térmico.")
    parser.add_argument("--max-imagenes", type=int, default=0, help="Limita cada split (prueba de humo).")
    return parser.parse_args()


# --------------------------------------------------------------------- datos
def leer_splits(ruta: Path, raiz: Path, max_imagenes: int) -> dict[str, list[tuple[Path, str]]]:
    """Lee la lista relativa y la resuelve contra la raíz local de imágenes."""
    splits: dict[str, list[tuple[Path, str]]] = {"train": [], "val": [], "test": []}
    with gzip.open(ruta, "rt", encoding="utf-8") as archivo:
        for linea in archivo:
            split, relativa = linea.rstrip("\n").split("\t")
            ruta_imagen = raiz.joinpath(*relativa.split("/"))
            splits[split].append((ruta_imagen, normalizar_nombre_especie(ruta_imagen.parent.name)))
    if max_imagenes:
        # Submuestra repartida por todo el split para que haya varias especies
        splits = {s: items[::max(1, len(items) // max_imagenes)][:max_imagenes] for s, items in splits.items()}
    return splits


def comprobar_imagenes(splits: dict[str, list[tuple[Path, str]]]) -> None:
    """Aborta si falta más del 1 % de las imágenes (dataset distinto en esta máquina)."""
    faltan = [str(r) for items in splits.values() for r, _ in items if not r.is_file()]
    total = sum(len(items) for items in splits.values())
    print(f"Imágenes encontradas: {total - len(faltan)}/{total}", flush=True)
    if len(faltan) > 0.01 * total:
        print("Ejemplos que faltan:\n  " + "\n  ".join(faltan[:10]), flush=True)
        raise SystemExit("Falta más del 1 % de las imágenes: revisa --raiz-imagenes.")
    for items in splits.values():
        items[:] = [(r, e) for r, e in items if r.is_file()]


class DatasetImagenes(Dataset):
    """Imagen -> pad-square -> [aumentación online -> pad] -> AutoImageProcessor."""

    def __init__(self, items: list[tuple[Path, str]], numero_especie: dict[str, int],
                 processor, entrenamiento: bool) -> None:
        self.items = items
        self.numero_especie = numero_especie
        self.processor = processor
        self.entrenamiento = entrenamiento
        self.aumentacion = transforms.Compose([
            crear_augmentation(),
            # Variación de encuadre y escala como en el entrenamiento de ResNet50
            transforms.RandomAffine(degrees=0, shear=10, scale=(0.8, 1.2), fill=(128, 128, 128)),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, int]:
        ruta, especie = self.items[indice]
        imagen = preparar_para_dinov2(Image.open(ruta).convert("RGB"))
        if self.entrenamiento:
            imagen = preparar_para_dinov2(self.aumentacion(imagen))
        pixeles = self.processor(images=imagen, return_tensors="pt")["pixel_values"][0]
        return pixeles, self.numero_especie[especie]


# -------------------------------------------------------------------- modelo
class DinoAjustado(nn.Module):
    """DINOv2 (últimos bloques entrenables) -> CLS -> L2 -> ClasificadorDiatomeas."""

    def __init__(self, backbone: nn.Module, mlp: ClasificadorDiatomeas, bloques: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.mlp = mlp
        self.backbone.requires_grad_(False)
        for bloque in self.backbone.encoder.layer[-bloques:]:
            bloque.requires_grad_(True)
        self.backbone.layernorm.requires_grad_(True)

    def forward(self, pixeles: torch.Tensor) -> torch.Tensor:
        embedding = self.backbone(pixel_values=pixeles).pooler_output.float()
        embedding = embedding / embedding.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return self.mlp(embedding)[0]

    def state_dict_inferencia(self) -> dict[str, torch.Tensor]:
        """Formato que entiende Inferir/: claves backbone.* + claves del MLP."""
        estado = {f"backbone.{k}": v for k, v in self.backbone.state_dict().items()}
        estado.update(self.mlp.state_dict())
        return estado


def temperatura_gpu() -> int:
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError):
        return 0


def freno_termico(limite: int) -> None:
    """Pausa al llegar a ``limite`` ºC hasta bajar 10 ºC; ``limite`` 0 lo desactiva."""
    if limite and temperatura_gpu() >= limite:
        print(f"GPU {temperatura_gpu()} C: pausa térmica", flush=True)
        while temperatura_gpu() > limite - 10:
            time.sleep(10)
        print(f"GPU {temperatura_gpu()} C: reanudo", flush=True)


@torch.no_grad()
def predecir(modelo: DinoAjustado, cargador: DataLoader, device: torch.device,
             tipo_amp: torch.dtype | None) -> tuple[torch.Tensor, torch.Tensor]:
    modelo.eval()
    logits, etiquetas = [], []
    for pixeles, y in cargador:
        with torch.autocast(device.type, dtype=tipo_amp or torch.float16, enabled=tipo_amp is not None):
            logits.append(modelo(pixeles.to(device, non_blocking=True)).float().cpu())
        etiquetas.append(y)
    return torch.cat(logits), torch.cat(etiquetas)


def metricas(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(1)
    top3 = (logits.topk(min(3, logits.shape[1]), dim=1).indices == y[:, None]).any(1)
    return {
        "loss": float(nn.functional.cross_entropy(logits, y)),
        "accuracy": float((pred == y).float().mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "top3": float(top3.float().mean()),
    }


def guardar_confusiones(pred: torch.Tensor, y: torch.Tensor, especies: list[str], ruta: Path,
                        minimo: int = 5) -> None:
    """Pares (real -> predicha) que se repiten al menos ``minimo`` veces en test, de más a menos."""
    pares: dict[tuple[int, int], int] = {}
    for real, predicha in zip(y.tolist(), pred.tolist()):
        if real != predicha:
            pares[(real, predicha)] = pares.get((real, predicha), 0) + 1
    lineas = [f"{especies[r]:35s} -> {especies[p]:35s} : {n} veces"
              for (r, p), n in sorted(pares.items(), key=lambda kv: -kv[1]) if n >= minimo]
    texto = f"=== Confusiones en test con {minimo} o más casos ===\n" + "\n".join(lineas) + "\n"
    ruta.write_text(texto, encoding="utf-8")


def graficar(historial: list[dict], ruta: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epocas = [h["epoca"] for h in historial]
    figura, ejes = plt.subplots(1, 2, figsize=(12, 4))
    ejes[0].plot(epocas, [h["loss_train"] for h in historial], label="train")
    ejes[0].plot(epocas, [h["val"]["loss"] for h in historial], label="val")
    ejes[0].set_title("Pérdida"); ejes[0].legend()
    ejes[1].plot(epocas, [h["val"]["accuracy"] for h in historial], label="accuracy val")
    ejes[1].plot(epocas, [h["val"]["macro_f1"] for h in historial], label="macro-F1 val")
    ejes[1].set_title("Validación"); ejes[1].legend()
    figura.tight_layout(); figura.savefig(ruta, dpi=150); plt.close(figura)


# ---------------------------------------------------------------------- main
def main() -> None:
    args = argumentos()
    fijar_semilla(42)
    args.salida.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Precisión según la GPU: bf16 nativo desde Ampere (capacidad >= 8), fp16 con tensor cores
    # en Volta/Turing (7.x) y fp32 en Pascal o anteriores (p. ej. Quadro P4000, 6.1), donde
    # bf16 se emula y fp16 no acelera. None = fp32 sin autocast.
    capacidad = torch.cuda.get_device_capability()[0] if device.type == "cuda" else 0
    tipo_amp = torch.bfloat16 if capacidad >= 8 else torch.float16 if capacidad == 7 else None
    gpu = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    print(f"{datetime.now():%H:%M} device={device} ({gpu}) amp={tipo_amp or "fp32"} bloques={args.bloques}", flush=True)

    especies = sorted(json.loads(args.metadatos.read_text(encoding="utf-8"))["especies_filtradas"]
                      if args.metadatos else VARIABLES_GLOBALES["ESPECIES_FILTRADAS"])
    numero_especie = {e: i for i, e in enumerate(especies)}
    splits = leer_splits(args.splits, args.raiz_imagenes, args.max_imagenes)
    comprobar_imagenes(splits)

    from transformers import AutoImageProcessor, AutoModel
    ruta_modelo, solo_local = resolver_modelo_dinov2()
    processor = AutoImageProcessor.from_pretrained(ruta_modelo, local_files_only=solo_local)
    backbone = AutoModel.from_pretrained(ruta_modelo, local_files_only=solo_local)

    mlp = ClasificadorDiatomeas(len(especies), len(construir_numero_genero(set(especies))))
    if args.pesos_mlp:
        mlp.load_state_dict(torch.load(args.pesos_mlp, map_location="cpu", weights_only=True))
    modelo = DinoAjustado(backbone, mlp, args.bloques).to(device)
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {entrenables / 1e6:.1f} M", flush=True)

    # Desbalance: mismo suavizado 1/n^EXP que main.py (sampler y pesos de la pérdida)
    y_train = torch.tensor([numero_especie[e] for _, e in splits["train"]])
    conteo = torch.bincount(y_train, minlength=len(especies)).float().clamp_min(1)
    peso_clase = 1.0 / conteo.pow(VARIABLES_GLOBALES["EXPONENTE_PESO_CLASE"])
    sampler = WeightedRandomSampler(peso_clase[y_train], num_samples=len(y_train), replacement=True)

    opciones = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=device.type == "cuda",
                    persistent_workers=args.workers > 0)
    cargadores = {
        "train": DataLoader(DatasetImagenes(splits["train"], numero_especie, processor, True), sampler=sampler, **opciones),
        "val": DataLoader(DatasetImagenes(splits["val"], numero_especie, processor, False), **opciones),
        "test": DataLoader(DatasetImagenes(splits["test"], numero_especie, processor, False), **opciones),
    }

    perdida = nn.CrossEntropyLoss(label_smoothing=VARIABLES_GLOBALES["LABEL_SMOOTHING"], weight=peso_clase.to(device))
    params_backbone = [p for p in modelo.backbone.parameters() if p.requires_grad]
    optimizador = torch.optim.AdamW([
        {"params": params_backbone, "lr": args.lr_backbone},
        {"params": modelo.mlp.parameters(), "lr": args.lr_cabeza},
    ], weight_decay=VARIABLES_GLOBALES["WEIGHT_DECAY"])
    pasos_epoca = len(cargadores["train"])
    pasos_totales, pasos_warmup = pasos_epoca * args.epocas, pasos_epoca

    def factor_lr(paso: int) -> float:
        if paso < pasos_warmup:
            return (paso + 1) / pasos_warmup
        progreso = (paso - pasos_warmup) / max(1, pasos_totales - pasos_warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progreso)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizador, factor_lr)
    escalador = torch.amp.GradScaler(enabled=tipo_amp == torch.float16)

    ruta_ultimo, ruta_mejor = args.salida / "ultimo.pth", args.salida / "mejor_entrenamiento.pth"
    estado = {"epoca": 0, "mejor_f1": -1.0, "sin_mejora": 0, "historial": []}
    if ruta_ultimo.is_file():
        punto = torch.load(ruta_ultimo, map_location=device, weights_only=False)
        modelo.load_state_dict(punto["modelo"]); optimizador.load_state_dict(punto["optimizador"])
        scheduler.load_state_dict(punto["scheduler"]); escalador.load_state_dict(punto["escalador"])
        estado = punto["estado"]
        print(f"Reanudando tras la época {estado['epoca']}", flush=True)

    while estado["epoca"] < args.epocas and estado["sin_mejora"] < args.paciencia:
        epoca = estado["epoca"] + 1
        modelo.train()
        acumulada, inicio = 0.0, time.time()
        for paso, (pixeles, y) in enumerate(cargadores["train"]):
            pixeles, y = pixeles.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizador.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=tipo_amp or torch.float16, enabled=tipo_amp is not None):
                loss = perdida(modelo(pixeles), y)
            escalador.scale(loss).backward()
            escalador.unscale_(optimizador)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            escalador.step(optimizador); escalador.update(); scheduler.step()
            acumulada += loss.item()
            if paso % 100 == 0:
                print(f"  época {epoca} paso {paso}/{pasos_epoca} loss {loss.item():.4f} "
                      f"{(time.time() - inicio) / 60:.1f} min GPU {temperatura_gpu()}C", flush=True)
            if paso % 20 == 0:
                freno_termico(args.temperatura_pausa)
        val = metricas(*predecir(modelo, cargadores["val"], device, tipo_amp))
        registro = {"epoca": epoca, "loss_train": acumulada / pasos_epoca, "val": val,
                    "minutos": round((time.time() - inicio) / 60, 1)}
        estado["historial"].append(registro)
        print(f"Época {epoca}/{args.epocas} — loss train {registro['loss_train']:.4f} — loss val {val['loss']:.4f} — "
              f"acc val {val['accuracy']:.4f} — macro-F1 val {val['macro_f1']:.4f} — {registro['minutos']} min", flush=True)
        if val["macro_f1"] > estado["mejor_f1"]:
            estado["mejor_f1"], estado["sin_mejora"] = val["macro_f1"], 0
            torch.save(modelo.state_dict(), ruta_mejor)
        else:
            estado["sin_mejora"] += 1
            print(f"No mejora ({estado['sin_mejora']}/{args.paciencia})", flush=True)
        estado["epoca"] = epoca
        torch.save({"modelo": modelo.state_dict(), "optimizador": optimizador.state_dict(),
                    "scheduler": scheduler.state_dict(), "escalador": escalador.state_dict(),
                    "estado": estado}, ruta_ultimo)
        graficar(estado["historial"], args.salida / f"curvas_entrenamiento_{PRUEBA}.png")

    # Mejor época -> calibración de temperatura en val -> evaluación en test
    modelo.load_state_dict(torch.load(ruta_mejor, map_location=device, weights_only=True))
    logits_val, y_val = predecir(modelo, cargadores["val"], device, tipo_amp)
    temperatura = ajustar_temperatura(logits_val, y_val)
    with torch.no_grad():
        modelo.mlp.cabeza_especie.weight.div_(temperatura)
        modelo.mlp.cabeza_especie.bias.div_(temperatura)
    logits_test, y_test = predecir(modelo, cargadores["test"], device, tipo_amp)
    resultado = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "configuracion": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "mejor_epoca": max(estado["historial"], key=lambda h: h["val"]["macro_f1"])["epoca"],
        "mejor_macro_f1_validacion": estado["mejor_f1"],
        "temperatura": temperatura,
        "test": metricas(logits_test, y_test),
        "historial": estado["historial"],
    }
    (args.salida / "metricas.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.salida / "reporte_test.txt").write_text(classification_report(
        y_test, logits_test.argmax(1), labels=list(range(len(especies))), target_names=especies,
        digits=3, zero_division=0), encoding="utf-8")
    guardar_confusiones(logits_test.argmax(1), y_test, especies, args.salida / "confusiones.txt")
    torch.save(modelo.state_dict_inferencia(), args.salida / f"modelo_{PRUEBA}.pth")
    metadatos = {
        "version_pipeline": "dinov2-77-ft-v1",
        "backbone": f"dinov2-base-ft-ultimos{args.bloques}",
        "especies_filtradas": especies,
        "preprocesado": CONFIGURACION_PREPROCESADO.como_dict(),
        "origen_mlp": args.pesos_mlp.name if args.pesos_mlp else "desde cero",
    }
    (args.salida / "metadatos_modelo.json").write_text(json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8")
    t = resultado["test"]
    print(f"\nTEST — accuracy {t['accuracy']:.4f} — macro-F1 {t['macro_f1']:.4f} — top-3 {t['top3']:.4f} "
          f"(75_objetivo_ft: 0.937 / 0.918 / 0.990) — temperatura {temperatura}", flush=True)
    print(f"Modelo para inferencia: {args.salida / f'modelo_{PRUEBA}.pth'}", flush=True)


if __name__ == "__main__":
    main()
