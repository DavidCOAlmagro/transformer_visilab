"""Embeddings DINOv2 con pad-square (sin augmentation) para train/val/test de 75_objetivo. Solo experimento."""
import sys, time, subprocess, torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor
RAIZ = Path(r"C:\VISILAB\transformer\proyecto_transformer_v2"); sys.path.insert(0, str(RAIZ / "src"))
from embeddings import inicializar_dinov2, resolver_modelo_dinov2
from preprocesado import preparar_para_dinov2
from generar_leer_splits import leer_split
_ruta, _local = resolver_modelo_dinov2()
PROCESSOR = AutoImageProcessor.from_pretrained(_ruta, local_files_only=_local)
class DS(Dataset):
    def __init__(s, items): s.items = items
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        img = preparar_para_dinov2(Image.open(s.items[i][0]).convert("RGB"))
        return PROCESSOR(images=img, return_tensors="pt")["pixel_values"][0]
def temp_gpu():
    try: return int(subprocess.run(["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader"],capture_output=True,text=True).stdout.strip())
    except Exception: return 0
def freno_termico():
    t = temp_gpu()
    if t >= 85:
        print(f"GPU {t}C: pausa termica", flush=True)
        while temp_gpu() > 75: time.sleep(10)
        print(f"GPU {temp_gpu()}C: reanudo", flush=True)
def main():
    salida = Path(sys.argv[1]); salida.mkdir(exist_ok=True)
    _, model, device, _ = inicializar_dinov2(torch.device("cuda"))
    for split in ("val", "test", "train"):
        if (salida / f"embeddings_{split}.pt").exists():
            print("ya existe", split, flush=True); continue
        items = leer_split(RAIZ / "data" / "splits" / "75_objetivo" / f"{split}.txt")
        embs = []
        with torch.inference_mode():
            ds = DS(items)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(8) as pool:
              for n, ini in enumerate(range(0, len(ds), 64)):
                x = torch.stack(list(pool.map(ds.__getitem__, range(ini, min(ini + 64, len(ds))))))
                e = model(pixel_values=x.to(device)).pooler_output.float()
                embs.append((e / e.norm(dim=1, keepdim=True)).cpu())
                if n % 50 == 0: print(split, n * 64, "/", len(items), f"{temp_gpu()}C", flush=True)
                if n % 10 == 0: freno_termico()
        torch.save({"embeddings": torch.cat(embs), "etiquetas": [e for _, e in items]}, salida / f"embeddings_{split}.pt")
        print("guardado", split, flush=True)
if __name__ == "__main__":
    main()
