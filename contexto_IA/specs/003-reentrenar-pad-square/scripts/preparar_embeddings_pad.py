"""
Spec 003 — Ensambla los embeddings de la PRUEBA 75_objetivo_pad.

- val/test: reutiliza los .pt pad-square sin augmentation de la spec 002
  (tras comprobar que coinciden con embeddings.get_embedding).
- train: originales de la spec 002 + vistas aumentadas calculadas igual que
  embeddings.calcular_embeddings (1 aumentada + copias extra por especie),
  con la misma cadena pad -> augmentation -> pad, pero en lotes en GPU.

Uso: python preparar_embeddings_pad.py <carpeta_embeddings_spec002>
"""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image

RAIZ = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(RAIZ / "src"))
from embeddings import get_embedding, inicializar_dinov2  # noqa: E402
from generar_leer_splits import leer_split  # noqa: E402
from preparar_datos import (calcular_conteo_por_especie,  # noqa: E402
                            calcular_copias_extra_por_especie, fijar_semilla)
from preprocesado import preparar_para_dinov2  # noqa: E402

PRUEBA = "75_objetivo_pad"
RUTA_SPLITS = RAIZ / "data" / "splits" / PRUEBA
RUTA_SALIDA = RAIZ / "data" / "embeddings_procesado" / PRUEBA
LOTE = 64


def temperatura_gpu() -> int:
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.strip())
    except (OSError, ValueError):
        return 0


def freno_termico() -> None:
    """Pausa si la GPU llega a 85 ºC hasta que baje a 75 ºC."""
    if temperatura_gpu() >= 85:
        print(f"GPU {temperatura_gpu()} C: pausa térmica", flush=True)
        while temperatura_gpu() > 75:
            time.sleep(10)
        print(f"GPU {temperatura_gpu()} C: reanudo", flush=True)


def main() -> None:
    origen = Path(sys.argv[1])
    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    fijar_semilla(42)
    processor, model, device, augmentation = inicializar_dinov2()

    # 1) Verificación de equivalencia con el pipeline oficial
    val = leer_split(RUTA_SPLITS / "val.txt")
    datos_val = torch.load(origen / "embeddings_val.pt", weights_only=True)
    assert datos_val["etiquetas"] == [e for _, e in val], "orden de val distinto"
    muestra = range(0, len(val), len(val) // 64)[:64]
    oficial = torch.cat([get_embedding(val[i][0], processor, model, device, augmentation, False)
                         for i in muestra])
    diferencia = (oficial - datos_val["embeddings"][list(muestra)]).abs().max().item()
    print(f"Diferencia máxima con get_embedding: {diferencia:.2e}", flush=True)
    if diferencia > 1e-3:
        raise SystemExit("Los embeddings de la spec 002 no son equivalentes; abortar.")

    for split in ("val", "test"):
        datos = torch.load(origen / f"embeddings_{split}.pt", weights_only=True)
        assert datos["etiquetas"] == [e for _, e in leer_split(RUTA_SPLITS / f"{split}.txt")]
        torch.save(datos, RUTA_SALIDA / f"embeddings_{split}.pt")
        print(f"{split}: {len(datos['etiquetas'])} copiados", flush=True)

    # 2) train: originales + vistas aumentadas
    train = leer_split(RUTA_SPLITS / "train.txt")
    originales = torch.load(origen / "embeddings_train.pt", weights_only=True)
    assert originales["etiquetas"] == [e for _, e in train], "orden de train distinto"
    copias = calcular_copias_extra_por_especie(calcular_conteo_por_especie(train))
    # Cada imagen de train: 1 vista aumentada + copias extra de su especie
    tareas = [(i, ruta) for i, (ruta, especie) in enumerate(train)
              for _ in range(1 + copias.get(especie, 0))]
    print(f"train: {len(train)} originales + {len(tareas)} aumentadas", flush=True)

    def vista_aumentada(ruta: str) -> torch.Tensor:
        imagen = preparar_para_dinov2(Image.open(ruta).convert("RGB"))
        imagen = preparar_para_dinov2(augmentation(imagen))
        return processor(images=imagen, return_tensors="pt")["pixel_values"][0]

    aumentados = []
    with torch.inference_mode(), ThreadPoolExecutor(8) as hilos:
        for n, inicio in enumerate(range(0, len(tareas), LOTE)):
            lote = tareas[inicio:inicio + LOTE]
            x = torch.stack(list(hilos.map(vista_aumentada, [ruta for _, ruta in lote])))
            e = model(pixel_values=x.to(device)).pooler_output.float()
            aumentados.append((e / e.norm(dim=1, keepdim=True)).cpu())
            if n % 50 == 0:
                print(f"train aug {inicio}/{len(tareas)} {temperatura_gpu()}C", flush=True)
            if n % 10 == 0:
                freno_termico()

    # Mismo orden que calcular_embeddings: original seguido de sus vistas aumentadas
    aumentados = torch.cat(aumentados)
    por_imagen: dict[int, list[int]] = {}
    for fila, (i, _) in enumerate(tareas):
        por_imagen.setdefault(i, []).append(fila)
    embeddings, etiquetas = [], []
    for i, (_, especie) in enumerate(train):
        embeddings.append(originales["embeddings"][i:i + 1])
        embeddings.append(aumentados[por_imagen[i]])
        etiquetas.extend([especie] * (1 + len(por_imagen[i])))
    datos = {"embeddings": torch.cat(embeddings), "etiquetas": etiquetas}
    torch.save(datos, RUTA_SALIDA / "embeddings_train.pt")
    print(f"train guardado: {len(etiquetas)} filas", flush=True)


if __name__ == "__main__":
    main()
