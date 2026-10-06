# 06 — Configuración, CLI y entorno

## `src/constantes.py` → `VARIABLES_GLOBALES`
| Clave | Valor | Notas |
|---|---|---|
| `PRUEBA` | `"75_objetivo"` | sobrescribible con `--prueba` |
| `RUTA_BASE` | `<repo>/data` | |
| `RUTA_SPLITS` / `RUTA_EMBEDDINGS` | `data/splits/<PRUEBA>`, `data/embeddings_procesado/<PRUEBA>` | main.py las recalcula tras `--prueba` |
| `RUTA_MODELOS` | `<repo>/modelos` | |
| `EXTENSIONES_VALIDAS` | jpg, jpeg, png, tif, tiff, bmp | (auditoría/inferencia heredada añaden webp) |
| `HF_TOKEN` | env `HF_TOKEN` | |
| `ESPECIES_FILTRADAS` | set de 77 especies | debe coincidir con metadatos de la PRUEBA |
| `BATCH_SIZE` | 32 | |
| `DIM_EMBEDDING` | 768 | dinov2-base |
| `num_epocas` | 40 | (única clave en minúscula) |
| `NUM_WORKERS` / `PIN_MEMORY` / `PERSISTENT_WORKERS` | 4 / True / True | |
| `EPOCAS_WARMUP` | 3 | |
| `PACIENCIA` | 7 | early stopping (notas.md dice 5: desactualizado) |
| `UMBRAL_CONF` | 0.80 | revisión manual en inferencia |
| `LEARNING_RATE` / `WEIGHT_DECAY` | 3e-4 / 1e-4 | AdamW |
| `LABEL_SMOOTHING` | 0.05 | |
| `PESO_GENERO` / `USAR_PERDIDA_GENERO` | 0.0 / False | ablación |
| `LAMBDA_CENTER_LOSS` / `USAR_CENTER_LOSS` | 0.0 / False | ablación |
| `MINIMO_IMAGENES_POR_ESPECIE` | 5 | solo advertencia |
| `EXPONENTE_PESO_CLASE` | 0.3 | sampler y loss: peso = 1/n^exp |
| `DIM_CAPA_1/2`, `DROPOUT_CAPA_1/2` | 512, 256, 0.3, 0.2 | |
| `DEVICE` | cuda si disponible | inferencia lo sobrescribe |

Otros parámetros fijos en código: semilla 42; split 0.30 → 0.50; `max_copias=3`; percentil centroides 95;
`UMBRAL_MINIMO=5` en confusiones; `clip_grad_norm 1.0`; LBFGS 100 iter lr 0.01.

Variables de entorno: `HF_TOKEN`, `DINOV2_MODEL_PATH` (carpeta local del modelo HF).

## CLI de `main.py`
```bash
python3 src/main.py --reentrenar s|n --regenerar-splits s|n --recalcular-embeddings s|n --prueba NOMBRE
```
Sin flags pregunta interactivamente. Si no existe `.pth`, entrena directamente.
Otros entry points: ver `03` y `05`. Lista de combinaciones en `comandos.txt`.

## Entorno
- Windows 11, **Python 3.13.4** (README recomienda ≥ 3.10). Origen Linux (máquina VISILAB).
- Dependencias (`requirements.txt`): torch 2.7.1, torchvision 0.22.1, transformers 5.12.1,
  huggingface-hub 1.5.0, scikit-learn 1.9.0, numpy 2.0.1, matplotlib 3.9.0, pillow 11.3.0, tqdm,
  pandas, xlsxwriter, ultralytics (sin versión fijada en las tres últimas).
- GPU: instalar torch CUDA antes (cu118/cu121/cu124).
- Las dependencias (torch 2.7.1+cu118) están en el **Python del sistema**
  (`C:\Users\david\AppData\Local\Programs\Python\Python313`); la `.venv` del repo **no** las tiene.
- Ejecutar tests: `python -m unittest tests.test_preprocesado` desde la raíz
  (3 tests OK a 2026-10-06; `discover -s tests` falla por los imports `src.`).
