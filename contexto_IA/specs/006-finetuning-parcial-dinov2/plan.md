# PLAN 006

## Configuración (una sola corrida; ajustes según la curva)
| | Valor | Motivo |
|---|---|---|
| Backbone | `facebook/dinov2-base`; **entrenables: los 4 últimos bloques (de 12) + `layernorm` final** (~28 M de 87 M parámetros) | los bloques altos codifican lo específico de la tarea; los bajos (bordes, texturas) se mantienen |
| Cabeza | El MLP `ClasificadorDiatomeas` (768→512→256→77), **inicializado con los pesos de `75_objetivo_pad`** | parte de un buen punto: no empieza de cero |
| Learning rates | backbone **1e-5**, MLP **1e-4**; AdamW wd 1e-4; warmup 1 época + coseno | LR bajo en el backbone para no destruir lo preentrenado |
| Épocas | máx. **15**, early stopping por macro-F1 de val (paciencia 4) | |
| Batch / precisión | 32, AMP (bf16/fp16) | cabe en 8 GB: los bloques congelados no guardan activaciones |
| Entrada | `preparar_para_dinov2` (pad-square) → aumentación **online** → pad → `AutoImageProcessor` (224) | igual que `75_objetivo_pad`, pero con una variante nueva en cada época |
| Aumentación | `crear_augmentation()` del repo + `RandomAffine(scale 0.8–1.2, shear 10)` como en ResNet | más variación de encuadre (YOLO) y escala |
| Desbalance y pérdida | Sampler y CE con label smoothing 0.05 y pesos 1/n^0.3 (como `main.py`) | comparabilidad con `75_objetivo_pad` |
| Embedding | CLS → L2 → MLP (igual que ahora) | compatibilidad con la inferencia |
| Calibración | temperatura en val, horneada en `cabeza_especie` (como `calibracion_confianza`) | confianzas comparables |

## Checkpoint y compatibilidad con la inferencia
- Un único `state_dict`: claves `backbone.*` (DINOv2 completo) + `tronco.*` + `cabeza_especie.*` (+ `cabeza_genero.*`).
- `DinoClassifier` (Inferir): si el checkpoint trae claves `backbone.*`, las carga en el backbone (strict); si no, comportamiento actual.
- `metadatos_modelo.json` con `especies_filtradas`, `preprocesado` y `"backbone": "dinov2-base-ft-ultimos4"`.

## Ejecución
1. **Windows (humo):** `--max-imagenes 256 --epocas 1 --batch 8` en la RTX 3050 → valida el código, la memoria y el checkpoint.
2. **Ubuntu:** `git pull` y luego `python3 contexto_IA/specs/006-finetuning-parcial-dinov2/scripts/finetune_dinov2.py --temperatura-pausa 0 2>&1 | tee log_ft.txt` (sin freno térmico en la torre, a petición del usuario).
   Reanudable (`ultimo.pth` por época); comprueba antes que las imágenes de los splits existen.
   Estimación a confirmar con la 1.ª época: ~8–15 min/época (46k imágenes pasan por todo DINOv2 + val) → **2–4 h**.
3. **Campo:** inferencia con `--classifier dinov2 --dino-weights modelos/75_objetivo_ft/modelo_75_objetivo_ft.pth` → `evaluar_campo.py`.

## Lectura de la curva ("a ver cómo evoluciona")
El log imprime por época loss train/val, accuracy y macro-F1 de val. Si val sube y luego cae mientras train sigue bajando → sobreajuste
→ próxima corrida con 2 bloques o LR 5e-6. Si val apenas mejora sobre 0.894 (val de `75_objetivo_pad`) → probar 6–8 bloques o LoRA.
