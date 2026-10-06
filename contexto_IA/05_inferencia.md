# 05 — Inferencia

Hay **dos** implementaciones que hacen lo mismo con distinto nivel de madurez.

## A) `src/inferencia.py` (oficial en README, más simple)
```bash
python3 src/inferencia.py [entrada] --classifier {dinov2,resnet,both} --device cuda --conf 0.25 --output x.xlsx
```
- Entrada por defecto `Inferir/imagenes_inferencia/` (recursivo, `rglob`).
- YOLO por imagen (sin batch), cada recorte se clasifica uno a uno.
- `ClasificadorDino` usa `modelo.cargar_modelo_entrenado` → incluye detección de **Desconocida**
  por centroides (sobrescribe `especie_predicha="Desconocida"`, `motivo_revision="especie_desconocida"`).
- `ClasificadorResnet`: `resnet50` con `fc = Dropout(0.5)-Linear(2048,256)-ReLU-Dropout(0.3)-Linear(256,78)`,
  `Resize((256,256))` + normalización ImageNet. Acepta `state_dict` con prefijo `module.`.
- Umbral de revisión: `UMBRAL_CONF` (0.80) de `constantes.py`.
- Salida: un solo Excel (`predicciones.xlsx` en la carpeta de entrada).
- ⚠️ Bugs conocidos: no L2-normaliza el embedding; fila con `"x2": y2` y sin `y2`. Ver `08`.

## B) `Inferir/infer_and_split_resnet_single_folder.py` (heredado, más completo, 690 líneas)
```bash
python3 Inferir/infer_and_split_resnet_single_folder.py [input] --classifier both --threshold 0.80 --conf 0.30 --imgsz 1024 --output-dir DIR
```
- Batch de recortes, carga de imágenes en hilos (`IMAGE_LOAD_WORKERS=3`), autocast fp16 en CUDA.
- **Sí** L2-normaliza el embedding. **No** usa detección de desconocidas (reconstruye tronco+cabeza a mano).
- Excluye carpetas generadas (`crops`, `bbox`, `runs`, `resultados`, `resultados_inferencia`…).
- `--output-dir` no puede estar dentro de la entrada. Excel combinado + uno por modelo + imágenes
  anotadas en `bbox/dinov2/` y `bbox/resnet/`. Guarda Excel parcial en `finally`.
- `--reinhard-reference` existe pero lanza error a propósito (no implementado).
- Hay `.backup` del script en el repo.

## Pesos necesarios (no versionados)
| Archivo | Ruta por defecto | Flag |
|---|---|---|
| YOLO | `Inferir/yolo_dinov2/yolo_best.pt` | `--yolo-weights` |
| DINOv2-MLP | `modelos/75_objetivo/modelo_75_objetivo.pth` | `--dino-weights` |
| ResNet50 | `Inferir/yolo_dinov2/resnet50_checkpoint_epoch50.pth` | `--resnet-weights` |
| Clases DINO (77) | `Inferir/txt_classes/classes_77(dino).txt` | `--dino-classes` |
| Clases ResNet (78) | `Inferir/txt_classes/classes_78(resnet).txt` | `--resnet-classes` |

A fecha 2026-10-06 la carpeta `Inferir/yolo_dinov2/` **no existe localmente**.

## Columnas del Excel (prefijo `dinov2_` / `resnet_`)
`imagen, deteccion, confianza_yolo_%, x1, y1, x2, y2, <p>_especie_predicha, <p>_especie_mas_parecida,
<p>_confianza_%, <p>_revisar, <p>_motivo_revision, <p>_top{1..3}, <p>_top{1..3}_confianza_%`.
Sin detecciones → fila con `motivo_revision = sin_detecciones`.

## Reglas
- La lista DINO debe ser **idéntica y en el mismo orden** que `sorted(metadatos.especies_filtradas)`.
- El preprocesado de inferencia debe ser **idéntico** al de los embeddings de entrenamiento
  (`preparar_para_dinov2` + processor + L2 norm).
- Unificar ambas implementaciones es deuda pendiente (ver `08`).
