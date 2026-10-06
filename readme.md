# Clasificación de diatomeas con DINOv2

Pipeline para clasificar especies de diatomeas a partir de imágenes de microscopía. Usa `facebook/dinov2-base` como extractor congelado de embeddings de 768 dimensiones y entrena una red neuronal ligera sobre ellos.

El modelo realiza dos predicciones a la vez: especie (tarea principal) y género (tarea auxiliar de regularización, más Center Loss opcional para compactar embeddings por clase). Las especies a usar, las rutas y los hiperparámetros se definen en `src/constantes.py`.

## Estructura

```text
proyecto_transformer_v2/
├── requirements.txt
├── manual_de_uso.txt                            
├── readme.md                           
├── data/
│   ├── imagenes_visilab(raw)/          # Carpetas de especies (cualquier subcarpeta se detecta sola)
│   ├── splits/<PRUEBA>/                # train.txt, val.txt y test.txt por experimento
│   └── embeddings_procesado/<PRUEBA>/  # Embeddings guardados (.pt) por experimento
├── modelos/
│   └── <PRUEBA>/                       # Pesos y resultados de cada experimento
├── Inferir/
│   ├── infer_and_split_resnet_single_folder.py  # Inferencia YOLO + DINOv2/ResNet50
│   └── txt_classes/                    # Listas de clases DINOv2 (77) y ResNet50 (78)
└── src/
    ├── main.py                         # Entrenamiento y evaluación completa
    ├── constantes.py                   # Rutas, especies y parámetros
    ├── embeddings.py                   # DINOv2 y aumentos de datos
    ├── preprocesado.py                 # Padding cuadrado común a train/inferencia
    ├── auditoria_dataset.py            # Auditoría global y recomendación de especies
    ├── clasificador.py                 # MLP compartido y cabezas de especie/género
    ├── CenterLoss.py                   # Pérdida auxiliar de compactación de embeddings
    ├── evaluar_campo.py                # Accuracy por imagen en campo desde los Excel de inferencia
    ├── evaluar_test_metricas.py        # Métricas, matriz y curvas
    ├── errores.py                      # Lista de errores del conjunto de test
    └── confusiones.py                  # Pares de especies más confundidos
```

Cada experimento (`PRUEBA` en `constantes.py`, o `--prueba` por línea de comandos) tiene su propia carpeta de splits, embeddings y modelo.

## Instalación

Se recomienda Python 3.10 o superior. Crea y activa un entorno virtual e instala las dependencias:

```bash
pip install -r requirements.txt
```

Para usar una GPU NVIDIA, instala antes la variante de PyTorch que corresponda a tu versión de CUDA. Por ejemplo, para CUDA 12.1:

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
## Datos

Coloca tus imágenes en `data/imagenes_visilab(raw)/<algún_nombre_de_carpeta>/<especie>/*.jpg`. Cualquier subcarpeta dentro de `imagenes_visilab(raw)/` se recorre automáticamente, no hace falta declarar nombres de grupo en ningún sitio.

Las especies a incluir en el experimento se definen en `ESPECIES_FILTRADAS` (`src/constantes.py`) con el mismo nombre que su carpeta.

Al regenerar splits se crean particiones estratificadas reproducibles 70/15/15 en `data/splits/<PRUEBA>/`.
El experimento DINOv2 actual mantiene exactamente 77 clases y ordena las clases
desde `metadatos_modelo.json`. No se añaden especies automáticamente.

Para auditar el dataset completo antes de decidir una ampliación:

```bash
python3 src/auditoria_dataset.py data/imagenes_visilab(raw) \
  --clases-activas modelos/75_objetivo/metadatos_modelo.json \
  --minimo 20 --salida auditoria_dataset.json
```

El informe incluye total global, distribución por clase y grupo inferido,
duplicados exactos y candidatas adicionales. Una candidata debe revisarse
manualmente y debe generar un experimento nuevo; no se modifica el modelo de
77 clases en silencio. Si la auditoría detecta duplicados, regenera los splits
tras decidir cómo agruparlos para evitar fuga entre train/val/test.

## Entrenar y evaluar

```bash
python3 src/main.py
```

Sin argumentos, el programa pregunta interactivamente si reentrenar, regenerar splits y recalcular embeddings. También admite flags para automatizar corridas sin preguntas:

```bash
python3 src/main.py --reentrenar s --regenerar-splits n --recalcular-embeddings n --prueba "mi_experimento"
```

| Flag | Valores | Qué hace |
|---|---|---|
| `--reentrenar` | s / n | Entrena de nuevo o usa el modelo ya guardado |
| `--regenerar-splits` | s / n | Regenera train/val/test |
| `--recalcular-embeddings` | s / n | Recalcula los `.pt` de DINOv2 |
| `--prueba` | texto | Carpeta de experimento (sobreescribe `PRUEBA`) |


Durante el entrenamiento:

- DINOv2 permanece congelado, solo calcula embeddings.
- Antes de pasar por `AutoImageProcessor`, cada ROI se centra en un lienzo
  cuadrado con padding de color medio del borde (`dinov2-pad-square-v1`).
  Así se evita perder extremos de ROI alargados por un center crop y no se
  deforma la imagen.
- `train` recibe aumento de datos y copias extra para clases minoritarias.
- Un `WeightedRandomSampler` y una loss ponderada (exponente configurable en `EXPONENTE_PESO_CLASE`) compensan el desbalance de especies.
- El clasificador usa un tronco `768 → 512 → 256`, con ReLU y dropout, y dos cabezas lineales: especie y género.
- La pérdida principal es CrossEntropy de especie. Género y Center Loss se
  conservan como ablaciones configurables, pero están desactivados por defecto
  (`PESO_GENERO=0` y `LAMBDA_CENTER_LOSS=0`) para no perjudicar la clasificación
  sin evidencia experimental.
- AdamW, warmup + descenso coseno, early stopping según macro F1 de validación.

Al finalizar, se guardan en `modelos/<PRUEBA>/`: `modelo_<PRUEBA>.pth`, `metadatos_modelo.json` (especies con las que se entrenó), curvas, matriz de confusión y reporte de test.

## Inferencia YOLO + DINOv2/ResNet50

```bash
python3 Inferir/infer_and_split_resnet_single_folder.py --classifier both
```

Recorre recursivamente `Inferir/imagenes_inferencia/` (o la carpeta que se pase
como primer argumento), ejecuta YOLO una vez por imagen y clasifica los recortes
en lote con DINOv2 y/o ResNet50 (`--classifier dinov2|resnet|both`;
`--classifier-architecture` es un alias).

Pesos por defecto (no versionados; `.gitignore` excluye `*.pt`, `*.pth`, `*.ckpt`):

| Modelo | Ruta por defecto | Opción |
|---|---|---|
| YOLO | `Inferir/yolo_dinov2/yolo_best.pt` | `--yolo-weights` |
| DINOv2 + MLP | `modelos/75_objetivo_pad/modelo_75_objetivo_pad.pth` | `--dino-weights` |
| ResNet50 | `Inferir/yolo_dinov2/resnet50_checkpoint_epoch50.pth` | `--resnet-weights` |

Las clases DINOv2 se leen de `metadatos_modelo.json` en la carpeta de los pesos y
se contrastan con `Inferir/txt_classes/classes_77(dino).txt` (deben ser exactamente
77 y en el mismo orden). ResNet50 usa `classes_78(resnet).txt` (78 clases). Cualquier
incompatibilidad de arquitectura, clases o archivos produce un error explícito.

El preprocesado DINOv2 de inferencia (`dinov2-pad-square-v1` + normalización L2) debe
coincidir con el del entrenamiento: `75_objetivo_pad` se entrenó así; el modelo
antiguo `75_objetivo` se entrenó con center-crop y no es coherente con él.
Al cambiar el padding o las augmentations hay que recalcular los embeddings y
reentrenar el MLP.

Salida: un Excel combinado, un Excel por modelo (top-1/2/3 con confianza, confianza
YOLO y marca de revisión) y las imágenes anotadas en `bbox/dinov2/` y `bbox/resnet/`.
`--threshold 0.80` marca para revisión las predicciones de baja confianza;
`--conf` (0.30) es el umbral de YOLO e `--imgsz` (1024) su tamaño de entrada.
Las carpetas generadas (`crops`, `bbox`, `runs`, `resultados`, ...) se excluyen de la
búsqueda, `--output-dir` no puede estar dentro de la carpeta de entrada y los Excel se
intentan guardar también si la inferencia falla. `--reinhard-reference` aún no está
implementado y produce un error explícito.

```bash
python3 Inferir/infer_and_split_resnet_single_folder.py otra_carpeta --classifier dinov2 --device cuda
python3 Inferir/infer_and_split_resnet_single_folder.py --dino-weights modelos/<PRUEBA>/modelo_<PRUEBA>.pth
```

### Evaluación en campo

Con los Excel de la inferencia y una tabla de etiquetas por imagen (`imagen`, `etiqueta`):

```bash
python3 src/evaluar_campo.py --dino classification_results_dinov2.xlsx   --resnet classification_results_resnet.xlsx --etiquetas cruce_ground_truth.xlsx   --salida informe_campo.xlsx
```

Calcula la accuracy por imagen (reglas voto, recorte mayor y suma de confianza) de
cada modelo y del ensamble. `Debris`, `Fragments` y las clases `*_fp` (posición
pleural) no votan en la variante oficial.

## Análisis de resultados

```bash
python3 src/evaluar_test_metricas.py  # Matriz, reporte y exactitud de género
python3 src/errores.py                # Rutas y etiquetas de las predicciones erróneas
python3 src/confusiones.py            # Confusiones repetidas entre especies
```

## Parámetros principales

Centralizados en `src/constantes.py`: dispositivo, batch size, épocas, learning rate, `EXPONENTE_PESO_CLASE`, `LAMBDA_CENTER_LOSS`, `PESO_GENERO`, `USAR_CENTER_LOSS`, `USAR_PERDIDA_GENERO`, paciencia, umbral de confianza y `ESPECIES_FILTRADAS`.