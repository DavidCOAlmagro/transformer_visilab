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
└── src/
    ├── main.py                         # Entrenamiento y evaluación completa
    ├── constantes.py                   # Rutas, especies y parámetros
    ├── embeddings.py                   # DINOv2 y aumentos de datos
    ├── clasificador.py                 # MLP compartido y cabezas de especie/género
    ├── CenterLoss.py                   # Pérdida auxiliar de compactación de embeddings
    ├── inferencia.py                   # Predicción de una imagen o carpeta
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
Para el selector de archivos gráfico de `inferencia.py`, en Linux instala también:

```bash
sudo apt install python3-tk
```
## Datos

Coloca tus imágenes en `data/imagenes_visilab(raw)/<algún_nombre_de_carpeta>/<especie>/*.jpg`. Cualquier subcarpeta dentro de `imagenes_visilab(raw)/` se recorre automáticamente, no hace falta declarar nombres de grupo en ningún sitio.

Las especies a incluir en el experimento se definen en `ESPECIES_FILTRADAS` (`src/constantes.py`) con el mismo nombre que su carpeta.

Al regenerar splits se crean particiones estratificadas reproducibles 70/15/15 en `data/splits/<PRUEBA>/`.

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
- `train` recibe aumento de datos y copias extra para clases minoritarias.
- Un `WeightedRandomSampler` y una loss ponderada (exponente configurable en `EXPONENTE_PESO_CLASE`) compensan el desbalance de especies.
- El clasificador usa un tronco `768 → 512 → 256`, con ReLU y dropout, y dos cabezas lineales: especie y género.
- La pérdida combina CrossEntropy de especie (con label smoothing), CrossEntropy de género (ponderada por `PESO_GENERO`) y, Center Loss (ponderada por `LAMBDA_CENTER_LOSS`) para compactar los embeddings de cada especie.
- AdamW, warmup + descenso coseno, early stopping según macro F1 de validación.

Al finalizar, se guardan en `modelos/<PRUEBA>/`: `mejor_modelo.pth`, `metadatos_modelo.json` (especies con las que se entrenó), curvas, matriz de confusión y reporte de test.

## Inferencia

```bash
python3 src/inferencia.py
```

Clasifica una imagen suelta o una carpeta entera (genera `predicciones.xlsx` con top-3 y marca de revisión bajo `UMBRAL_CONF`).

## Análisis de resultados

```bash
python3 src/evaluar_test_metricas.py  # Matriz, reporte y exactitud de género
python3 src/errores.py                # Rutas y etiquetas de las predicciones erróneas
python3 src/confusiones.py            # Confusiones repetidas entre especies
```

## Parámetros principales

Centralizados en `src/constantes.py`: dispositivo, batch size, épocas, learning rate, `EXPONENTE_PESO_CLASE`, `LAMBDA_CENTER_LOSS`, `PESO_GENERO`, paciencia, umbral de confianza y `ESPECIES_FILTRADAS`.