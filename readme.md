# Clasificación de diatomeas con DINOv2

Pipeline para clasificar especies de diatomeas a partir de imágenes de microscopía. Usa `facebook/dinov2-base` como extractor congelado de embeddings de 768 dimensiones y entrena una red neuronal ligera sobre ellos.

El modelo actual realiza dos predicciones a la vez: especie (tarea principal) y género (tarea auxiliar de regularización). Está configurado para 20 especies filtradas; la selección, las rutas y los hiperparámetros se definen en `src/constantes.py`.

## Estructura

```text
proyecto_transformer_v2/
├── requirements.txt
├── data/
│   ├── imagenes_visilab(raw)/          # Imágenes originales organizadas por especie
│   ├── splits/                         # train.txt, val.txt y test.txt
│   └── embeddings_procesado/           # Embeddings guardados (.pt)
├── modelos/
│   └── 20_especies/                    # Pesos y resultados del experimento
└── src/
    ├── main.py                         # Entrenamiento y evaluación completa
    ├── constantes.py                   # Rutas, especies y parámetros
    ├── embeddings.py                   # DINOv2 y aumentos de datos
    ├── clasificador.py                 # MLP compartido y cabezas de especie/género
    ├── inferencia.py                   # Predicción de una imagen o carpeta
    ├── evaluar_test_metricas.py        # Métricas, matriz y curvas
    ├── errores.py                      # Lista de errores del conjunto de test
    └── confusiones.py                  # Pares de especies más confundidos
```

## Instalación

Se recomienda Python 3.10 o superior. Crea y activa un entorno virtual y después instala las dependencias:

```
pip install -r requirements.txt
```

Para usar una GPU NVIDIA, instala antes la variante de PyTorch que corresponda a tu versión de CUDA. Por ejemplo, para CUDA 12.1:

```
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Configura un token de Hugging Face con permiso de lectura, necesario para descargar DINOv2 la primera vez:

```powershell
# PowerShell, solo para la sesión actual
$env:HF_TOKEN = "tu_token"



## Datos

Las imágenes se buscan en `data/imagenes_visilab(raw)/` dentro de los grupos definidos en `src/preparar_datos.py`. Cada especie debe corresponder a una carpeta y su nombre debe estar incluido en `ESPECIES_FILTRADAS` de `src/constantes.py`.

Al regenerar los splits, se crean particiones estratificadas reproducibles de 70 % entrenamiento, 15 % validación y 15 % test en `data/splits/`.

## Entrenar y evaluar

Desde la raíz del proyecto:

```
py src/main.py
```

El programa pregunta si se desea reentrenar y, en ese caso, si se deben regenerar los splits y los embeddings. Durante el entrenamiento:

- DINOv2 permanece congelado y solo calcula embeddings.
- Para `train` se aplica aumento de datos y copias adicionales para clases minoritarias.
- Un `WeightedRandomSampler` compensa el desbalance de especies.
- El clasificador usa un tronco `768 → 512 → 256`, con ReLU y dropout, y dos cabezas lineales: especie y género.
- La pérdida combina `CrossEntropy` de especie (con label smoothing) y de género, ponderada por `PESO_GENERO`.
- Se usa AdamW, warmup seguido de descenso coseno y early stopping según el macro F1 de validación.

Al finalizar, se guardan en `modelos/20_especies/` el mejor modelo (`mejor_modelo.pth`), las curvas de entrenamiento, la matriz de confusión de test y el reporte de clasificación. El nombre de la carpeta de experimento se controla con `PRUEBA`.

## Inferencia

Ejecuta el asistente interactivo:

```
py src/inferencia.py
```

Permite clasificar una sola imagen o todas las imágenes de una carpeta. En el segundo caso crea `predicciones.xlsx` con la especie más probable, el top 3, sus confianzas y una marca de revisión para resultados por debajo del umbral `UMBRAL_CONF`.

## Análisis de resultados

Con un modelo y los embeddings de test disponibles, se pueden ejecutar:

```
py src/evaluar_test_metricas.py  # Matriz, reporte y exactitud de género
py src/errores.py                # Rutas y etiquetas de las predicciones erróneas
py src/confusiones.py            # Confusiones repetidas entre especies
```

## Parámetros principales

Los parámetros se centralizan en `src/constantes.py`: dispositivo (`cuda` cuando está disponible), tamaño de lote, número de épocas, número de workers, tasa de aprendizaje, regularización, paciencia, umbral de confianza y especies incluidas. Modifícalos allí antes de iniciar un nuevo experimento.
