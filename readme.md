# Clasificación de Diatomeas mediante DINOv2

Pipeline de clasificación de especies de diatomeas (microorganismos) a partir de
imágenes de microscopía, desarrollado en VISILAB para un proyecto medioambiental.

El sistema usa **DINOv2** (Vision Transformer autosupervisado) como extractor de
características congelado (embeddings de 768 dimensiones), sobre el cual se
entrena un clasificador ligero (una capa lineal) para predecir la especie de
cada imagen.

## Estructura del proyecto

```
PROYECTO_TRANSFORMER/
├── requirements.txt
├── .gitignore
├── data/
│   ├── imagenes_visilab(raw)/
│   │   ├── Common_species/
│   │   ├── Unique_species/
│   │   └── Seleccion_5_especies_por_especie/
│   ├── embeddings_procesado/         # Embeddings generados
│   ├── metadata/                     # Documentación y referencias
│   └── splits/                       # Rutas de train/val/test generadas automáticamente
└── src/
    ├── constantes.py                 # Configuración global del proyecto
    ├── generar_leer_splits.py        # Generación y lectura de splits
    ├── embeddings.py                 # Extracción de embeddings con DINOv2
    ├── preparar_datos.py             # Codificación de etiquetas y filtrado
    ├── dataset.py                    # Dataset PyTorch basado en embeddings
    ├── dataloader.py                 # DataLoaders de train/val/test
    ├── clasificador.py               # Capa lineal para clasificación
    ├── entrenamiento.py              # Bucle de entrenamiento + early stopping
    ├── evaluar_test_metricas.py      # Métricas avanzadas y matriz de confusión
    └── main.py                       # Punto de entrada del pipeline
```

## Instalación

1. Clona el repositorio e instala las dependencias:

```bash
pip install -r requirements.txt
```

2. Si tienes GPU NVIDIA, instala antes la build de PyTorch con soporte CUDA
   correspondiente a tu versión (comprueba tu versión con `nvidia-smi`):

```bash
# Ejemplo para CUDA 12.1
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121
```

   Si no tienes GPU, el paso 1 ya instala la versión CPU sin nada más que hacer.

3. Configura el token de Hugging Face como variable de entorno (necesario para
   descargar el modelo DINOv2). Puedes generar uno en
   https://huggingface.co/settings/tokens (basta con permisos de lectura/"Read"):

```bash
# Linux / macOS
export HF_TOKEN="tu_token_aqui"

# Windows (cmd)
set HF_TOKEN=tu_token_aqui
```

## Uso del pipeline

Ejecutar el pipeline completo:

```bash
py src/main.py
```

El script:

1. Pregunta si quieres reentrenar o usar el modelo existente.
2. Genera splits reproducibles.
3. Calcula embeddings (con augmentation en train).
4. Entrena el clasificador lineal.
5. Guarda el mejor modelo.
6. Evalúa automáticamente en test.
7. Genera:
   - matriz de confusión
   - reporte de clasificación
   - curvas de entrenamiento

## Detalles técnicos

**Embeddings (DINOv2):**
- Modelo: `facebook/dinov2-base`
- Dimensión patch: 768
- Normalización L2 embeddings
- Modelo congelado (no se entrena); de DINOv2 solo se extraen los embeddings,
  solo se entrena nuestro clasificador lineal (`nn.Linear(768, num_clases)`)

**Clasificador:**
- Capa lineal: `Linear(768 → num_clases)`

  Es decir:
  - Entrada: un vector de 768 números (embedding de DINOv2)
  - Salida: un vector de tamaño num_clases (logits)

- Inicialización Xavier (normalización de pesos)
- Loss: CrossEntropy con label smoothing
- Optimizer: AdamW
- Scheduler: warmup + coseno

**Data augmentation (solo en train):**
- Flips horizontales y verticales
- Rotación 360°
- Traslación + escala
- Ajuste aleatorio de brillo y contraste
- Aplicación de un desenfoque gaussiano aleatorio
- Augmentation extra para clases minoritarias

## Flujo del sistema (PyTorch)

Este es el recorrido que sigue cada imagen dentro del pipeline:

```
Imagen
   ↓
Tensor
   ↓
Dataset
   ↓
DataLoader
   ↓
Modelo (DINOv2 congelado + capa lineal entrenable)
   ↓
Loss (CrossEntropy)
   ↓
Optimizer (AdamW)
   ↓
Actualización de pesos (solo en la capa lineal)
```

**Breve explicación del flujo:**

- **Tensor:** la imagen se carga con PIL y se transforma en un tensor normalizado
  mediante el `AutoImageProcessor` de DINOv2.
- **Dataset:** guarda embeddings y etiquetas, y permite acceder a cada muestra por índice.
- **DataLoader:** mezcla los datos (`shuffle=True` en train), los agrupa en batches,
  usa workers en paralelo y los sirve al modelo durante el entrenamiento.
- **Modelo:** DINOv2 + clasificador lineal.
- **Loss:** compara la predicción con la etiqueta real para ver cuánto ha fallado.
- **Optimizer:** calcula gradientes, aplica el scheduler (warmup + coseno, que ajusta
  el learning rate por épocas) y finalmente actualiza los pesos de
  `nn.Linear(768 → num_clases)`.