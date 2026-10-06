# 01 — Visión del proyecto

## Qué es
Pipeline de deep learning para **clasificar especies de diatomeas** (microalgas usadas
como bioindicadores de calidad del agua) a partir de imágenes de microscopía (ROIs).
Proyecto de VISILAB. Repo: https://github.com/DavidCOAlmagro/transformer_visilab
Autor: David Calzado Olmo. Rama principal: `main`.

## Enfoque técnico (por qué así)
- Pocas imágenes en muchas clases (algunas < 50) → **transfer learning + feature extraction**.
- `facebook/dinov2-base` **congelado** genera embeddings de 768 (pooler_output / CLS).
- Se entrena solo un **MLP ligero** sobre embeddings precalculados → entrenamiento en minutos.
- Cabeza auxiliar de **género** y **Center Loss** existen pero están **desactivadas** por defecto.
- **Temperature scaling** horneado en los pesos para que la confianza sea honesta.
- Detección de **especies desconocidas** por distancia a centroides (percentil 95 en val).

## Caso de uso final
Un usuario (laboratorio) pone fotos completas de microscopio en `Inferir/imagenes_inferencia/`.
YOLO detecta cada diatomea, cada recorte se clasifica (DINOv2-MLP y ResNet50 en paralelo) y se
genera un Excel con top-1/2/3, confianza y marca de **revisión manual** si confianza < 0.80.

## Estado actual (2026-10-06)
- Experimento vigente: **`75_objetivo`** (nombre histórico; tiene **77 clases**, incluye `Debris` y `Fragments`).
- Mejor resultado registrado (2026-09-24): macro-F1 test **0.877**, accuracy **0.891**,
  top-3 **0.977**, acc. género **0.955**, mejor época 34.
- Rechazo de desconocidas (percentil 95): 16 % de desconocidas rechazadas, 2.6 % falsos rechazos
  → el detector de desconocidas es **débil** (ver `08_deuda_tecnica.md`).
- Confusiones dominantes: `Nitzschia_inconspicua ↔ Nitzschia_soratensis`,
  `Planothidium_frequentissimum ↔ P. lanceolatum`, `Cyclotella_atomus ↔ Discostella_pseudostelligera`,
  `Mayamaea_permitis ↔ Fistulifera_saprophila`.
- `metadatos_modelo.json` de 75_objetivo indica que los **pesos actuales son previos** al
  preprocesado `dinov2-pad-square-v1`: hay que **recalcular embeddings y reentrenar** para aplicarlo.
- ResNet50 (78 clases) es un modelo externo/heredado usado solo para comparar en inferencia.

- **En campo** (inferencia sobre 12 857 imágenes reales): acc por imagen DINOv2 **0.675** vs ResNet50 **0.760**
  (suma de confianza). Detalle y causas en `12_estado_actual.md` y `13_diagnostico_dinov2_campo.md`.

## Objetivos (confirmados 2026-10-06)
- Construir dataset con **ground truth** desde inferencias de campo para entrenar **nuevas especies**.
- Evaluar la inferencia en ambiente **no controlado**.
- **Superar a ResNet50** con DINOv2 en campo.

## Objetivos abiertos (hipótesis, confirmar con el usuario)
- Mejorar especies confundidas del mismo género.
- Mejorar el rechazo de especies desconocidas.
- Posible ampliación de clases (vía `auditoria_dataset.py`, siempre como experimento nuevo).
- Comparar DINOv2 vs ResNet50 sobre ground truth (`analisis_clasificadores.xlsx`).
