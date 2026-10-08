# SPEC 005 — ResNet50 entrenada con NUESTROS datos (comparación justa)

- **Estado:** Pospuesta (2026-10-07): el usuario prioriza el fine-tuning parcial de DINOv2 (spec 006)
- **Fecha:** 2026-10-07
- **Relacionado:** `13_diagnostico_dinov2_campo.md` §1b (receta de ResNet), spec 003 (DINO pad: campo 0.788).

## 1. Pregunta
En campo, la ResNet50 recibida (0.827) supera a DINOv2-pad (0.788). ¿Es por **cómo se entrena** (fine-tuning completo +
aumentación online) o por **con qué datos** se entrenó (dataset desconocido, quizá con imágenes del dominio de campo o del propio test)?

## 2. Experimento
Entrenar una ResNet50 con la receta del script recibido, pero con **exactamente nuestros datos**: las 77 clases y los splits
train/val/test de `75_objetivo`. Medirla en el test interno y en campo, y compararla con DINO-pad y con la ResNet recibida.

| Resultado en campo de "ResNet-nuestra" | Interpretación | Siguiente paso |
|---|---|---|
| ≈ 0.83 (como la recibida) | La **receta** explica la diferencia | Fine-tuning parcial + aumentación online para DINO |
| ≈ 0.79 o menos (como DINO) | La diferencia venía de **sus datos** (posible fuga) | Datos de campo con ground truth en el train; preguntar al autor |
| Intermedio | Las dos cosas | Ambas líneas |

## 3. Alcance
- **Incluye:** script de entrenamiento autocontenido (lanzado en Ubuntu, 8 GB VRAM); evaluación en test interno con el mismo formato que DINO;
  inferencia de campo con la ResNet nueva y `evaluar_campo.py`.
- **Cambio mínimo en `Inferir/infer_and_split_resnet_single_folder.py`**: el número de clases de ResNet sale del fichero de clases
  (hoy fijado a 78) para poder cargar una ResNet de 77 clases. El comportamiento por defecto (78) no cambia.
- **No incluye:** sustituir la ResNet recibida en producción; tocar DINO, `data/`, los splits ni los pesos existentes.

## 4. Criterios de aceptación
- [ ] CA1: el entrenamiento termina y guarda en `modelos/resnet50_75_objetivo/`: mejor checkpoint (macro-F1 de val), checkpoint de la última época,
      `clases.txt`, `metricas.json` (test: accuracy, macro-F1, top-3) y curvas.
- [ ] CA2: el script verifica que todas las imágenes de los splits existen en Ubuntu (o informa cuántas faltan) antes de entrenar.
- [ ] CA3: el script de inferencia carga la ResNet nueva con `--resnet-weights … --resnet-classes …` y los tests siguen en verde.
- [ ] CA4: tabla comparativa en test interno y en campo: ResNet-nuestra vs DINO-pad (0.788) vs ResNet recibida (0.827).

## 5. Impacto
PRUEBA nueva `resnet50_75_objetivo` (solo `modelos/`). Los splits de `75_objetivo` se reutilizan sin regenerar (lista relativa versionada en la spec).
