# SPEC 003 — Reentrenar el modelo DINOv2 de 77 clases con pad-square

- **Estado:** Completada en test interno (2026-10-06). CA4 (campo) pendiente en el equipo Ubuntu.
- **Fecha:** 2026-10-06
- **Relacionado:** deuda B4 (crítica), D-011, spec 002 (experimento), `13_diagnostico_dinov2_campo.md` E2.

## 1. Contexto y problema
El modelo oficial `75_objetivo` (2026-09-24) se entrenó con resize + center crop 224, pero el código actual
(desde `fe127a3`) aplica pad-square en inferencia: hay un desajuste entre entrenamiento e inferencia (B4).
La spec 002 demostró que pad-square, incluso sin augmentation, mejora el test interno (macro-F1 0.887 frente a 0.857)
y supera al modelo oficial (0.877), sobre todo en diatomeas alargadas.

## 2. Objetivo
Tener un modelo de 77 clases entrenado con el pipeline completo (pad-square + augmentation + calibración)
que sea coherente con la inferencia actual y que mejore al modelo oficial.

## 3. Alcance
- **Incluye:** nueva PRUEBA `75_objetivo_pad` (mismas 77 especies y **mismos splits**, copiados); embeddings pad-square
  con la augmentation actual; entrenamiento con `main.py`; calibración; evaluación de test, errores y confusiones. **Sin `unknown`** (decisión del usuario: ahorra ~30 min de GPU; `evaluar_desconocidas.py` queda pendiente).
- **No incluye:** cambiar especies, hiperparámetros ni augmentation; tocar `75_objetivo` (se conserva como baseline);
  cambiar el modelo por defecto de la inferencia (se decide tras ver el resultado de campo).

## 4. Criterios de aceptación
- [x] CA1: `modelos/75_objetivo_pad/` contiene pesos calibrados, `metadatos_modelo.json` (preprocesado `dinov2-pad-square-v1`) y reportes.
- [x] CA2: macro-F1 de test ≥ 0.877 (modelo oficial actual). Objetivo: ≥ 0.887 (pad sin augmentation, spec 002).
- [x] CA3: `75_objetivo` queda intacto.
- [ ] CA4: si están disponibles las imágenes de campo, accuracy por imagen con `evaluar_campo.py` (suma_conf sin `_fp`) > 0.741.

## 6. Impacto en experimentos
- PRUEBA nueva `75_objetivo_pad`. Splits **no** se regeneran (copia exacta de `data/splits/75_objetivo/`).
- Se recalculan embeddings (pad-square). Se reentrena.

## 7. Preguntas abiertas
- Imágenes de campo: están en el equipo Ubuntu → CA4 se mide allí más adelante.
