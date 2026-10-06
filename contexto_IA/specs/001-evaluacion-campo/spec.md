# SPEC 001 — Evaluador de campo reproducible (por imagen)

- **Estado:** Completada
- **Fecha:** 2026-10-06
- **Relacionado:** `13_diagnostico_dinov2_campo.md` §4 puntos 7 y 8; decisión D-015.

## 1. Contexto y problema
El rendimiento real se mide hoy con un Excel hecho a mano (`C:\VISILAB\cruce_ground_truth.xlsx`).
No hay un script versionado que lo reproduzca, así que no se pueden comparar modelos nuevos de forma justa.
Además, `_fp` significa **posición pleural** (vista lateral, a nivel de género): no es una especie.
Aun así, sus predicciones votan en la agregación por imagen.

## 2. Objetivo
Un script `src/evaluar_campo.py` que, a partir de los Excel de inferencia por recorte y de las etiquetas por imagen,
calcule la accuracy por imagen de cada modelo con varias reglas de agregación y exclusión, y del ensamble.

## 3. Alcance
- **Incluye:** lectura de los Excel de `Inferir/infer_and_split_resnet_single_folder.py`; mapeo de erratas de las clases ResNet;
  reglas `voto`, `mayor`, `suma_conf`; exclusión de {Debris, Fragments} y, opcionalmente, de `*_fp`; ensamble DINO+ResNet;
  accuracy a nivel de género; informe por especie; tests unitarios de la lógica pura.
- **No incluye:** volver a lanzar la inferencia, construir el ground truth ni reentrenar.

## 4. Criterios de aceptación
- [x] CA1: con la exclusión {Debris, Fragments} reproduce el cruce manual: voto 0.6406/0.7151, mayor 0.4752/0.5211,
      suma_conf 0.6746/0.7603 (±0.001), sobre 8 234 imágenes evaluables.
- [x] CA2: informa el efecto de excluir `_fp` y del ensamble.
- [x] CA3: tests unitarios en verde.

## 6. Impacto en experimentos
Ninguno (solo evaluación). Será la **métrica de campo oficial** para las specs siguientes.
