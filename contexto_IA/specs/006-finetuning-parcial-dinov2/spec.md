# SPEC 006 — Fine-tuning parcial de DINOv2

- **Estado:** En curso (aprobada 2026-10-07; código listo, falta la corrida en Ubuntu)
- **Fecha:** 2026-10-07
- **Relacionado:** `13_diagnostico_dinov2_campo.md` §1b y §4 punto 4; spec 003 (DINO-pad: test 0.893, campo 0.788); spec 005 (pospuesta).

## 1. Problema
DINOv2 está congelado: sus rasgos no se adaptan a las diatomeas ni al microscopio de campo. La ResNet50 (fine-tuning completo
y aumentación online) llega a 0.827 en campo frente a 0.788 de DINO-pad.

## 2. Objetivo
Entrenar los **últimos bloques de DINOv2** junto con el MLP, con aumentación **online** (una variante nueva por época), y medir si mejora
en el test interno y, sobre todo, en campo.

## 3. Alcance
- **Incluye:** script de entrenamiento autocontenido para Ubuntu (8 GB), reanudable y con freno térmico; prueba de humo en Windows con pocas
  imágenes; carga de un backbone ajustado en `Inferir/infer_and_split_resnet_single_folder.py`; evaluación en test interno y en campo.
- **No incluye:** cambiar especies o splits; tocar `75_objetivo` ni `75_objetivo_pad`; cambiar el modelo por defecto (se decide con el resultado).

## 4. Criterios de aceptación
- [x] CA1: prueba de humo en Windows (subconjunto pequeño, 1 época) sin errores, y checkpoint cargable por el script de inferencia.
- [ ] CA2: en Ubuntu, `modelos/75_objetivo_ft/` con `modelo_75_objetivo_ft.pth` (backbone + MLP), `metadatos_modelo.json`, `metricas.json` y curvas.
- [x] CA3: el script de inferencia carga el backbone ajustado y, con un checkpoint antiguo (sin backbone), se comporta igual que antes. Tests en verde.
- [ ] CA4: tabla en test interno (vs 0.893) y en campo (vs DINO-pad 0.788 y ResNet 0.827). **Éxito en campo: > 0.788**; objetivo: ≥ 0.827.

## 5. Impacto
PRUEBA nueva `75_objetivo_ft`, con los mismos splits (lista relativa versionada en la spec).
