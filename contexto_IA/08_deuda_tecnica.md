# 08 — Deuda técnica, bugs e inconsistencias

Detectados en la revisión del 2026-10-06. Marcar `[x]` y referenciar la spec al resolver.

## Bugs probables (prioridad alta)
- [ ] **B1 — Inferencia sin L2-norm** (`src/inferencia.py`, `ClasificadorDino.predecir`):
  usa `pooler_output.float()` sin normalizar, pero el MLP se entrenó con embeddings de norma 1
  (`embeddings.get_embedding`). El script heredado sí normaliza. Predicciones/confianzas distintas entre scripts.
- [ ] **B2 — Coordenadas mal escritas** (`src/inferencia.py`, `inferir`): `"x2": y2` y falta `y2`.
- [ ] **B3 — Centroides sin entrenar**: con `LAMBDA_CENTER_LOSS=0`, `center_loss.centros` (init `randn`)
  no reciben gradiente útil (solo weight decay) y aun así `calibrar_pesos` los usa para
  `umbral_distancia`. El rechazo de desconocidas se basa en centros casi aleatorios → explica el
  pobre 16 % de rechazo. Alternativa: calcular centros como **media de embeddings del tronco por clase en train**.
- [ ] **B4 — Pesos vs preprocesado** (CRÍTICO): embeddings y pesos de 75_objetivo (2026-09-24) se calcularon con
  resize + **center crop 224** (sin pad). Desde `fe127a3` ambas inferencias aplican pad-square → desajuste
  train/inferencia hasta reentrenar. Los Excel de campo del 2026-10-06 son anteriores y NO lo sufren.
- [ ] **B5 — Clases ResNet con erratas**: `classes_78(resnet).txt` tiene `Denticula_tenius`, `Nitzschia_dessertorum`
  (y `Planothidium_fp` extra) frente a `Denticula_tenuis`, `Nitzschia_desertorum` en DINO → mapear al cruzar.

## Inconsistencias
- [ ] `modelo.py` exige **exactamente 77 clases** → `20_especies` y cualquier PRUEBA nueva no cargan.
  Debería validar contra metadatos, no contra un número fijo (77 también hardcodeado en inferencia).
- [ ] README/manual dicen `mejor_modelo.pth`; el código guarda `modelo_<PRUEBA>.pth`.
- [ ] `reporte_test.txt` sin sufijo, mientras matriz/métricas/confusiones llevan `_<PRUEBA>`.
- [ ] `obtener_especies_activas()`: `return ESPECIES_FILTRADAS` final inalcanzable; docstring dice fallback que no existe.
- [ ] `codificacion`: `sorted(sorted(...))` redundante.
- [ ] `main.py` escribe `metadatos_modelo.json` (versión `dinov2-77-v2`) aunque se reentrene con los mismos datos, perdiendo claves extra del existente.
- [ ] `EXTENSIONES_VALIDAS` sin `.webp`, pero auditoría e inferencia heredada sí lo aceptan.
- [ ] `notas.md` desactualizado (PACIENCIA 5, “capa lineal”, ESPECIES_MINORITARIAS que ya no existen).
- [ ] `Seleccion_5_especies_por_especie/` dentro de `imagenes_visilab(raw)` se recorre como grupo → posibles duplicados/fuga entre splits.
- [ ] Splits con rutas absolutas → no portables entre máquinas.

## Deuda / limpieza
- [ ] Dos implementaciones de inferencia (src vs Inferir) → unificar en un módulo compartido.
- [ ] `get_embedding` procesa imágenes de una en una (lento con ~46k × 2+ imágenes); batchear.
- [ ] Ficheros basura en el repo: `git` (vacío), `Inferir/*.py.backup`; `*.pt` duplicado en `.gitignore`.
- [ ] `auxiliar/` con rutas absolutas Linux hardcodeadas.
- [ ] Tests mínimos: sin tests de `codificacion`, pesos de clase, `lr_lambda`, carga de modelo, `_top3`.
- [ ] `requirements.txt`: pandas, xlsxwriter, ultralytics sin versión; falta `openpyxl` si se lee Excel.
- [ ] `center_loss` se crea y se suma (×0) aunque esté desactivado: coste inútil y centros en el optimizador.
