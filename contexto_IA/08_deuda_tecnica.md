# 08 — Deuda técnica, bugs e inconsistencias

Detectados en la revisión del 2026-10-06. Marcar `[x]` y referenciar la spec al resolver.

## Bugs probables (prioridad alta)
> B1 y B2 **resueltos por eliminación** de `src/inferencia.py` (spec 004).
- [x] **B1 — Inferencia sin L2-norm** (`src/inferencia.py`, `ClasificadorDino.predecir`):
  usa `pooler_output.float()` sin normalizar, pero el MLP se entrenó con embeddings de norma 1
  (`embeddings.get_embedding`). El script heredado sí normaliza. Predicciones/confianzas distintas entre scripts.
- [x] **B2 — Coordenadas mal escritas** (`src/inferencia.py`, `inferir`): `"x2": y2` y falta `y2`.
- [ ] **B3 — Centroides sin entrenar**: con `LAMBDA_CENTER_LOSS=0`, `center_loss.centros` (init `randn`)
  no reciben gradiente útil (solo weight decay) y aun así `calibrar_pesos` los usa para
  `umbral_distancia`. El rechazo de desconocidas se basa en centros casi aleatorios → explica el
  pobre 16 % de rechazo. Alternativa: calcular centros como **media de embeddings del tronco por clase en train**.
- [x] **B4 — Pesos vs preprocesado** (resuelto: spec 003 entrena `75_objetivo_pad` y la spec 004 lo pone por defecto en la inferencia; `75_objetivo` solo es coherente con center-crop): embeddings y pesos de 75_objetivo (2026-09-24) se calcularon con
  resize + **center crop 224** (sin pad). Desde `fe127a3` ambas inferencias aplican pad-square → desajuste
  train/inferencia hasta reentrenar. Los Excel de campo del 2026-10-06 son anteriores y NO lo sufren.
- [ ] **B5 — Clases ResNet con erratas**: `classes_78(resnet).txt` tiene `Denticula_tenius`, `Nitzschia_dessertorum`
  (y `Planothidium_fp` extra) frente a `Denticula_tenuis`, `Nitzschia_desertorum` en DINO → mapear al cruzar.

## Inconsistencias
- [ ] `modelo.py` exige **exactamente 77 clases** → `20_especies` y cualquier PRUEBA nueva no cargan.
  Debería validar contra metadatos, no contra un número fijo (77 también hardcodeado en inferencia).
- [x] README/manual dicen `mejor_modelo.pth` → corregido a `modelo_<PRUEBA>.pth` en el README (spec 004).
- [ ] `reporte_test.txt` sin sufijo, mientras matriz/métricas/confusiones llevan `_<PRUEBA>`.
- [ ] `obtener_especies_activas()`: `return ESPECIES_FILTRADAS` final inalcanzable; docstring dice fallback que no existe.
- [ ] `codificacion`: `sorted(sorted(...))` redundante.
- [ ] `main.py` escribe `metadatos_modelo.json` (versión `dinov2-77-v2`) aunque se reentrene con los mismos datos, perdiendo claves extra del existente.
- [ ] `EXTENSIONES_VALIDAS` sin `.webp`, pero auditoría e inferencia heredada sí lo aceptan.
- [ ] `notas.md` desactualizado (PACIENCIA 5, “capa lineal”, ESPECIES_MINORITARIAS que ya no existen).
- [ ] `Seleccion_5_especies_por_especie/` dentro de `imagenes_visilab(raw)` se recorre como grupo → posibles duplicados/fuga entre splits.
- [ ] Splits con rutas absolutas → no portables entre máquinas.

## Deuda / limpieza
- [ ] `accuracy_genero_test` se sigue calculando y guardando aunque la cabeza de género no se entrena (PESO_GENERO=0) → da ~8 % (spec 003). Ocultar o calcularla desde las especies predichas.
- [ ] `errores.py`, `confusiones.py` y `evaluar_desconocidas.py` no aceptan `--prueba` (solo `PRUEBA` de constantes.py).
- [x] `src/inferencia.py` eliminado y README/manual actualizados (spec 004).
- [ ] `get_embedding` procesa imágenes de una en una (lento con ~46k × 2+ imágenes); batchear.
- [ ] Ficheros basura en el repo: `git` (vacío), `Inferir/*.py.backup`; `*.pt` duplicado en `.gitignore`.
- [ ] `auxiliar/` con rutas absolutas Linux hardcodeadas.
- [ ] Tests mínimos: sin tests de `codificacion`, pesos de clase, `lr_lambda`, carga de modelo, `_top3`.
- [ ] `requirements.txt`: pandas, xlsxwriter, ultralytics sin versión; falta `openpyxl` si se lee Excel.
- [ ] `center_loss` se crea y se suma (×0) aunque esté desactivado: coste inútil y centros en el optimizador.
