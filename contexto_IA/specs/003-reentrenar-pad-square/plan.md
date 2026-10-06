# PLAN 003 — Reentrenar con pad-square

## 1. Enfoque
Reutilizar el pipeline oficial (`main.py`) sin modificar código de `src/`. Para ahorrar ~40 min de GPU se reutilizan
los embeddings pad-square **no aumentados** de la spec 002, que se calcularon con exactamente la misma operación que
`embeddings.get_embedding(is_train=False)`: `preparar_para_dinov2` → processor → CLS → float32 → L2.
Antes de reutilizarlos se verifica la equivalencia con `get_embedding` sobre una muestra (diferencia máxima < 1e-3).

Alternativa descartada: `main.py --recalcular-embeddings s` desde cero (~2–2.5 h de GPU, porque `get_embedding` procesa las
imágenes de una en una).

## 2. Pasos y artefactos
| Paso | Resultado |
|---|---|
| Copiar `data/splits/75_objetivo/*.txt` → `data/splits/75_objetivo_pad/` | splits idénticos |
| Verificar equivalencia scratch ↔ `get_embedding` (muestra de 64 imágenes de val) | OK / abortar |
| val, test: copiar los `.pt` de la spec 002 | `data/embeddings_procesado/75_objetivo_pad/embeddings_{val,test}.pt` |
| train: originales de la spec 002 + aumentadas con `get_embedding(is_train=True)` (1 + copias extra por especie, igual que `calcular_embeddings`) | `embeddings_train.pt` (95 754 filas) |
| `python3 src/main.py --prueba 75_objetivo_pad --reentrenar s --regenerar-splits n --recalcular-embeddings n` | pesos calibrados + reportes |
| `errores.py`, `confusiones.py`, `evaluar_desconocidas.py` con `PRUEBA=75_objetivo_pad` | reportes |

Script de ensamblado: `contexto_IA/specs/003-reentrenar-pad-square/scripts/preparar_embeddings_pad.py`, con freno térmico (85 °C → pausa hasta 75 °C).

## 3. Riesgos e invariantes
- Orden de clases: `sorted`, sin cambios. Mismos splits → comparación justa con `75_objetivo` y con la spec 002.
- El orden de `embeddings_test.pt` = orden de `test.txt` (lo comprueba `errores.py`).
- GPU de portátil (RTX 3050 4 GB): ~50 k aumentadas + 45 k unknown ≈ 60–70 min. Freno térmico activo.
- `main.py` escribe `metadatos_modelo.json` con la versión `dinov2-77-v2` y el bloque `preprocesado` pad-square (correcto).

## 4. Validación
- CA2: comparar `resumen_entrenamiento.json` con 75_objetivo (0.877) y con la spec 002 (0.887).
- Tests: `python -m unittest tests.test_preprocesado tests.test_evaluar_campo`.

## 5. Documentación a actualizar
`12_estado_actual.md`, `04_datos_y_experimentos.md` (nueva PRUEBA), `08` (B4), `10` (D-016 si se adopta), `specs/README.md`.
