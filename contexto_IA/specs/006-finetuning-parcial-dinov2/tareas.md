# TAREAS 006

- [x] T1 — Lista de splits relativa (`splits_75_objetivo_relativos.txt.gz`: 45 985 / 9 854 / 9 854).
- [x] T2 — `scripts/finetune_dinov2.py` (reanudable, freno térmico, comprueba imágenes, calibra temperatura, guarda formato de inferencia).
- [x] T3 — `Inferir/`: `backbone_state()` + carga del backbone ajustado en `DinoClassifier`; `tests/test_inferir_checkpoint.py`.
- [x] T4 — Humo en Windows (RTX 3050, 256 imágenes/split, 1 época, batch 8): sin errores, bf16, 28.9 M parámetros entrenables.
      La inferencia carga el backbone ajustado (= checkpoint; bloque 0 congelado igual al original) y predice. CA1 ✔ CA3 ✔.
- [ ] T5 — Entrenamiento completo en Ubuntu (8 GB).
- [ ] T6 — Test interno y campo (CA4); actualizar contexto.
