# TAREAS 006

- [x] T1 — Lista de splits relativa (`splits_75_objetivo_relativos.txt.gz`: 45 985 / 9 854 / 9 854).
- [x] T2 — `scripts/finetune_dinov2.py` (reanudable, freno térmico, comprueba imágenes, calibra temperatura, guarda formato de inferencia).
- [x] T3 — `Inferir/`: `backbone_state()` + carga del backbone ajustado en `DinoClassifier`; `tests/test_inferir_checkpoint.py`.
- [x] T4 — Humo en Windows (RTX 3050, 256 imágenes/split, 1 época, batch 8): sin errores, bf16, 28.9 M parámetros entrenables.
      La inferencia carga el backbone ajustado (= checkpoint; bloque 0 congelado igual al original) y predice. CA1 ✔ CA3 ✔.
- [x] T5 — Entrenamiento completo en Ubuntu (Quadro P4000, bf16 emulado, sin freno térmico, ~64 min/época, 15 épocas, ~16 h).
      Val macro-F1 por época: 0.877, 0.886, 0.888, …, 0.911 (E7), 0.914 (E10), **0.917 (E12, mejor)**, 0.917, 0.915, 0.917. Temperatura 0.865.
      **Test: accuracy 0.9371 · macro-F1 0.9179 · top-3 0.9896** (75_objetivo_pad: 0.910 / 0.893 / 0.985).
- [ ] T6 — Test interno y campo (CA4); actualizar contexto.
