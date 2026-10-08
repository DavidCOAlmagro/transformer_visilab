# Specs — índice

Una carpeta por cambio: `NNN-nombre-corto/` con `spec.md`, `plan.md`, `tareas.md`
(copiar de `_plantilla/`). Proceso en `../09_metodologia_SDD.md`.

| ID | Nombre | Estado | Fecha | Resumen |
|---|---|---|---|---|
| 006 | finetuning-parcial-dinov2 | En curso | 2026-10-07 | Entrenar los 4 últimos bloques de DINOv2 + MLP con aumentación online (`75_objetivo_ft`) |
| 005 | resnet-mismos-datos | Pospuesta | 2026-10-07 | ResNet50 con nuestros splits: ¿receta o datos? |
| 004 | limpieza-inferencia | Completada | 2026-10-06 | Elimina `src/inferencia.py`; `75_objetivo_pad` por defecto en `Inferir/`; README/manual actualizados |
| 003 | reentrenar-pad-square | Completada | 2026-10-07 | `75_objetivo_pad`: macro-F1 test 0.893 (antes 0.877); **campo 0.788** (antes 0.741; ResNet 0.827) |
| 002 | experimento-pad-square | Completada | 2026-10-06 | pad-square +3.0 macro-F1 en test interno (0.887 vs 0.857), mayor ganancia en alargadas |
| 001 | evaluacion-campo | Completada | 2026-10-06 | `src/evaluar_campo.py`: métrica de campo oficial; excluir `_fp` sube DINO a 0.741 y ResNet a 0.827 |

## Candidatas priorizadas (ver `../13_diagnostico_dinov2_campo.md` §4)
1. ~~Reentrenar con pad-square y medir en campo~~ (spec 003 ✔: 0.741 → 0.788). Investigar las regresiones (Stephanodiscus, A. eutrophilum, Encyonopsis_minuta).
2. **Test de campo oficial**: conjunto fijo de imágenes con ground truth + script de evaluación por imagen (suma de confianza).
3. **Incorporar recortes YOLO de campo** (ground truth) al train, sin fuga por imagen/muestra; nuevas especies como PRUEBA nueva.
4. Aumentación realista / normalización de color homogénea.
5. Fine-tuning parcial de DINOv2 (LoRA o últimos bloques).
6. Tratamiento de clases `_fp` (jerarquía / nivel género).
7. ~~Eliminar `src/inferencia.py`~~ (spec 004 ✔). Pendiente: quitar el hardcode de 77 clases (modelo.py e `Inferir/`).
