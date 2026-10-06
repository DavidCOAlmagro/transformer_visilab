# Specs — índice

Una carpeta por cambio: `NNN-nombre-corto/` con `spec.md`, `plan.md`, `tareas.md`
(copiar de `_plantilla/`). Proceso en `../09_metodologia_SDD.md`.

| ID | Nombre | Estado | Fecha | Resumen |
|---|---|---|---|---|
| 001 | evaluacion-campo | Completada | 2026-10-06 | `src/evaluar_campo.py`: métrica de campo oficial; excluir `_fp` sube DINO a 0.741 y ResNet a 0.827 |

## Candidatas priorizadas (ver `../13_diagnostico_dinov2_campo.md` §4)
1. **Reentrenar 75_objetivo con pad-square** (resuelve B4) + comparar en test interno y en campo.
2. **Test de campo oficial**: conjunto fijo de imágenes con ground truth + script de evaluación por imagen (suma de confianza).
3. **Incorporar recortes YOLO de campo** (ground truth) al train, sin fuga por imagen/muestra; nuevas especies como PRUEBA nueva.
4. Aumentación realista / normalización de color homogénea.
5. Fine-tuning parcial de DINOv2 (LoRA o últimos bloques).
6. Tratamiento de clases `_fp` (jerarquía / nivel género).
7. Corregir B1/B2 en `src/inferencia.py` y unificar las dos inferencias; quitar hardcode de 77 clases.
