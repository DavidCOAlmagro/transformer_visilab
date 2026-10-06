# contexto_IA — Índice

Carpeta de contexto para asistentes IA (y humanos). Objetivo: **no tener que releer
todo el repo** en cada sesión. Leer solo el documento necesario para la tarea.

> Última actualización: 2026-10-06 (commit base `fe127a3`).
> Si el código cambia y un documento queda desactualizado, **actualizar el documento
> en el mismo cambio** (regla SDD, ver `09_metodologia_SDD.md`).

## Orden de lectura según la tarea

| Tarea | Leer |
|---|---|
| **Inicio de CUALQUIER sesión** | **`12_estado_actual.md`** (qué se está haciendo ahora) |
| Primera vez / visión general | `01_vision_proyecto.md` → `02_arquitectura.md` |
| Mejorar DINOv2 / rendimiento en campo / ground truth | `13_diagnostico_dinov2_campo.md`, `12` |
| Tocar entrenamiento, pérdidas, dataloaders | `02`, `03_modulos_src.md`, `06_configuracion.md` |
| Tocar inferencia YOLO + DINOv2/ResNet | `05_inferencia.md` (script oficial: `Inferir/infer_and_split_resnet_single_folder.py`), `preprocesado.py` |
| Datos, splits, nuevas especies, experimentos | `04_datos_y_experimentos.md`, `06` |
| Cambiar hiperparámetros o flags CLI | `06_configuracion.md` |
| Escribir código nuevo | `07_convenciones.md` |
| Arreglar bugs / refactor | `08_deuda_tecnica.md` |
| Empezar cualquier feature/cambio | `09_metodologia_SDD.md` + `specs/_plantilla/` |
| Por qué se decidió X | `10_decisiones.md` |
| Términos del dominio | `11_glosario.md` |

## Mapa de documentos

- `01_vision_proyecto.md` — qué es, objetivo, estado actual y métricas.
- `02_arquitectura.md` — pipeline extremo a extremo y flujo de datos/artefactos.
- `03_modulos_src.md` — referencia por archivo: responsabilidades y funciones clave.
- `04_datos_y_experimentos.md` — estructura de `data/`, splits, experimentos y resultados.
- `05_inferencia.md` — los dos scripts de inferencia y sus diferencias.
- `06_configuracion.md` — `constantes.py`, hiperparámetros, flags CLI, entorno.
- `07_convenciones.md` — estilo de código, idioma, invariantes a respetar.
- `08_deuda_tecnica.md` — bugs detectados, inconsistencias y deuda pendiente.
- `09_metodologia_SDD.md` — cómo trabajamos (Spec-Driven Development).
- `10_decisiones.md` — registro de decisiones (ADR ligeros).
- `11_glosario.md` — vocabulario del dominio y del proyecto.
- `12_estado_actual.md` — **documento vivo**: foco actual, resultados de campo, avisos, próximos pasos.
- `13_diagnostico_dinov2_campo.md` — por qué DINOv2 < ResNet50 en campo y plan de mejora.
- `specs/` — una carpeta por cambio: `spec.md`, `plan.md`, `tareas.md`.

## Resumen en 5 líneas

1. Clasificador de **especies de diatomeas** (microscopía) — VISILAB, autor David Calzado Olmo.
2. **DINOv2-base congelado** → embedding 768 (CLS, L2-normalizado) → **MLP 768→512→256** → 77 especies.
3. Embeddings precalculados en `.pt`; solo se entrena el MLP (rápido).
4. Inferencia (`Inferir/infer_and_split_resnet_single_folder.py`, modelo por defecto `75_objetivo_pad`): **YOLO** detecta ROIs → se clasifican con DINOv2-MLP y/o **ResNet50 (78 clases)** → Excel.
5. Experimento vigente: `75_objetivo` (77 clases), macro-F1 test ≈ 0.877, acc ≈ 0.89, top-3 ≈ 0.977.
6. **En campo** (12 857 imgs Aqualitas/DBO5): acc por imagen DINOv2 0.675 vs **ResNet50 0.760** → foco actual: ground truth + mejorar DINOv2.
