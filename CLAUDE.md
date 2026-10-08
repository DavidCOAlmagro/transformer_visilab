# CLAUDE.md

Clasificador de especies de diatomeas: DINOv2-base congelado → MLP (77 clases) + inferencia YOLO/ResNet50.

**Antes de trabajar, lee `contexto_IA/00_INDICE.md` y `contexto_IA/12_estado_actual.md`** (estado vivo; actualízalo al terminar cada sesión/spec) y solo los documentos que la tarea necesite.
No releas el repo entero si el contexto ya responde a la pregunta (pero verifica el código antes de modificarlo).

## Metodología: SDD
Todo cambio no trivial: spec → plan → tareas → implementar → verificar → actualizar `contexto_IA/`.
Detalle en `contexto_IA/09_metodologia_SDD.md`; specs en `contexto_IA/specs/`.
No escribir código de una feature sin spec aprobada por el usuario.

## Reglas rápidas
- Código, comentarios y commits en **español** (salvo `Inferir/infer_and_split_resnet_single_folder.py`, en inglés).
- Orden de clases = `sorted(especies)`. Preprocesado idéntico en train e inferencia.
- No tocar `data/`, splits, embeddings ni pesos sin que la spec lo indique.
- Tests: `python -m unittest tests.test_preprocesado tests.test_evaluar_campo tests.test_inferir_checkpoint tests.test_anotacion` desde la raíz (Python del sistema; la `.venv` del repo está vacía, sin dependencias).
- Al cambiar algo documentado, actualizar el `.md` correspondiente de `contexto_IA/` en el mismo cambio.
