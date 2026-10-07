---
name: spec-sdd
description: Abre o cierra una spec de la metodología SDD del proyecto (contexto_IA/specs/), con su rama, su PR y la actualización del contexto. Úsala al empezar cualquier cambio no trivial o al terminarlo.
---

# Spec SDD

Proceso completo en `contexto_IA/09_metodologia_SDD.md`. Tamaño: trivial → sin spec; pequeño → un único `spec.md` con plan y tareas.

## Abrir
1. Siguiente número libre: el mayor `NNN` de `contexto_IA/specs/` + 1.
2. Rama desde `main` actualizado: `git switch main && git pull && git switch -c sdd/NNN-nombre-corto`.
3. Copia `contexto_IA/specs/_plantilla/` a `contexto_IA/specs/NNN-nombre-corto/` y rellena `spec.md` (problema, objetivo,
   alcance, criterios de aceptación **medibles**, impacto en experimentos) y `plan.md` (archivos, diseño, riesgos, validación).
4. Añade la fila a `contexto_IA/specs/README.md` con estado `Borrador`.
5. **Pide aprobación al usuario antes de escribir código** (explica en lenguaje claro qué se hará y cuánto cuesta en tiempo y GPU).

## Durante
- Marca `[x]` en `tareas.md` según avances; estado `En curso` en la spec y en el README.
- Commits pequeños en español; push a la rama. Si algo se ejecuta en Ubuntu, usa la skill `trabajo-ubuntu`.

## Cerrar
1. Resultados y desviaciones en `tareas.md`; criterios de aceptación `[x]`; estado `Completada` en la spec y en el README.
2. Actualiza siempre `contexto_IA/12_estado_actual.md`; y, si aplica, `08_deuda_tecnica.md`, `10_decisiones.md` (nueva D-NNN)
   y los documentos de referencia afectados (`02`–`06`, `readme.md`, `manual_de_uso.txt`).
3. Tests: `python -m unittest tests.test_preprocesado tests.test_evaluar_campo tests.test_inferir_checkpoint`.
4. PR con `"C:/Program Files/GitHub CLI/gh.exe" pr create --repo DavidCOAlmagro/transformer_visilab --base main ...`
   (cuerpo: qué cambia, por qué, resultados, notas para el revisor; terminar con la línea de "Generated with Claude Code").
5. Si se aprendió algo sobre cómo trabaja el usuario, actualiza la memoria persistente.
