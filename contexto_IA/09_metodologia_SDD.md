# 09 — Metodología SDD (Spec-Driven Development)

Todo cambio no trivial pasa por **especificación → plan → tareas → implementación → verificación**.
La spec es la fuente de verdad; el código la implementa; `contexto_IA/` se mantiene al día.

## Flujo
```
0. Contexto    Leer 00_INDICE.md y solo los docs relevantes.
1. SPEC        specs/NNN-nombre/spec.md     QUÉ y POR QUÉ (sin cómo). Criterios de aceptación medibles.
               → el usuario la revisa y aprueba.  ⛔ no se escribe código antes.
2. PLAN        specs/NNN-nombre/plan.md     CÓMO: archivos afectados, diseño, riesgos, impacto en
               experimentos (¿recalcular embeddings? ¿nueva PRUEBA?), estrategia de validación.
               → aprobación del usuario.
3. TAREAS      specs/NNN-nombre/tareas.md   Lista atómica y ordenada, cada una verificable.
4. IMPLEMENTAR Una tarea cada vez, marcando [x]. Cambios mínimos, respetando 07_convenciones.md.
5. VERIFICAR   Tests (unittest) + comprobación de criterios de aceptación. Métricas si aplica.
6. CERRAR      Actualizar 12_estado_actual.md (siempre), docs de contexto_IA afectados, 08 (si resuelve deuda), 10 (si hubo
               decisión), estado de la spec → "Completada". Commit solo si el usuario lo pide.
```

## Tamaño del cambio
| Tipo | Proceso |
|---|---|
| Trivial (typo, renombre local, 1-2 líneas obvias) | Directo, sin spec. Mencionarlo al usuario. |
| Pequeño (bug acotado, 1 archivo) | `spec.md` breve con plan y tareas incluidos en el mismo archivo. |
| Medio/grande (feature, refactor, nuevo experimento) | Las 3 piezas completas. |

## Convenciones de specs
- Carpeta `contexto_IA/specs/NNN-nombre-corto/` (NNN correlativo: 001, 002…).
- Estados: `Borrador → Aprobada → En curso → Completada` (o `Descartada`).
- Índice de specs en `specs/README.md` (actualizar al crear/cerrar).
- Plantillas en `specs/_plantilla/`.

## Para experimentos de ML
La spec debe fijar **antes** de entrenar:
- Hipótesis y métrica objetivo (macro-F1 test, F1 por especie, % rechazo…), y baseline de referencia.
- PRUEBA nueva o existente, si se regeneran splits (normalmente **no**) y si se recalculan embeddings.
- Criterio de éxito/fracaso. El resultado (aunque sea negativo) se registra en `10_decisiones.md`.

## Reglas para la IA
- No asumir: si la spec es ambigua, preguntar antes de implementar.
- Leer el código real antes de modificar (los docs pueden estar desfasados: verificar).
- No tocar `data/`, pesos ni splits existentes sin que la spec lo diga explícitamente.
- Mantener sincronizados `readme.md`/`manual_de_uso.txt` cuando cambie el uso.
