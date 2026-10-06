# SPEC 004 — Limpieza de la inferencia y modelo por defecto `75_objetivo_pad`

- **Estado:** Completada (2026-10-06)
- **Tipo:** pequeño (spec, plan y tareas en un solo archivo)
- **Relacionado:** deuda B1, B2 y B4; `05_inferencia.md`; spec 003.

## Problema
- `src/inferencia.py` no se usa, pero el README y el manual la presentan como la inferencia principal; además tiene bugs (B1, B2).
- La inferencia real (`Inferir/infer_and_split_resnet_single_folder.py`) carga por defecto `75_objetivo`, entrenado con
  center-crop, mientras que el script aplica pad-square (B4).

## Cambios
1. Eliminar `src/inferencia.py` (nadie lo importa; comprobado con grep).
2. `DEFAULT_DINO_WEIGHTS` → `modelos/75_objetivo_pad/modelo_75_objetivo_pad.pth`. Las clases salen de los
   `metadatos_modelo.json` de esa carpeta; `classes_77(dino).txt` coincide exactamente (comprobado).
3. `readme.md` y `manual_de_uso.txt`: documentar el script de `Inferir/` como la inferencia, añadir `evaluar_campo.py`
   y quitar la nota de `python3-tk` (no hay selector gráfico en el código).
4. Actualizar contexto_IA (02, 03, 05, 08, 12, specs/README).

## Criterios de aceptación
- [x] CA1: no quedan referencias a `src/inferencia.py` en el código ni en la documentación de uso.
- [x] CA2: `--help` del script oficial funciona y el valor por defecto de `--dino-weights` es `75_objetivo_pad`.
- [x] CA3: el `DinoClassifier` del script oficial, cargado con los pesos nuevos, reproduce la accuracy de test de la spec 003
      sobre una muestra de imágenes de test (comprueba que inferencia y entrenamiento son coherentes).
- [x] CA4: tests en verde.

## Fuera de alcance
Unificar tronco/cabeza del script con `ClasificadorDiatomeas`; quitar el hardcode de 77 clases; detección de desconocidas en el script.

## Resultado
- CA3, sobre la misma muestra de 2 000 imágenes de test: embeddings del entrenamiento 0.896 · script oficial fp32 0.896 · fp16 0.8955.
  La inferencia oficial reproduce el entrenamiento (fp16 cambia 1 de 2 000). La muestra es algo más difícil que el test completo (0.910).
