# SPEC 008 — Nuevo diseño de las imágenes anotadas de la inferencia

- **Estado:** Completada (2026-10-08)
- **Tipo:** pequeño (spec, plan y tareas en un solo archivo)
- **Relacionado:** `05_inferencia.md`; rama creada desde `sdd/006-finetuning-dinov2`.

## Problema
En `bbox/dinov2/` y `bbox/resnet/` el recuadro salía naranja o azul y la etiqueta era diminuta (`arial.ttf` no existe en Ubuntu →
fuente bitmap de PIL de unos 10 px, con tamaño fijo de 16) e incluía el porcentaje y el prefijo `REVISION:`.

## Cambio (`Inferir/infer_and_split_resnet_single_folder.py`: `_font`, `_annotation_label`, `_annotate_image`)
- Recuadro y fondo de la etiqueta en **verde** (`ANNOTATION_COLOR = (0, 200, 0)`) para los dos modelos.
- Etiqueta = **solo la especie**, con espacios (`Gomphonema rhombicum`), negro sobre verde. Sin porcentaje ni `REVISION`
  (siguen en el Excel).
- Tamaño proporcional: fuente `max(24, lado_menor // 30)` px y recuadro `max(4, lado_menor // 200)` px.
- Fuente escalable: DejaVu Sans Bold (Ubuntu) → `arialbd.ttf` / `arial.ttf` (Windows) → `ImageFont.load_default(size=...)`.
- Posición de la etiqueta: encima de la caja; si no cabe o pisa otra etiqueta, debajo; si no, dentro. Sin salirse por la derecha.

## Criterios de aceptación
- [x] CA1: `tests/test_anotacion.py` (etiqueta sin `%` ni `REVISION`, recuadro verde, fuente escalable, etiquetas cercanas sin solaparse).
- [x] CA2: imagen de muestra generada en Windows y revisada (2 cajas cercanas, una de baja confianza).
- [ ] CA3: comprobado en Ubuntu en la próxima inferencia de campo.
