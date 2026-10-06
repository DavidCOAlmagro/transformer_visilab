# 07 — Convenciones de código

## Idioma y nombres
- **Español** en identificadores, comentarios, docstrings, mensajes y commits
  (`calcular_pesos_clases`, `ruta_mejor_modelo`, `especies_filtradas`).
- Excepción: el script heredado `Inferir/infer_and_split_resnet_single_folder.py` está en **inglés**.
  Mantener el idioma del archivo que se toque.
- Funciones/variables `snake_case`; clases `PascalCase` (excepción histórica: `center_loss`).
- Constantes globales en MAYÚSCULAS dentro de `VARIABLES_GLOBALES` (excepción: `num_epocas`).

## Estilo
- Type hints en firmas (`list[str]`, `dict[str, int]`, `X | None`); `from __future__ import annotations` en módulos nuevos.
- Docstrings explicativos de **por qué**, con tono didáctico (el autor aprende con el código; ver `notas.md`).
- `pathlib.Path` para rutas; `encoding="utf-8"` siempre al abrir ficheros.
- `torch.load(..., weights_only=True)` siempre.
- Errores explícitos (`ValueError`/`FileNotFoundError` con mensaje en español) antes que fallos silenciosos.
- `@torch.inference_mode()` / `@torch.no_grad()` en evaluación.

## Invariantes que NO se deben romper
1. Orden de clases = `sorted(especies)`; índices de logits dependen de ello.
2. Preprocesado idéntico en embeddings de entrenamiento e inferencia (`preparar_para_dinov2` + L2 norm).
3. Augmentation solo en train.
4. Test no participa en ninguna decisión (selección de época, temperatura, umbrales → val).
5. Cambiar especies ⇒ nueva PRUEBA; nunca sobrescribir silenciosamente un experimento.
6. No versionar pesos (`*.pt`, `*.pth`, `*.ckpt`) ni `data/`.
7. Semilla 42 y `random_state=42` para reproducibilidad.
8. El orden de `embeddings_test.pt` = orden de `test.txt` (lo usa `errores.py`).

## Git
- Rama `main`, remoto `origin` → `DavidCOAlmagro/transformer_visilab`.
- Mensajes de commit cortos en español, imperativo (“Corregir…”, “Mejora…”, “Elimina…”).
- Commit/push solo cuando el usuario lo pida.

## Tests
- `unittest` estándar en `tests/`, importando como `src.<modulo>` desde la raíz.
- Nuevo código con lógica pura (sin GPU/datos) ⇒ añadir test. Usar `tempfile` para ficheros.
