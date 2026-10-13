# Auditoría del código (2026-10-09, rama `refactor/limpieza`)

Ámbito: todos los ficheros versionados (`src/` 11, `Inferir/` 1 script, `tareas.py`, `Makefile`, `tests/` 6, docs).
Línea base antes de tocar nada: `python tareas.py test` → 17 OK; inferencia de referencia con `75_objetivo_ft` y ResNet50
sobre 8 imágenes de validación + `main()` completo con YOLO simulado (ver INFORME.md).

## 1. Crítico
| # | Dónde | Problema | Acción |
|---|---|---|---|
| C1 | `src/entrenar.py` (`--salida`) | La salida por defecto es `modelos/75_objetivo_ft`, la carpeta de los **pesos de referencia**. Como allí hay un `ultimo.pth`, `python tareas.py entrenar` sin `--salida` "reanudaría" la época 15, saltaría al final, **volvería a dividir la cabeza por la temperatura** (doble calibración) y **sobrescribiría `modelo_75_objetivo_ft.pth`**. | Salida por defecto con nombre propio y fecha (`modelos/<prueba>_<fecha>`) y negarse a escribir en una carpeta que ya tenga un modelo final. |

## 2. Alto
| # | Dónde | Problema | Acción |
|---|---|---|---|
| A1 | `Inferir/…py` `main()`, `tareas.py inferir`, `Makefile` | Tres sitios deciden dónde van los resultados de inferencia (carpeta hermana de la entrada `<entrada>_resultados`, `../resultados_<fecha>` o `--output-dir`). Resultados repartidos por el disco. | Una sola función: `Inferir/resultados_inferencia/<modelo>/<prueba>/`, con prueba = fecha y hora si no se da nombre; nunca se sobrescribe. |
| A2 | `tareas.py limpiar` | Incluye `Inferir/imagenes_inferencia/crops` y `bbox` (resultados antiguos dentro de la carpeta de entrada) y borraría los pesos de `modelos/75_objetivo*` distintos de ft con `--si`. Además busca `__pycache__` en **todo** el repo, incluido `data/` (millones de ficheros: muy lento). | Quitar los resultados antiguos de la lista; buscar cachés solo en carpetas de código. |
| A3 | `Inferir/…py` `DinoClassifier` | Descarga o lee `facebook/dinov2-base` del Hub directamente (aviso de "unauthenticated requests" en cada ejecución; falla sin red), mientras que el entrenamiento usa `resolver_modelo_dinov2()` (caché local o `DINOV2_MODEL_PATH`). Dos formas de cargar lo mismo. | Usar `resolver_modelo_dinov2()` también en inferencia (mismos pesos, sin red). Verificar que las salidas no cambian. |

## 3. Medio
| # | Dónde | Problema | Acción |
|---|---|---|---|
| M1 | `Inferir/…py` | Código muerto: `extract_original_class()` (con un `import re` repetido dentro), `_PIL_IMAGE_OPEN`, parámetro `model_name` de `_annotate_image()` (el color ya es único), opción `--reinhard-reference` (solo lanza `NotImplementedError`). | Eliminar. |
| M2 | `Inferir/…py` `load_torch_checkpoint()` | Rama de compatibilidad para versiones de torch sin `weights_only` (torch 2.7 lo tiene siempre). | Simplificar a una llamada. |
| M3 | `Inferir/…py` `DinoClassifier.__init__` | ~50 líneas de validación manual clave a clave (forma de la cabeza, claves requeridas, incompatibles) que `load_state_dict(strict=True)` ya hace, con 77 escrito a mano; `read_metadata_classes()` y `resolve_dino_classes()` también exigen 77, y ResNet 78. | Construir la cabeza con `len(classes)` y validar con `strict=True` (mensaje claro); mismos errores para checkpoints malos. |
| M4 | `src/constantes.py` | Claves sin uso: `DEVICE` (obliga a importar torch y consultar CUDA al importar `constantes`), `HF_TOKEN`, `RUTA_MODELOS`. Formato con comentarios y separadores rotos. | Quitar las claves sin uso y el import de torch/os. |
| M5 | `src/entrenar.py` | f-string con comillas dobles anidadas (`{tipo_amp or "fp32"}`): solo válido en Python ≥ 3.12. El docstring habla de `main.py` (ya no existe). | Corregir. |

## 4. Bajo
| # | Dónde | Problema | Acción |
|---|---|---|---|
| B1 | `src/clasificador.py` | Docstring de `forward` menciona center loss (eliminado). `cabeza_genero`, `centros` y `umbral_distancia` ya no se usan, **pero están en el checkpoint de `75_objetivo_ft`**: quitarlos rompería `--pesos-mlp` con ese modelo. | Solo corregir el docstring; mantener las capas (anotado en INFORME). |
| B2 | `src/dividir_campo.py` `dividir()` | `etiquetas.index.isin(etiquetas.index[indices_test])` es una forma rebuscada de una máscara. | Simplificar sin cambiar el resultado (las listas deben salir idénticas). |
| B3 | `src/preprocesado.py` | Rama `"gray"` sin uso; `if not media` inalcanzable. | Dejar (es la pieza que garantiza el mismo preprocesado en train e inferencia; ganancia mínima, riesgo no nulo). |
| B4 | `src/auditoria_dataset.py` | No normaliza nombres de carpeta: contaría `Fistulifera saprophila` como clase aparte y la propondría como "clase adicional". | **No se cambia** (es un cambio de comportamiento): a decidir por el usuario. |
| B5 | Docs (`readme.md`, `manual_de_uso.txt`) | Describen las rutas de salida antiguas. | Actualizar al terminar. |

## 5. Fuera del código (no se toca; para decidir)
- `Inferir/imagenes_inferencia.zip` (2,7 GB) y `Inferir/Nueva carpeta.rar` (137 MB): sin seguimiento, no se tocan.
- `modelos/75_objetivo/` y `modelos/75_objetivo_pad/`: solo quedan sus `.pth` (sin seguimiento).
- `notas.md`: cuaderno desactualizado (pendiente de decisión desde la spec 012).
- Las columnas `<modelo>_especie_predicha` y `<modelo>_especie_mas_parecida` del Excel son siempre idénticas: se mantienen porque cambiar el formato del Excel afectaría a quien lo use.
