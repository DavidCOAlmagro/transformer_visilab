# 03 — Referencia de módulos (`src/`)

Imports **planos** (`from constantes import ...`): los scripts se ejecutan desde `src/`
(`python3 src/main.py` funciona porque Python añade el dir del script a `sys.path`).
Toda la configuración global vive en el dict mutable `VARIABLES_GLOBALES` (`constantes.py`).

| Archivo | Líneas | Rol |
|---|---|---|
| `constantes.py` | 157 | `PRUEBA`, rutas, `ESPECIES_FILTRADAS` (77), hiperparámetros, `DEVICE`. Ver `06`. |
| `main.py` | 275 | Orquestador entrenamiento/evaluación. CLI s/n. Escribe `metadatos_modelo.json`. |
| `preparar_datos.py` | 338 | Utilidades de datos, géneros, semilla, argparse, resumen JSON, especies activas. |
| `generar_leer_splits.py` | 104 | Split 70/15/15 estratificado + `unknown.txt` (especies fuera del experimento). |
| `preprocesado.py` | 74 | `pad_to_square` con color medio del borde. Versión `dinov2-pad-square-v1`. |
| `embeddings.py` | 201 | Carga DINOv2 (cache local/`DINOV2_MODEL_PATH`), augmentation, cálculo de embeddings. |
| `dataset.py` | 30 | `MyDataset(emb, et_especie, et_genero=None)`. |
| `dataloader.py` | 81 | DataLoaders + `WeightedRandomSampler`; pesos de clase `1/n^EXPONENTE`. |
| `clasificador.py` | 66 | `ClasificadorDiatomeas` (tronco + 2 cabezas + buffers de centroides). |
| `CenterLoss.py` | 27 | `center_loss(nn.Module)` con centros `nn.Parameter` (init `randn`). |
| `entrenamiento.py` | 176 | `entrenar_epoca`, `validacion`, `entrenar_modelo` (early stopping por macro-F1). |
| `calibracion_confianza.py` | 117 | Temperature scaling horneado + `calibrar_centroides` (percentil). |
| `modelo.py` | 73 | `cargar_modelo_entrenado()` seguro (`weights_only=True`), **exige 77 clases**. |
| `evaluar_test_metricas.py` | 264 | Métricas test, matriz, curvas, métricas por especie, top-3, acc. género. |
| `errores.py` | 133 | Lista errores de test con ruta y confianza → `errores_a_revisar_<PRUEBA>.txt`. |
| `confusiones.py` | 80 | Pares confundidos ≥ 5 veces → `confusiones_<PRUEBA>.txt`. |
| `evaluar_desconocidas.py` | 102 | Diagnóstico de rechazo con `unknown` + barrido de percentiles. |
| `inferencia.py` | 249 | Inferencia YOLO + DINOv2/ResNet50 → Excel. Ver `05`. |
| `exportar_config.py` | 46 | Vuelca hiperparámetros a `config_modelo_<PRUEBA>.json`. |
| `auditoria_dataset.py` | 87 | Auditoría CLI: distribución, duplicados SHA-256, clases candidatas. |
| `evaluar_campo.py` | ~190 | **Métrica de campo oficial**: accuracy por imagen desde los Excel de inferencia (reglas voto/mayor/suma_conf, ±excluir `_fp`, ensamble, nivel género). Ver `12`. |

## Funciones clave por módulo

### main.py
- `lr_lambda(epoca)` — warmup lineal `EPOCAS_WARMUP` + coseno hasta `num_epocas`.
- `resolver_si_no(flag, pregunta)` — usa flag CLI o pregunta interactiva.
- `main()` — aplica `--prueba`, recalcula `RUTA_SPLITS/RUTA_EMBEDDINGS`, fija semilla 42,
  entrena (si procede), calibra, evalúa test, guarda resumen.
- `preparar_embeddings_splits()` — calcula los 4 `.pt` (train con augmentation).

### preparar_datos.py
- `get_datos(split)` → dict del `.pt`. `codificacion(datos)` → (emb, etiquetas long, `{especie: idx}`), filtra especies no activas.
- `rutas_imagenes(incluir_todas=False)` — recorre `data/imagenes_visilab(raw)/<grupo>/<especie>/`.
- `calcular_copias_extra_por_especie(conteo, max_copias=3)` — `round(mediana/n - 1)` acotado.
- `obtener_genero(especie)` = primer token antes de `_`. `construir_numero_genero`, `etiquetas_a_generos`.
- `obtener_especies_activas()` — **lee `modelos/<PRUEBA>/metadatos_modelo.json`**; error si no existe.
- `verificar_especies_consistentes()` — impide reentrenar una PRUEBA con otras especies.
- `guardar_resumen_entrenamiento()` — añade entrada al histórico `resumen_entrenamiento.json`.
- `parsear_argumentos()` — `--reentrenar --regenerar-splits --recalcular-embeddings` (s/n), `--prueba`.

### embeddings.py
- `resolver_modelo_dinov2()` — `DINOV2_MODEL_PATH` > cache HF > descarga `facebook/dinov2-base`.
- `inicializar_dinov2(device=None)` → (processor, model eval/frozen, device, augmentation).
- `crear_augmentation()` — flips H/V, rotación 15°, affine leve, ColorJitter 0.08, blur p=0.15; relleno gris.
- `get_embedding(ruta, ..., is_train)` — pad → (augment → pad) → processor → CLS → L2 norm → CPU. Una imagen por llamada (sin batch).
- `calcular_embeddings(imagenes, ..., is_train, copias_por_especie)`.

### clasificador.py — `ClasificadorDiatomeas(num_clases, num_generos)`
- `forward(x)` → `(logits_especie, logits_genero, embedding_tronco[256])`.
- `distancia_a_centro(emb, idx)` (L2²), `es_desconocida(emb, idx_pred)` → bool si umbral finito y distancia > umbral.

### calibracion_confianza.py
- `ajustar_temperatura(logits, etiquetas)` — LBFGS sobre log T.
- `hornear_temperatura_en_pesos(modelo, T)` — divide W y b de `cabeza_especie` por T (in-place).
- `calibrar_centroides(modelo, centros, emb, et, percentil=95)`.
- `calibrar_pesos(modelo, centros, ruta)` — todo lo anterior y sobrescribe `.pth`.

### modelo.py
- `cargar_modelo_entrenado(ruta_pesos=None, ruta_metadatos=None, clases=None)` → `(modelo eval, especies ordenadas)`.
  Valida duplicados, orden vs txt, **len == 77** (hardcode), claves del state_dict.

### Tests
`python -m unittest tests.test_preprocesado tests.test_evaluar_campo` desde la raíz (importa `src.preprocesado`, `src.auditoria_dataset`).
Cubre: padding sin deformar, color de borde determinista, auditoría (duplicados + candidata).
