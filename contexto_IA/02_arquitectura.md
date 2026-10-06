# 02 — Arquitectura

## Estructura del repo
```text
proyecto_transformer_v2/
├── src/                 # Código principal (imports planos: ejecutar desde src/ o con src en sys.path)
├── Inferir/             # Script de inferencia heredado + listas de clases (txt_classes/)
│   ├── infer_and_split_resnet_single_folder.py   # inferencia por lotes, más completa
│   ├── txt_classes/classes_77(dino).txt, classes_78(resnet).txt
│   ├── imagenes_inferencia/   (ignorado por git)
│   └── yolo_dinov2/           (NO existe localmente: aquí van yolo_best.pt y resnet50_*.pth)
├── modelos/<PRUEBA>/    # Pesos (.pth ignorado) + metadatos, reportes, gráficas
├── data/                # IGNORADO por git: imágenes, splits, embeddings, metadata
├── auxiliar/            # Scripts sueltos de mantenimiento del dataset (duplicados, conteos…)
├── tests/               # unittest (solo preprocesado + auditoría)
├── contexto_IA/         # ESTE contexto
├── readme.md, manual_de_uso.txt, notas.md (cuaderno didáctico), comandos.txt
└── analisis_clasificadores.xlsx, comparacion_confusiones.png
```

## Pipeline de entrenamiento (`src/main.py`)
```
imagenes_visilab(raw)/<grupo>/<especie>/*.png|jpg|tif
   │  generar_leer_splits.generar_split()   70/15/15 estratificado, random_state=42
   ▼
data/splits/<PRUEBA>/{train,val,test,unknown}.txt   (rutas absolutas; especie = carpeta padre)
   │  main.preparar_embeddings_splits() → embeddings.calcular_embeddings()
   │    preprocesado.pad_to_square(mean_border) → [augment solo train] → AutoImageProcessor(224)
   │    → DINOv2 pooler_output → float32 → L2-normalizar
   │    train: original + 1 aumentada + copias extra (minoritarias, hasta 3)
   ▼
data/embeddings_procesado/<PRUEBA>/embeddings_<split>.pt   {"embeddings": [N,768], "etiquetas": [str]}
   │  preparar_datos.get_datos() + codificacion()  (especies ordenadas alfabéticamente → índice)
   ▼
dataloader.crear_dataloaders()   WeightedRandomSampler (1/n^0.3) en train
   │
   ▼
clasificador.ClasificadorDiatomeas
   tronco: Linear(768,512)-ReLU-Drop(0.3)-Linear(512,256)-ReLU-Drop(0.2)
   cabeza_especie: Linear(256, 77)    cabeza_genero: Linear(256, n_generos)
   buffers: centros[77,256], umbral_distancia[77]
   │  entrenamiento.entrenar_modelo()  AdamW + warmup(3)+coseno, CE(label_smoothing 0.05, pesos 1/n^0.3)
   │  early stopping por macro-F1 val (paciencia 7), clip grad 1.0
   ▼
modelos/<PRUEBA>/modelo_<PRUEBA>.pth   (mejor época)
   │  calibracion_confianza.calibrar_pesos(): temperatura (LBFGS en val) horneada en cabeza_especie
   │  + centroides/umbral p95 por clase → sobrescribe .pth
   ▼
evaluar_test_metricas.main() → reporte_test.txt, matriz, métricas por especie, top-3
preparar_datos.guardar_resumen_entrenamiento() → resumen_entrenamiento.json (append histórico)
```

Pérdida total: `CE_especie + PESO_GENERO·CE_genero + LAMBDA_CENTER_LOSS·center` (ambos pesos = 0 hoy).

## Pipeline de inferencia (`Inferir/infer_and_split_resnet_single_folder.py`; `src/inferencia.py` no se usa)
```
imagen completa → YOLO (ultralytics) → cajas xyxy → recorte
   ├── DINOv2: pad_to_square → processor → backbone → [L2 norm] → MLP → softmax → top-3
   │           + es_desconocida() por distancia a centroide (solo src/inferencia.py)
   └── ResNet50: Resize 256 → Normalize ImageNet → fc custom (2048→256→78) → top-3
→ Excel (predicciones.xlsx) + (script heredado) Excel por modelo + bbox anotadas
```
Detalle en `05_inferencia.md`.

## Artefactos y contratos entre etapas
| Artefacto | Productor | Consumidor | Contrato |
|---|---|---|---|
| `splits/*.txt` | generar_split | preparar_embeddings_splits, errores.py | una ruta por línea; especie = `Path.parent.name` |
| `embeddings_*.pt` | calcular_embeddings | get_datos | dict `embeddings` Tensor[N,768] float32 L2-norm, `etiquetas` list[str]; orden test = orden test.txt (errores.py lo comprueba) |
| `metadatos_modelo.json` | main.py | obtener_especies_activas, modelo.py, inferencia | clave `especies_filtradas` (lista); orden de logits = `sorted()` |
| `modelo_<PRUEBA>.pth` | entrenar_modelo + calibrar_pesos | modelo.cargar_modelo_entrenado | state_dict; `centros`/`umbral_distancia` opcionales |
| `classes_77(dino).txt` | manual | inferencia | 77 líneas = `sorted(especies_filtradas)` exactamente |

**Invariante crítico:** el índice de clase = posición en la lista **alfabéticamente ordenada** de especies.
