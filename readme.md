# Clasificación de diatomeas con DINOv2

Clasifica especies de diatomeas en imágenes de microscopio. **YOLO** detecta cada diatomea y un clasificador
**DINOv2-base con fine-tuning parcial** (sus 4 últimos bloques más un MLP 768→512→256→77) decide la especie
entre 77 clases. ResNet50 se mantiene en la inferencia para comparar y para el ensamble.

Modelo vigente: `modelos/75_objetivo_ft/` — test interno accuracy 0.937, macro-F1 0.918; en el test de campo fijo
(`recursos/campo_test.txt`) 0.830 por imagen, frente a 0.815 de ResNet50 y 0.857 del ensamble.

## Estructura

```text
proyecto_transformer_v2/
├── tareas.py            # Todos los comandos del proyecto (Windows y Ubuntu)
├── Makefile             # Atajos para Ubuntu: make inferir, make evaluar…
├── src/
│   ├── entrenar.py          # Fine-tuning parcial de DINOv2 (único entrenamiento)
│   ├── dividir_datos.py     # Reparto train/val/test → recursos/splits_<prueba>.txt.gz
│   ├── dividir_campo.py     # Test de campo fijo / pool, sin fuga por muestra
│   ├── evaluar_campo.py     # Accuracy por imagen en campo (DINO, ResNet y ensamble)
│   ├── auditoria_dataset.py # Distribución, duplicados y especies candidatas
│   ├── preprocesado.py      # Padding cuadrado (igual en entrenamiento e inferencia)
│   └── constantes.py, clasificador.py, embeddings.py, preparar_datos.py, calibracion_confianza.py
├── Inferir/
│   ├── infer_and_split_resnet_single_folder.py   # Inferencia YOLO + DINOv2/ResNet50 → Excel + imágenes anotadas
│   └── txt_classes/                              # Clases DINOv2 (77) y ResNet50 (78)
├── recursos/            # Splits con rutas relativas y listas campo_test / campo_pool
├── modelos/75_objetivo_ft/   # Metadatos e informes del modelo (el .pth no se versiona)
├── tests/
└── data/                # (no versionado) imagenes_visilab(raw)/<fuente>/<especie>/*.png|tif
```

## Instalación

Python ≥ 3.10. Para GPU NVIDIA, instala primero el PyTorch de tu versión de CUDA y después el resto:

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Los pesos (`*.pt`, `*.pth`) no están en git. Hay que colocarlos en:
`modelos/75_objetivo_ft/modelo_75_objetivo_ft.pth`, `Inferir/yolo_dinov2/yolo_best.pt` e `Inferir/yolo_dinov2/resnet50_checkpoint_epoch50.pth`.
Compruébalo con `python tareas.py estado`.

## Uso

| Qué | Comando (Windows o Ubuntu) | Ubuntu |
|---|---|---|
| Comprobar modelo, pesos y GPU | `python tareas.py estado` | `make estado` |
| Inferencia de campo | `python tareas.py inferir [--entrada CARPETA] [--prueba NOMBRE] [--classifier both\|dinov2\|resnet]` | `make inferir` |
| Evaluar una inferencia en el test de campo | `python tareas.py evaluar --resultados DIR --etiquetas cruce_ground_truth.xlsx` | `make evaluar RESULTADOS=… ETIQUETAS=…` |
| Reparto train/val/test | `python tareas.py dividir-datos` | `make dividir-datos` |
| Test de campo fijo / pool | `python tareas.py dividir-campo --etiquetas cruce_ground_truth.xlsx` | `make dividir-campo ETIQUETAS=…` |
| Entrenar | `python tareas.py entrenar [opciones de src/entrenar.py]` | `make entrenar` |
| Tests | `python tareas.py test` | `make test` |
| Liberar espacio (lista; borra con `--si`) | `python tareas.py limpiar` | `make limpiar` |

### Inferencia
Recorre la carpeta de entrada (por defecto `Inferir/imagenes_inferencia/`), ejecuta YOLO una vez por imagen y clasifica los
recortes en lote. Todos los resultados van a una única carpeta, sin sobrescribir nunca:
`Inferir/resultados_inferencia/<modelo>/<prueba>/` (`<modelo>` = carpeta de los pesos DINOv2 y/o `resnet50`;
`<prueba>` = `--prueba` o la fecha y hora). Contiene:
- un Excel combinado y uno por modelo (top-1/2/3, confianza y marca de revisión con `--threshold 0.80`);
- las imágenes anotadas en `bbox/` (recuadro verde con el nombre de la especie);
- los recortes en `crops/`.

### Evaluación en campo
`evaluar` mide la accuracy por imagen con la suma de confianza de sus recortes. `Debris`, `Fragments` y las clases `_fp`
(posición pleural, solo género) no votan. Por defecto usa solo `recursos/campo_test.txt`, que nunca debe usarse para entrenar;
`--todo` evalúa todas las imágenes.

### Entrenamiento
`src/entrenar.py` lee `recursos/splits_75_objetivo_relativos.txt.gz` (o `--splits`), comprueba que las imágenes existen,
entrena con aumentación online, calibra la temperatura en validación y guarda en `modelos/<prueba>_<fecha>/` (o `--salida`;
nunca en una carpeta con un modelo terminado, como `modelos/75_objetivo_ft/`) el modelo, los metadatos,
`metricas.json`, `reporte_test.txt`, `confusiones.txt` y las curvas. Es reanudable (relanzar con la misma `--salida`, que se imprime al empezar) y elige la precisión según la GPU (fp32 en Pascal).
Los nombres de carpeta con espacios o puntos se normalizan (`Fistulifera saprophila` → `Fistulifera_saprophila`).
