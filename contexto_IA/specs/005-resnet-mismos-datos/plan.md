# PLAN 005

## Receta (fiel al script recibido, con lo mínimo para que sea justa y quepa en 8 GB)
| | Script recibido | Spec 005 | Motivo del cambio |
|---|---|---|---|
| Modelo | ResNet50 ImageNet, todas las capas | igual | — |
| Cabeza | `Dropout(.5)-Linear(2048,256)-ReLU-Dropout(.3)-Linear(256,C)` (la del checkpoint de campo) | igual, C = 77 | — |
| Entrada | `Resize((256,256))` | igual | — |
| Aumentación | RandomAffine(shear 10, scale 0.8–1.2) + HFlip | igual | — |
| Optimizador | SGD lr 0.002, momentum 0.9, batch 4 | SGD momentum 0.9, **batch 32, lr 0.016** (escala lineal ×8) | con batch 4, 50 épocas son demasiado lentas |
| Precisión | fp32 | **AMP** (fp16 mixto) | velocidad y memoria |
| Épocas | 50 | 50 | — |
| Checkpoint | época 50 | **época 50 y la mejor por macro-F1 de val** | se reportan las dos |
| Desbalance | nada | nada | fidelidad a la receta |

## Datos
- `splits_75_objetivo_relativos.txt.gz` (versionado en esta carpeta): `split<TAB>ruta relativa a imagenes_visilab(raw)`,
  generado desde `data/splits/75_objetivo/*.txt`. El script busca cada imagen en `<repo>/data/imagenes_visilab(raw)/` de Ubuntu.
- Clases = `sorted(metadatos de 75_objetivo)` (igual que DINO).

## Ejecución en Ubuntu (8 GB)
- `nohup python3 contexto_IA/specs/005-resnet-mismos-datos/scripts/entrenar_resnet.py > log_resnet.txt 2>&1 &`
- Reanudable: guarda `ultimo.pth` (modelo + optimizador + época) en cada época; si se corta, se relanza igual y continúa.
- Estimación: ~3–5 min/época (lo más lento es decodificar los .tif/.png) → **3–4 h** en total. Opción `--epocas` para hacer una prueba corta.

## Campo
1. Inferencia con `--classifier resnet --resnet-weights modelos/resnet50_75_objetivo/mejor.pth --resnet-classes modelos/resnet50_75_objetivo/clases.txt`.
2. `evaluar_campo.py --resnet <excel>` (DINO-pad y ResNet recibida ya están medidas).

## Riesgos
- Imágenes que falten en Ubuntu (dataset distinto): el script aborta si falta más del 1 % e informa de cuáles.
- Calentamiento: no es un portátil, pero el script lleva el mismo freno térmico (pausa a 85 °C).
