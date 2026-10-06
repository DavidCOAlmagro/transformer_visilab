# 11 — Glosario

| Término | Significado |
|---|---|
| Diatomea | Microalga con pared de sílice (frústulo); su composición de especies indica la calidad del agua. |
| ROI / recorte | Región de la imagen de microscopio con una diatomea; lo que clasifica el modelo. |
| Especie | Clase objetivo, nombre `Genero_epiteto[_var_x]` = nombre de carpeta (p. ej. `Nitzschia_palea_var_palea`). |
| Género | Primer token antes de `_` (p. ej. `Nitzschia`). Tarea auxiliar opcional. |
| `_fp` | **Posición pleural**: la diatomea vista de lado (cíngulo), no de valva. Solo identificable a nivel de **género** (`Gomphonema_fp`). No es una especie. En campo, una imagen de una especie puede contener recortes en posición pleural de esa especie. |
| Vista valvar | Vista frontal de la valva; la que permite identificar la especie. Las clases de especie son vistas valvares. |
| `_sp` | Especie no determinada del género (aparece en datos/confusiones antiguas). |
| `Debris`, `Fragments` | Clases no-diatomea: restos y fragmentos. Forman parte de las 77. |
| PRUEBA | Nombre de experimento; define carpetas en `data/splits`, `data/embeddings_procesado`, `modelos`. |
| `75_objetivo` | Experimento vigente (77 clases pese al nombre). |
| unknown | Split con imágenes de especies fuera del experimento, para evaluar rechazo. |
| Desconocida | Etiqueta de inferencia cuando la distancia al centroide supera el umbral de la clase. |
| Embedding | Vector 768 de DINOv2 (CLS) normalizado L2. |
| Tronco | Parte compartida del MLP (salida 256), usada también para centroides. |
| Temperatura (T) | Escalar de calibración; se divide en los pesos de la cabeza de especie. |
| Reinhard | Normalización de color; el dataset UDE ya viene normalizado así. No implementada en inferencia. |
| UDE Diatoms in the Wild 2024 | Dataset público (~84k ROIs) integrado como fuente de datos. |
| YOLO | Detector (ultralytics) que localiza diatomeas en la imagen completa. |
| ResNet50 | Clasificador CNN alternativo, 78 clases, pesos externos; se compara con DINOv2. |
| Revisión | Marca en el Excel cuando confianza < `UMBRAL_CONF`, sin detecciones o desconocida. |
| SDD | Spec-Driven Development (ver `09_metodologia_SDD.md`). |
