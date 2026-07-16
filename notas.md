# Notas del proyecto Clasificador de diatomeas (DINOv2 + capa lineal)

Este documento explica los conceptos de deep learning usados en el proyecto,
en el orden en el que aparecen al ejecutar `main.py`, y por qué se tomó cada
decisión de diseño. Está pensado como mi cuaderno de aprendizaje.

Al final de cada sección tienes un apartado **"Dónde estaba esto en el
código → bórralo de ahí"** con el comentario exacto a eliminar.

---

## 0. Visión general: qué hace el pipeline y por qué

El objetivo es clasificar imágenes de diatomeas (microalgas) en su especie.
Hay dos grandes enfoques posibles:

1. **Entrenar una red desde cero** con las imágenes: necesita muchísimos
   datos por clase y mucho tiempo de entrenamiento.
2. **Usar un extractor de características ya entrenado** (DINOv2) y montar
   encima algo pequeño y rápido de entrenar.

Se eligió la opción 2. Con pocas imágenes por especie (algunas clases
tienen menos de 50), entrenar una red grande desde cero sería inviable:
no habría datos suficientes para que aprendiera texturas y formas por sí
misma. DINOv2, en cambio, ya ha visto millones de imágenes genéricas y ha
aprendido a representar "qué hay en una imagen" de forma muy general solo
hace falta enseñarle a un clasificador pequeño cómo traducir esa
representación a "qué especie de diatomea es esta".

**Flujo completo (orden real de ejecución en `main.py`):**

1. `generar_leer_splits.py` → decide qué imágenes van a train/val/test.
2. `embeddings.py` → convierte cada imagen en un vector de 768 números
   (embedding) usando DINOv2, y los guarda en disco.
3. `preparar_datos.py` → carga esos embeddings y convierte las etiquetas de
   texto ("Nitzschia_sp") en números (0, 1, 2...).
4. `dataset.py` + `dataloader.py` → organizan los embeddings en lotes
   (batches) para dárselos al modelo durante el entrenamiento.
5. `clasificador.py` → la red neuronal pequeña que aprende a mapear
   embedding → especie.
6. `entrenamiento.py` → el bucle que entrena esa red época a época.
7. `evaluar_test_metricas.py` → mide qué tal funciona el modelo ya
   entrenado, sobre imágenes que nunca ha visto.
8. `errores.py` → lista los fallos concretos, para revisarlos a mano.
9. `inferencia.py` → usa el modelo ya entrenado sobre imágenes nuevas,
   fuera del dataset original.

---

## 1. Splits: por qué dividir en train / val / test

**Qué es:** dividir el conjunto de imágenes en tres grupos separados que no
se solapan entre sí:

- **train** (70%): con esto se ajustan los pesos de la red.
- **val** (15%): se usa *durante* el entrenamiento, después de cada época,
  para comprobar si el modelo generaliza o si solo está memorizando train.
  No se entrena con estos datos, solo se miden.
- **test** (15%): se usa una única vez, al final, sobre el modelo ya
  entrenado y elegido. Es la medida más honesta de cómo se comportaría el
  modelo con imágenes reales que nunca ha visto ni de forma indirecta.

**Por qué separar val de test si los dos son "datos no vistos":** porque
val se usa muchas veces (cada época) para tomar decisiones cuál es la
mejor época, cuándo hacer early stopping, qué hiperparámetros probar. Al
usarlo tantas veces para decidir, el modelo acaba "sobreajustando"
un poco a val sin querer, aunque el modelo nunca vea esas imágenes
directamente en el entrenamiento. Por eso hace falta un tercer conjunto,
test, que no interviene en ninguna decisión y solo se mira al final.

**Por qué es estratificado:** significa que la proporción de cada especie
se mantiene igual en los tres conjuntos. Si no se hiciera así, podría pasar
por azar que una especie rara casi no tenga representación en val o test, y
entonces sus métricas serían poco fiables (calculadas sobre 2-3 imágenes).

**Por qué se guarda en `.txt` en vez de recalcularlo cada vez:** para que
el split sea siempre el mismo entre ejecuciones. Si se regenerara al azar
cada vez, no se podría comparar de forma justa un entrenamiento con otro
no sabrías si un cambio en el modelo mejoró las cosas, o si simplemente le
tocó un split más fácil.

---

## 2. Embeddings: qué son y por qué se usan en vez de imágenes crudas

**Qué es un embedding:** un vector de números (en nuestro caso, 768
números) que representa el "contenido" de una imagen de forma compacta.
Imágenes parecidas (misma especie, formas similares) producen vectores
parecidos entre sí; imágenes distintas producen vectores alejados. DINOv2
es el modelo que convierte imagen → estos 768 números.

**Por qué no entrenar directamente sobre las imágenes:** porque eso
implicaría o bien entrenar DINOv2 entero (carísimo computacionalmente e
innecesario con este tamaño de dataset) o bien pasar cada imagen por
DINOv2 en cada época de entrenamiento (mucho más lento, y repetitivo,
porque DINOv2 no cambia sus pesos están congelados).

**Por qué se precalculan y se guardan en `.pt` en vez de calcularlos al
vuelo:** DINOv2 está congelado (`model.requires_grad_(False)`), así que su
salida para una misma imagen no cambia nunca durante el entrenamiento. Si
se recalculara en cada época, se repetiría el mismo trabajo caro (pasar la
imagen por una red grande) cientos de veces. Calculándolo una vez y
guardándolo, cada época de entrenamiento solo tiene que entrenar la parte
pequeña (`ClasificadorDiatomeas`), que es rapidísima.

**Por qué solo train tiene augmentation y val/test no:**

- **Augmentation** (flips, rotaciones, jitter de color, blur) genera
  variaciones artificiales de las mismas imágenes para que el modelo vea
  más diversidad y no memorice detalles concretos (overfitting). Esto
  tiene sentido solo en train, porque ahí es donde se ajustan los pesos.
- En val y test no se usa augmentation, porque necesitas medir el
  rendimiento sobre imágenes "reales", tal cual, para que la métrica sea
  representativa de un caso de uso real (una imagen nueva que llega, sin
  transformar).

**Qué es `pooler_output` (el "token CLS"):** DINOv2 procesa la imagen como
una rejilla de parches y genera una representación por cada parche, más un
token especial que resume toda la imagen. Ese resumen es lo que se usa
como embedding, porque está pensado precisamente para tareas de
clasificación (representar "toda la imagen" en un solo vector). Esta idea
realmente está tomada de los transformers originales NLP porque realmente
se podria hacer con mean pooling.

**Por qué se normaliza el embedding (norma 1):** ayuda a que todos los
vectores tengan una "escala" comparable entre sí, lo cual hace más estable
el entrenamiento de la capa lineal que viene después (evita que unos
embeddings con valores muy grandes dominen sobre otros solo por escala, no
por contenido real).

### Augmentation extra para clases minoritarias

Las especies con pocas imágenes reciben más copias aumentadas (3 extra si
están en `ESPECIES_MINORITARIAS`, 5 extra si están en
`ESPECIES_MUY_MINORITARIAS`). Esto es una primera capa de compensación del
desbalance se solapa con el `WeightedRandomSampler` y los pesos de la
loss (ver sección 5), así que si el desbalance sigue siendo un problema al
escalar a 10 o ~100 especies, este es uno de los tres mecanismos a los que
puedes bajarle la intensidad para no sobrecorregir.

Ejemplo embedding [0.23, -1.4, 0.87, 0.01, ..., 0.55] = 768 números per imagen.



---

## 3. Dataset: qué es y por qué existe esta clase

**Qué es:** en PyTorch, un `Dataset` es simplemente un contrato: cualquier
clase que lo implemente debe saber responder a dos preguntas:

- `__len__`: ¿cuántos elementos hay en total?
- `__getitem__(i)`: dame el elemento número `i`.

PyTorch no necesita saber si esos elementos son imágenes, embeddings, filas
de un CSV o lo que sea con esas dos respuestas, ya sabe cómo recorrerlos,
mezclarlos y agruparlos en lotes a través del `DataLoader`.

**Por qué `MyDataset` guarda embeddings y etiquetas por separado (dos
listas/tensores) en vez de una lista de tuplas:** porque `__getitem__`
necesita devolver el par `(embedding, etiqueta)` para esa posición
concreta, y tenerlos como dos tensores alineados por índice hace esa
operación directa e instantánea (`self.embeddings[index],
self.etiquetas[index]`), sin tener que reconstruir nada.


---

## 4. DataLoader: qué es y qué decisiones se tomaron aquí

**Qué es:** el `DataLoader` es quien realmente usa el `Dataset` durante el
entrenamiento. Se encarga de:

- Agrupar ejemplos individuales en **batches** (lotes) de tamaño fijo
  (aquí, `BATCH_SIZE = 32`), en vez de pasar la red neuronal por una
  imagen cada vez (muy lento) o por todo el dataset a la vez (no cabría en
  memoria y sería inestable).
- Mezclar el orden de los datos en cada época (o, en este caso, muestrear
  con pesos ver más abajo) para que el modelo no aprenda patrones falsos
  basados en el orden.
- Cargar los datos en paralelo con varios procesos (`NUM_WORKERS`) para
  que la CPU prepare el siguiente batch mientras la GPU procesa el
  actual, y así no haya tiempos muertos.

**Por qué train usa `WeightedRandomSampler` y val no:** el sampler decide
con qué probabilidad se elige cada muestra en cada batch. Se usa solo en
train para combatir el desbalance de clases (que unas especies tengan
muchas más imágenes que otras) se le da más probabilidad de aparecer a
las clases raras. En val no tiene sentido usarlo: en val no se ajustan
pesos, solo se mide, y para medir de forma representativa necesitas ver
los datos con su proporción real, no artificialmente rebalanceada.

**Por qué `pin_memory` y `persistent_workers`:** son optimizaciones de
rendimiento. `pin_memory=True` acelera la transferencia CPU→GPU.
`persistent_workers=True` evita que los procesos worker se destruyan y
recreen en cada época (crearlos tiene un coste, así que se mantienen
vivos entre épocas).


---

## 5. El desbalance de clases: los tres mecanismos y por qué no deben sumarse sin control

Este proyecto he usado **tres** mecanismos distintos para lidiar con que unas
especies tienen muchas más imágenes que otras:

1. **Augmentation extra** en `embeddings.py` para clases minoritarias
   (más copias sintéticas de las especies raras).

2. **`WeightedRandomSampler`** en `dataloader.py` (más probabilidad de
   elegir muestras de clases raras en cada batch).

3. **Pesos en la loss** (`calcular_pesos_clases`) los errores en clases
   raras penalizan más al calcular cuánto se equivocó el modelo.

Los tres hacen, en esencia, lo mismo: decirle al modelo "presta más
atención a las clases con pocos ejemplos". El riesgo de tener los tres a
la vez con intensidades descoordinadas es que el modelo se pase de frenada
y empiece a "sobre-predecir" las clases raras es decir, alta recall pero
baja precision en esas clases (como se vio con `Gomphonema_pumilum`: recall
0.97 pero precision 0.76). Por eso se igualó el suavizado del sampler
(`1/sqrt(n)`) y la loss (`1/sqrt(n)`, antes era `1/n`) para que ambos
tiren con la misma fuerza y no se acumule doble corrección.

Si al escalar a 10 y luego a ~100 especies el problema persiste, el ajuste
fino pasa por bajar la intensidad de uno de estos tres mecanismos, no por
subir los tres a la vez.

---

## 6. El clasificador: por qué esta arquitectura

**Qué hace la capa lineal (`nn.Linear`):** transforma un vector de entrada
en otro de salida aplicando pesos y un sesgo literalmente, una
combinación lineal de los 768 números de entrada por cada una de las
salidas. Los "pesos" son justo los números que se ajustan durante el
entrenamiento para que esa transformación tenga sentido.

**Por qué varias capas (768→512→256→num_clases) y no una sola:** con una
sola capa lineal, el modelo solo puede separar clases si son "linealmente
separables" en el espacio de 768 dimensiones a veces eso basta, pero
aquí no fue suficiente. Añadir capas intermedias con `ReLU` permite
al modelo aprender relaciones no lineales entre los embeddings y las
clases, dándole más capacidad de separar especies parecidas entre sí.

**Qué es ReLU:** una función de activación muy simple: deja pasar los
valores positivos tal cual y convierte los negativos en cero. Sin este
tipo de función entre capas lineales, apilar varias capas lineales
seguidas equivaldría matemáticamente a tener una sola capa la no
linealidad es lo que le da poder real a la arquitectura multicapa.

**Qué es Dropout y por qué 0.3 y luego 0.2:** durante el entrenamiento,
apaga aleatoriamente ese porcentaje de neuronas en cada paso. Esto obliga
al modelo a no depender excesivamente de neuronas concretas, lo cual ayuda
a evitar overfitting (que memorice train en vez de generalizar). El valor
más alto (0.3) en la capa más grande y más bajo (0.2) en la más pequeña es
razonable: cuantas más neuronas hay, más margen hay para "apagar" algunas
sin perder demasiada capacidad.

**Por qué inicialización Xavier:** al crear la red, los pesos empiezan con
valores aleatorios, pero no cualquier aleatoriedad vale si empiezan
demasiado grandes o demasiado pequeños, el entrenamiento puede ser
inestable desde el principio. Xavier es una forma de inicializar esos
pesos aleatorios con una escala pensada para que la señal no explote ni se
desvanezca al pasar por varias capas.


## 7. Entrenamiento: qué hace cada pieza del bucle

**Optimizador (AdamW):** es el algoritmo que, en cada paso, decide cuánto
y en qué dirección mover cada peso de la red para reducir el error. Usa
`lr=0.0003` (qué tan grandes son esos pasos) y `weight_decay=0.0001`
(penaliza pesos demasiado grandes, otra forma más de evitar overfitting,
complementaria al Dropout).

**Scheduler (warmup + coseno):** el learning rate no es fijo durante todo
el entrenamiento.

- **Warmup** (primeras 3 épocas): el LR sube gradualmente desde casi 0
  hasta el valor normal. Al principio los pesos están inicializados al
  azar, así que un LR alto desde el primer paso podría dar un "giro" muy
  brusco a la red antes de que tenga ninguna noción de la tarea.

- **Descenso coseno** (resto de épocas): el LR baja suavemente. Esto
  permite pasos grandes al principio (explorar rápido) y pasos cada vez
  más finos según se acerca al final (afinar sin desestabilizar lo ya
  aprendido).

**Pérdida (loss) CrossEntropyLoss:** mide cuánto se aleja la predicción
del modelo (una puntuación por especie) de la especie real. Cuanto más
segura y equivocada esté la predicción, mayor es la pérdida. Es la
cantidad que el optimizador intenta minimizar.

- `label_smoothing=0.05`: en vez de pedirle al modelo que esté 100%
  seguro de la clase correcta, se le pide un 95%, dejando un pequeño
  margen. Esto evita que el modelo se vuelva excesivamente confiado

  (overconfident), lo cual suele generalizar mejor.
- `weight=pesos_clase`: ver sección 5.

**Gradient clipping:** después de calcular cuánto debería cambiar cada
peso (`backward()`), se limita ese cambio para que no sea desproporcionado
en ningún paso. Protege contra "explosiones" puntuales de gradiente que
podrían desestabilizar el entrenamiento.

**Validación tras cada época:** se mide el modelo en `val` (sin entrenar
con esos datos) para ver si mejora de verdad o si solo está memorizando
train. Se usa `macro F1` (media del F1 de cada clase, todas con el mismo
peso) en vez de accuracy simple, porque con clases desbalanceadas la
accuracy puede ser engañosa un modelo que siempre predice la clase
mayoritaria tendría accuracy alta pero sería inútil para las clases raras.

**Early stopping:** si el macro F1 en val no mejora durante `PACIENCIA=5`
épocas seguidas, se detiene el entrenamiento antes de llegar a las 40
épocas totales. Evita seguir entrenando (y arriesgarse a overfitting)
cuando el modelo ya ha dejado de mejorar de verdad.

**Por qué se guarda el modelo solo cuando mejora el macro F1:** así el
archivo `mejor_modelo.pth` siempre contiene la mejor versión vista durante
todo el entrenamiento, no la última (que podría ser peor si el modelo
empezó a sobreajustar en las últimas épocas).


## 8. Evaluación: matriz de confusión y métricas por clase

**Matriz de confusión:** una tabla donde cada fila es la especie real y
cada columna la especie predicha. La diagonal son los aciertos; todo lo
demás fuera de la diagonal son errores, y te dice *con qué* se confunde
cada especie información mucho más rica que un solo número de accuracy.

**Precision, recall, F1 (por clase):**
- **Precision**: de todo lo que el modelo etiquetó como especie X,
  ¿cuánto era realmente X? (mide falsos positivos).
- **Recall**: de todo lo que realmente era especie X, ¿cuánto detectó el
  modelo como X? (mide falsos negativos).
- **F1**: media armónica de las dos anteriores, un único número que
  penaliza si cualquiera de las dos es baja.
- **Macro F1**: la media del F1 de cada especie, dando el mismo peso a
  clases con muchas imágenes que a clases con pocas. Es la métrica
  correcta aquí porque el objetivo es que el modelo funcione bien en
  *todas* las especies, no solo en las mayoritarias.



## 9. `errores.py`: por qué existe este script aparte

No es una herramienta de entrenamiento ni de evaluación agregada es para
inspección manual, caso por caso. Su función es cruzar predicción vs
etiqueta real y devolver la ruta exacta de cada imagen mal clasificada,
para poder abrirla y comprobar a ojo si el fallo es del modelo o del
propio etiquetado del dataset.
---

## 10. `inferencia.py`: diferencia con el resto del pipeline

Todo lo anterior trabaja sobre el dataset conocido (train/val/test, con
etiquetas). `inferencia.py` es el único script pensado para imágenes
**nuevas**, sin etiqueta, fuera del dataset original el caso de uso real
final una vez el modelo esté listo para el cliente. Por eso calcula el
embedding al vuelo (no hay un `.pt` precalculado para imágenes que no
existían de antemano) y no usa augmentation (`is_train=False`), igual que
val/test.

`intervalo_confianza` (0.60 por defecto) es el umbral por debajo del cual
una predicción se marca como "revisar" no es que el modelo esté
"seguro" o no en un sentido absoluto, es una decisión de diseño: cuándo
confiar en la predicción automática y cuándo pedir revisión humana.