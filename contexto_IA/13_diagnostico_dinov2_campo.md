# 13 — Diagnóstico: por qué DINOv2 rinde peor que ResNet50 en campo

Análisis del 2026-10-06 sobre `C:\VISILAB\classification_results_{dinov2,resnet}.xlsx` +
`cruce_ground_truth.xlsx` + `data/splits/75_objetivo/train.txt`. Script de análisis ad hoc
(no versionado); repetir con pandas uniendo por `image,item,x1,y1,x2,y2` y por `image` con `Por_imagen`.

> Nota de método: a nivel de **recorte** la etiqueta es la de la imagen, que puede contener varias especies,
> así que la accuracy por recorte (~0.29 DINO / 0.35 ResNet) es ruidosa. Solo sirve para comparar los dos
> modelos entre sí; la métrica que vale es la de **imagen**.

## 1. Evidencias

### E1 — Desfase de dominio: especies entrenadas casi solo con UDE
Composición de train (75_objetivo, 45 985 imgs): UDE (otro laboratorio, normalizado Reinhard) 29 676 (65 %),
`dataset_aq_dbo5_agrupado` 10 329, `aq_dbo5_last` 4 032, Common 1 581, Unique 280, Seleccion_5 87.

| % de train procedente de UDE | nº especies | acc DINO | acc ResNet |
|---|---|---|---|
| < 20 % | 26 | 0.625 | 0.670 |
| 20–80 % | 30 | 0.591 | 0.673 |
| > 80 % | 9 | **0.514** | **0.749** |

- `Fistulifera_saprophila`: 100 % UDE/Selección (188) → DINO predice Mayamaea/Debris (acc 0.10, ResNet 0.90).
- `Rhoicosphenia_abbreviata` 76 % UDE, `Navicula_gregaria` 90 % UDE → las dos con huecos grandes.
- DINO gana donde hay mucho dato del dominio Aqualitas (`Cymbella_parva`, `Nitzschia_amphibia`).
- Spearman(acc DINO, nº imgs del dominio aq/Common) = **0.39**.
⇒ Con el backbone **congelado** y solo un MLP encima, DINOv2 generaliza mal de UDE al microscopio
de Aqualitas/DBO5. ResNet (probablemente fine-tuned de extremo a extremo, quizá con otros datos) se adapta mejor.

### E2 — Recortes alargados: el center-crop pierde los extremos
Accuracy por recorte según la relación de aspecto de la caja YOLO:

| relación de aspecto | n | DINO | ResNet | diferencia |
|---|---|---|---|---|
| 1–1.5 | 34 451 | 0.334 | 0.358 | −0.02 |
| 1.5–2 | 21 642 | 0.278 | 0.337 | −0.06 |
| 2–3 | 19 744 | 0.255 | 0.332 | −0.08 |
| 3–5 | 6 233 | 0.260 | 0.373 | −0.11 |
| > 5 | 684 | 0.133 | 0.251 | −0.12 |

El `AutoImageProcessor` de dinov2-base hace resize del lado corto a 256 + **center crop 224**
(comprobado: `do_center_crop=True`). Con una diatomea alargada se descartan los extremos (ápices, que son
claves para distinguir la especie). ResNet hace `Resize((256,256))`: deforma, pero ve el objeto entero.
La solución (pad-square, D-011) ya está programada. **Confirmado en spec 002**: +3.0 macro-F1 en test interno, +8.4 pts en relación 3–5. Falta reentrenar el modelo oficial.

### E3 — Recortes pequeños
Lado < 100 px: los dos modelos ≈ 0.09. Entre 100 y 300 px DINO va unos 4–5 puntos por detrás. Es un problema de YOLO o de la
resolución, común a ambos modelos.

### E4 — Clases `_fp` (posición pleural) que absorben predicciones y Debris/Fragments
> `_fp` = posición pleural (vista lateral, nivel género). **Resuelto en parte en la evaluación (spec 001):** si no votan,
> DINOv2 sube de 0.675 a **0.741** y ResNet de 0.762 a **0.827**. La diferencia entre modelos se mantiene (~8.5 pts) y se concentra en especies
> entrenadas casi solo con UDE (Fistulifera 0.19 vs 0.98, N. gregaria 0.34 vs 0.92, P. lanceolatum 0.47 vs 0.96).

- DINO predice Debris/Fragments en el 7.9 % de los recortes evaluables, frente al 3.7 % de ResNet.
- Las `_fp` absorben especies con poco train del mismo género:
  `Gomphonema_rhombicum` (62 en train) frente a `Gomphonema_fp` (741, misma fuente) → 166 imágenes confundidas.
  Además, la ponderación por clase (1/n^0.3) y el sampler empujan hacia las clases medianas.
- Lo sufren los dos modelos, así que es un problema de **taxonomía y etiquetas**, no del backbone.

### E5 — Agregación por imagen
Sumar la confianza por especie es la mejor regla de las tres para ambos modelos (+3.4 puntos DINO frente al voto).
Las confianzas de DINO están calibradas (T horneada): media 79 % en aciertos y 53 % en fallos. ResNet da 92 / 69 (más sobreconfiado).

### E6 — Lo que NO es la causa (de esta inferencia)
- L2-norm: el script `Inferir/` sí normaliza (el bug B1 solo afecta a `src/inferencia.py`).
- Desajuste por pad-square: los Excel se generaron con una versión sin pad (commit ≤ `0b42d93`), así que fueron coherentes.
- fp16 (autocast) en inferencia frente a fp32 en entrenamiento: impacto esperado mínimo (sin verificar).

## 2. Causas probables (por peso estimado)
1. **Desfase de dominio** + backbone congelado (E1).
2. **Pérdida de información por el center crop** en especies alargadas (E2).
3. **Aumentación pobre**: solo 1 vista aumentada fija por imagen (más copias en minoritarias), precalculada;
   no simula el encuadre de YOLO, la escala, el desenfoque ni el color de otro microscopio.
4. Taxonomía `_fp` / Debris (E4), que afecta a los dos modelos.

## 3. Comprobaciones pendientes (baratas, antes de reentrenar)
- [ ] Preguntar con qué datos y qué preprocesado se entrenó la ResNet50 (¿Aqualitas? ¿Reinhard? ¿recortes YOLO?).
- [ ] Ver si las imágenes de inferencia están normalizadas con Reinhard (`norm1`) y si los datos `aq_dbo5` de train lo están.
- [ ] Sobre unos 2 000 recortes de nivel A/C, comparar las predicciones con center-crop frente a pad-square usando los
      pesos actuales (cuantifica E2 sin reentrenar). Los recortes de `Inferir/imagenes_inferencia/crops` están vacíos: regenerar.

## 4. Plan de mejora propuesto (ranking por relación beneficio/coste)
| # | Acción | Coste | Esperado |
|---|---|---|---|
| 1 | **Reentrenar 75_objetivo con pad-square** (recalcular embeddings, mismos splits) | bajo (horas) | + en especies alargadas; elimina B4 |
| 2 | **Datos del dominio**: añadir recortes YOLO de Aqualitas/DBO5 de nivel A (y C/B revisados) como grupo nuevo de train, **separando por imagen o muestra** para evitar fuga; reservar un test de campo fijo | medio | el mayor impacto esperado (E1) |
| 3 | **Aumentación realista** precalculada K vistas: jitter de la caja tipo YOLO (escala/traslación), color/gamma/contraste más fuerte, blur, ruido; o normalización de color homogénea (Reinhard en train e inferencia) | medio | robustez de dominio |
| 4 | **Fine-tuning parcial de DINOv2** (últimos 2–4 bloques o LoRA) con LR baja | alto (GPU) | cerrar la diferencia con ResNet |
| 5 | Mejores features: CLS + media de tokens de parche; probar `dinov2-large` o una resolución mayor | medio | + discriminación fina |
| 6 | Decidir el tratamiento de `_fp`: fusionar en un nivel género, jerarquía, o métrica “género correcto” | bajo/medio | menos absorción |
| 7 | Inferencia: agregación por suma de confianza; ensamble DINO+ResNet | bajo | + inmediato en producción |
| 8 | Test de campo estándar (las imágenes de nivel A + revisadas) como métrica oficial junto al test interno | bajo | medir lo que importa |

El test interno (macro-F1 0.877) **no predice** el rendimiento en campo: hay que adoptar una métrica de campo (punto 8).
