# 12 — Estado actual y trabajo en curso (documento VIVO)

> Actualizar este documento al cerrar cada sesión o spec. Lo más reciente arriba.
> Última actualización: **2026-10-06** (noche) — spec 003 completada en test interno; siguiente: campo en Ubuntu.

## Foco actual del usuario
1. **Construir un dataset con ground truth** a partir de inferencias en campo (Aqualitas + DBO5 GT),
   usando el consenso DINOv2/ResNet50 + etiqueta como filtro y revisión manual del resto.
   Objetivo: **entrenar con nuevas especies** (36 especies sin cobertura, ver abajo) y con datos del dominio real.
2. **Evaluar la inferencia en ambiente no controlado** (imágenes completas → YOLO → clasificadores).
3. **Subir el rendimiento de DINOv2**, que hoy rinde **peor que ResNet50** en campo.

## Ficheros de trabajo (fuera del repo, en `C:\VISILAB\`)
| Fichero | Contenido |
|---|---|
| `classification_results_dinov2.xlsx` / `_resnet.xlsx` / `classification_results.xlsx` | Inferencia 2026-10-06 04:0x, 128 621 recortes YOLO, 12 857 imágenes. Generado con `Inferir/infer_and_split_resnet_single_folder.py` (versión **sin** pad-square, con L2-norm → coherente con los pesos actuales). Clave de recorte: `image + item + x1 y1 x2 y2`. Columnas top-1..3 + confianza. |
| `cruce_ground_truth.xlsx` | Cruce DINOv2 vs ResNet50 vs etiqueta (hojas: Resumen, Cruce_top1_top3, Por_imagen, Por_especie, Sin_cobertura, Confusiones). |
| `contexto.md` | Contexto previo escrito para otro agente: criterios para construir el ground truth (ver abajo). |
| `Normalizacion reinhard.rar`, `“UDE DIATOMS in the Wild 2024”.pdf`, `Documentacion_transformers&diatomeas/` | Material de referencia. |

## Resultados de campo (cruce_ground_truth.xlsx)
- 12 857 imágenes (12 498 Aqualitas, 359 DBO5 GT); 8 234 evaluables, 4 623 con especie fuera de las clases (nivel E).
- Niveles por imagen (voto de recortes, sin Debris/Fragments): **A** ambos+etiqueta 58 %, **B** modelos coinciden
  pero la etiqueta no 11 %, **C** solo uno coincide 20 %, **D** ninguno 11 %.
- **Accuracy por imagen (evaluables)**:

| Regla de agregación | DINOv2 | ResNet50 |
|---|---|---|
| Voto mayoritario de recortes | 0.641 | 0.715 |
| Recorte más grande | 0.475 | 0.521 |
| **Suma de confianza por especie** | **0.675** | **0.760** |

- Etiqueta dentro del top-3 de ambos modelos: 79.8 %.
- Peores especies DINO vs ResNet: `Fistulifera_saprophila` 0.10 vs 0.90, `Navicula_gregaria` 0.17 vs 0.64,
  `Rhoicosphenia_abbreviata` 0.11 vs 0.53, `Planothidium_lanceolatum` 0.52 vs 0.89,
  `Nitzschia_soratensis` 0.32 vs 0.67, `Nitzschia_fonticola` 0.44 vs 0.80, `Encyonema_silesiacum` 0.69 vs 0.94.
- Especies donde DINO gana: `Nitzschia_amphibia` (+0.34), `Tabellaria_flocculosa` (+0.20), `Diatoma_moniliformis` (+0.15), `Cymbella_parva` (+0.12).
- Confusiones dominantes (ambos modelos): especie → `<Genero>_fp` (`Gomphonema_rhombicum → Gomphonema_fp` 166 imgs con DINO).
- **36 especies sin cobertura** (mayores: `Cymbella_excisa_var_excisa` 438, `Nitzschia_umbonata` 424,
  `Fragilaria_arcus` 343, `Epithemia_turgida` 341, `Epithemia_sorex` 321, `Achnanthes_subhudsonis` 207…)
  → candidatas para el nuevo experimento con ground truth.

### Métrica de campo oficial (spec 001, `src/evaluar_campo.py`) — sustituye a la tabla anterior
`_fp` = **posición pleural** → no vota (D-015).

| suma_conf por imagen | DINOv2 | ResNet50 | Ensamble |
|---|---|---|---|
| con `_fp` votando | 0.675 | 0.762 | 0.767 |
| **sin `_fp`** (oficial) | **0.741** | **0.827** | **0.844** |

DINO pierde en 47 especies, gana en 12 (Nitzschia_amphibia, Tabellaria_flocculosa, Encyonopsis_minuta, Cymbella_excisiformis…).

Diagnóstico detallado y causas probables en **`13_diagnostico_dinov2_campo.md`**.

## Resultado en campo del modelo nuevo (2026-10-07) — spec 003 cerrada
| suma_conf sin `_fp` (8 234 imgs) | DINO antes | **DINO pad** | ResNet50 | Ensamble |
|---|---|---|---|---|
| accuracy por imagen | 0.741 | **0.788** | 0.827 | **0.850** |
DINO queda a 3.9 pts de ResNet (antes 8.6). Gana a ResNet en 14 especies y pierde en 32. Regresiones: Stephanodiscus_hantzschii → Cyclotella_atomus,
A. eutrophilum → A. rivulare, Encyonopsis_minuta → A. pyrenaicum. Detalle en `specs/003-reentrenar-pad-square/tareas.md`.
⚠️ Los Excel de `C:\VISILAB\classification_results*.xlsx` son ahora los del modelo pad (los de ayer se sobrescribieron;
sus métricas siguen en las specs 001/003 y en `cruce_ground_truth.xlsx`).

## Modelo nuevo: `75_objetivo_pad` (spec 003, 2026-10-06)
Pipeline completo con pad-square, mismos splits. Test interno: macro-F1 **0.893** (oficial 0.877), acc 0.910, top-3 0.985.
Pesos: `modelos/75_objetivo_pad/modelo_75_objetivo_pad.pth`. Medido en campo ✔ (ver arriba). Para usarlo en otra máquina, copiar los pesos
(relanzar `Inferir/infer_and_split_resnet_single_folder.py --classifier both`, que ya usa estos pesos por defecto, y después `src/evaluar_campo.py`; baseline de campo DINO 0.741, ResNet 0.827).
Embeddings en `data/embeddings_procesado/75_objetivo_pad/` (sin `unknown`). El modelo se ha entrenado en Windows y los pesos no están en git: copiarlos a Ubuntu.

## Experimentos en curso
- **EXP-pad (2026-10-06) — COMPLETADO → spec 002.** Sin augmentation, test interno: pad-square macro-F1 **0.887** vs
  center-crop 0.857 (+3.0), accuracy 0.903 vs 0.881; en imágenes alargadas (relación 3–5) 0.869 vs 0.785.
  Pad sin augmentation ya supera al modelo oficial (0.877). **Siguiente: spec 003 = reentrenar 75_objetivo con pad-square**
  (con augmentation) y medir en campo con `evaluar_campo.py` (requiere las imágenes originales de campo).

## Criterios acordados para el ground truth (de `C:\VISILAB\contexto.md`)
- Unidad de cruce = **recorte**, alineado por clave `image+item+x1+y1+x2+y2`, nunca por orden de filas.
- El nombre de archivo es una etiqueta de referencia, no un ground truth confirmado.
- El consenso entre modelos es una señal, **no** una prueba (pueden compartir sesgo).
- A revisión manual: conflictos, predicciones genéricas (`Debris`, `Fragments`, `*_fp`, `sp`, `cf`), baja confianza.
- Estados finales: `aceptado`, `rechazado`, `revisar`, `sin_datos`; registrar quién revisó y con qué evidencia.
- No inventar top-3 ni probabilidades; no sobrescribir datos originales.

## Avisos vigentes
- La inferencia es **`Inferir/infer_and_split_resnet_single_folder.py`** (`src/inferencia.py` eliminado en la spec 004).
- ✅ (resuelto en spec 004) ~~HEAD aplica pad-square en inferencia, pero los pesos de `75_objetivo` (2026-09-24)
  se entrenaron con resize + center crop.** No usar la inferencia actual con esos pesos
  hasta reentrenar.~~ La inferencia usa por defecto `75_objetivo_pad`, coherente con pad-square.
- Las clases de ResNet tienen erratas (`Denticula_tenius`, `Nitzschia_dessertorum`) y una clase extra `Planothidium_fp`;
  mapear antes de cruzar.

## Próximos pasos propuestos (pendientes de que el usuario los apruebe)
Ver ranking en `13_diagnostico_dinov2_campo.md` §4 y specs candidatas en `specs/README.md`.

## Preguntas abiertas al usuario
- Imágenes originales de campo (Aqualitas/DBO5) no están en esta máquina: hacen falta para probar cambios de preprocesado en campo.
- ResNet50: **el usuario no sabe cómo se entrenó** (2026-10-06). Checkpoint no disponible en local (`Inferir/yolo_dinov2/` no existe).
- ¿Las imágenes de inferencia (`diatomeas_infer(norm1)`, Aqualitas) están normalizadas con Reinhard?
- `_fp` = posición pleural ✔. Pendiente: ¿mantener como clase o tratar como “vista pleural + género”?
- ¿Formato/ubicación previstos para el dataset de ground truth (carpeta nueva en `imagenes_visilab(raw)`?).
