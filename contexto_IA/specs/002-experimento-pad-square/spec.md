# SPEC 002 — Experimento: center-crop vs pad-square (DINOv2)

- **Estado:** Completada (experimento). Siguiente paso: spec 003 = reentrenar 75_objetivo oficial con pad-square.
- **Fecha:** 2026-10-06
- **Relacionado:** deuda B4, D-011, `13_diagnostico_dinov2_campo.md` E2.

## Hipótesis
El center crop 224 del `AutoImageProcessor` corta los extremos de las diatomeas alargadas; pad-square (D-011) lo evita.

## Método (comparación justa)
- Mismos splits de 75_objetivo, misma semilla (42), hiperparámetros de `constantes.py`, mismo MLP y sampler.
- **Sin augmentation** en ambas variantes (solo embeddings originales).
- center-crop: embeddings originales extraídos de `embeddings_train.pt` (índices reconstruidos con
  `calcular_copias_extra_por_especie`; 95 754 = esperado) + val/test oficiales.
- pad-square: embeddings recalculados con `preparar_para_dinov2` (`scripts/emb_pad.py`, carga con hilos + freno térmico).
- Entrenamiento y evaluación: `scripts/comparar.py`. Relación de aspecto = lado mayor / menor de la imagen original de test.
- Los embeddings (~0.5 GB) quedaron en el scratch de Windows, **no versionados**; regenerables con los scripts (~30 min RTX 3050).

## Resultados (test interno, 9 854 imágenes)
| Métrica | center-crop | pad-square |
|---|---|---|
| Accuracy | 0.8809 | **0.9029** |
| Macro-F1 | 0.8567 | **0.8865** |
| Acc relación 1-1.5 (n=6063) | 0.886 | **0.898** |
| Acc relación 1.5-2 (n=2271) | 0.885 | **0.908** |
| Acc relación 2-3 (n=1323) | 0.865 | **0.918** |
| Acc relación 3-5 (n=191) | 0.785 | **0.869** |
| Acc relación 5-1e+09 (n=6) | 1.000 | **1.000** |

Especies que más mejoran (F1):
| Especie | center | pad | Δ |
|---|---|---|---|
| Reimeria_sp | 0.000 | 0.364 | +0.364 |
| Encyonopsis_fp | 0.667 | 1.000 | +0.333 |
| Nitzschia_fp | 0.286 | 0.500 | +0.214 |
| Tabellaria_flocculosa | 0.593 | 0.750 | +0.157 |
| Tabellaria_fp | 0.375 | 0.462 | +0.087 |
| Navicula_cryptotenella | 0.822 | 0.905 | +0.083 |
| Fistulifera_saprophila | 0.531 | 0.612 | +0.081 |
| Achnanthidium_pyrenaicum | 0.747 | 0.825 | +0.078 |

Especies que empeoran:
| Especie | center | pad | Δ |
|---|---|---|---|
| Fragments | 0.900 | 0.800 | -0.100 |
| Nitzschia_desertorum | 0.960 | 0.909 | -0.051 |
| Stephanodiscus_hantzschii | 0.816 | 0.773 | -0.042 |
| Fragilaria_perminuta | 0.984 | 0.952 | -0.031 |
| Aulacoseira_granulata | 0.977 | 0.947 | -0.030 |

## Conclusión
Pad-square mejora +2.2 pts accuracy y **+3.0 pts macro-F1**, y la ganancia crece con la elongación (+8.4 pts en relación 3–5).
Hipótesis confirmada. Referencia: el modelo oficial actual (con augmentation) tiene macro-F1 0.877; pad **sin** augmentation ya lo supera (0.887).
Pendiente: medir en **campo** (requiere imágenes originales y relanzar la inferencia con pesos pad).
