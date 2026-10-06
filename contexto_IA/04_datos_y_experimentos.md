# 04 — Datos y experimentos

## `data/` (ignorado por git, solo local)
```text
data/
├── imagenes_visilab(raw)/
│   ├── Common_species/                         # <especie>/*.  (origen: Common species.zip)
│   ├── Unique_species/                         # (origen: Unique species.zip)
│   ├── UDE_Diatoms_84k_normalizadas_reinhard/  # UDE Diatoms in the Wild 2024, normalizado Reinhard
│   ├── dataset_aq_dbo5_agrupado/ (+ .zip)
│   ├── dataset_aq_dbo5_last_agrupado/
│   ├── Seleccion_5_especies_por_especie/       # ¡OJO! también se recorre como grupo
│   └── subcarpetas_listado.txt
├── splits/<PRUEBA>/{train,val,test,unknown}.txt
├── embeddings_procesado/<PRUEBA>/embeddings_{train,val,test,unknown}.pt
├── metadata/        # excels de ground truth, comparativa CNN vs DINOv2, enlaces
└── Resultados inferencia/{Dinov2,ResNet50}.rar
```
- `rutas_imagenes()` recorre **todas** las subcarpetas de `imagenes_visilab(raw)/` como grupos.
  Añadir una carpeta allí la mete en el siguiente split regenerado.
- La especie = nombre de carpeta; debe coincidir **exactamente** con `ESPECIES_FILTRADAS`.
- Los splits guardan **rutas absolutas de Windows** (`C:\VISILAB\...`). Mover el repo rompe splits.
  Algunos scripts `auxiliar/` tienen rutas Linux (`/home/visilab/...`) de otra máquina.
- `Imagenes_Comprimidas_original/` (raíz, no versionado): zips originales.
- Duplicados entre fuentes ya tratados con `auxiliar/duplicados.py` y `borrar_duplicados.py`
  (log en `auxiliar/log_eliminacion_duplicados.txt`).

## Tamaño del experimento 75_objetivo
| Split | Imágenes |
|---|---|
| train | 45 985 (antes de augmentation) |
| val | 9 854 |
| test | 9 854 |
| unknown | 45 259 (especies no incluidas) |

Muy desbalanceado: `Achnanthidium_minutissimum` ≈ 16 000 imágenes totales; algunas clases < 20 en test
(p. ej. `Tabellaria_fp` 7 en test). Conteos: `auxiliar/conteo_especies*.txt`.

## Experimentos (`modelos/<PRUEBA>/`)
| PRUEBA | Clases | Estado | Mejor macro-F1 test |
|---|---|---|---|
| `20_especies` | 20 | histórico (incompatible con `modelo.py` actual, que exige 77) | ≈ 0.83 |
| `UMBRAL` | ? | histórico, pruebas de umbral/confianza | — |
| `75_objetivo` | **77** | baseline (center-crop), `PRUEBA` por defecto | 0.877 (2026-09-24) |
| `75_objetivo_pad` | **77** | **mejor** (pad-square, spec 003); mismos splits; sin `unknown` | **0.893** (2026-10-06) |

Cada carpeta contiene: `metadatos_modelo.json`, `resumen_entrenamiento.json` (histórico de corridas),
`reporte_test*.txt`, `confusiones*.txt`, `errores_a_revisar*.txt`, PNGs, `config_modelo_*.json`
y el `.pth` (no versionado). Hay ficheros con y sin sufijo `_<PRUEBA>` (versiones antiguas vs nuevas).

## Reglas para experimentos
1. Un cambio de especies ⇒ **nueva PRUEBA** (`--prueba nombre`). `verificar_especies_consistentes` lo exige.
2. Cambiar preprocesado o augmentation ⇒ **recalcular embeddings + reentrenar**.
3. Para comparar modelos, **no regenerar splits** (`--regenerar-splits n`).
4. Antes de ampliar clases: `python3 src/auditoria_dataset.py "data/imagenes_visilab(raw)" --clases-activas modelos/75_objetivo/metadatos_modelo.json --minimo 20`.
5. Registrar resultado relevante en `10_decisiones.md` o en la spec correspondiente.

## Especies excluidas (y por qué) — de `constantes.py`
- `Planothidium_fp` (3 imgs), `Eunotia exigua` (6), `Fragilaria radians` (3),
  `Achnanthidium rostropyrenaicum` (pocas), `Achnanthidium delmontii` (malos resultados),
  `Gomphonema pumilum var. rigidum` (no existe en dataset).
