# TAREAS 001 — Evaluador de campo

- [x] T1 — `src/evaluar_campo.py`: carga de los Excel por recorte, alias de clases ResNet, reglas voto/mayor/suma_conf, exclusión de `_fp`, ensamble, nivel género.
- [x] T2 — Reproducir el cruce manual (CA1).
- [x] T3 — `tests/test_evaluar_campo.py` (4 tests) en verde.
- [x] T4 — Actualizar contexto_IA (03, 10 D-015, 11, 12, 13, specs/README).

## Resultado / notas de cierre
Ejecución: `python src/evaluar_campo.py --dino C:/VISILAB/classification_results_dinov2.xlsx --resnet C:/VISILAB/classification_results_resnet.xlsx --etiquetas C:/VISILAB/cruce_ground_truth.xlsx --salida informe_campo.xlsx`

| Regla (8 234 imgs) | DINOv2 | ResNet50 | Ensamble |
|---|---|---|---|
| voto (con `_fp`) | 0.659 | 0.750 | 0.750 |
| suma_conf (con `_fp`) | 0.675 | 0.762 | 0.767 |
| voto **sin `_fp`** | 0.722 | 0.815 | 0.824 |
| **suma_conf sin `_fp`** | **0.741** | **0.827** | **0.844** |

Diferencias respecto al cruce manual (justificadas):
- Voto: el manual desempataba por orden de aparición (0.6406/0.7151, reproducido exactamente con ese criterio);
  aquí se desempata por suma de confianza (+1.8 pts DINO).
- ResNet suma_conf 0.7623 vs 0.7603 manual: aquí se mapean las dos erratas de clase (`Denticula_tenius`, `Nitzschia_dessertorum`);
  sin mapear da 0.7452.
- mayor y suma_conf DINO: idénticos (0.4752/0.5211, 0.6746).

Estado: **Completada**.
