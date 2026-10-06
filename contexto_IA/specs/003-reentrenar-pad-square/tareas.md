# TAREAS 003

- [x] T1 — Copiar splits a `75_objetivo_pad`.
- [x] T2 — Script de ensamblado + verificación de equivalencia con `get_embedding`.
- [x] T3 — Generar `embeddings_{train,val,test}.pt` (sin unknown) de `75_objetivo_pad`.
- [x] T4 — Entrenar con `main.py` y evaluar (test, errores, confusiones).
- [~] T5 — (test interno hecho; campo pendiente en Ubuntu) Comparar con 75_objetivo y la spec 002 (CA2) y medir en campo si hay imágenes (CA4).
- [x] T6 — Actualizar contexto_IA.

## Resultado / notas de cierre
- Verificación de equivalencia scratch ↔ `get_embedding`: diferencia máxima 3.8e-07 ✔. train = 45 985 originales + 49 769 aumentadas = 95 754 (= oficial).
- Entrenamiento: mejor época 22, macro-F1 val 0.894, early stopping en la 29, temperatura 0.846.

| Test interno (9 854) | 75_objetivo (oficial) | 002 pad sin aug | **75_objetivo_pad** |
|---|---|---|---|
| Accuracy | 0.891 | 0.903 | **0.910** |
| Macro-F1 | 0.877 | 0.887 | **0.893** |
| Top-3 | 0.977 | — | **0.985** |

- Confusiones (≥5) frente al oficial: N. inconspicua→soratensis 68→48, P. frequentissimum→lanceolatum 50→39,
  Cyclotella→Discostella 27→18. Nueva cabecera: A. minutissimum→A. pyrenaicum (64) → revisar.
- `accuracy_genero_test` = 0.083: la cabeza de género no se entrena (PESO_GENERO=0, D-010); la métrica no tiene sentido (deuda).
- La aleatoriedad de las vistas aumentadas no es bit-reproducible (hilos), aunque la semilla está fijada.
- Inferencia con el modelo nuevo: `--dino-weights modelos/75_objetivo_pad/modelo_75_objetivo_pad.pth` (las rutas por defecto siguen apuntando a 75_objetivo).
