# TAREAS 003

- [x] T1 — Copiar splits a `75_objetivo_pad`.
- [x] T2 — Script de ensamblado + verificación de equivalencia con `get_embedding`.
- [x] T3 — Generar `embeddings_{train,val,test}.pt` (sin unknown) de `75_objetivo_pad`.
- [x] T4 — Entrenar con `main.py` y evaluar (test, errores, confusiones).
- [x] T5 — Comparar con 75_objetivo y la spec 002 (CA2) y medir en campo si hay imágenes (CA4).
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

## Resultado en campo (2026-10-07)
Inferencia en Ubuntu con `Inferir/infer_and_split_resnet_single_folder.py --classifier both` (13 009 imágenes; las evaluables son las mismas 8 234).
`evaluar_campo.py` sobre `C:\VISILAB\classification_results.xlsx` (2026-10-07 00:01; **sobrescribió** el Excel del 2026-10-06).

| Por imagen (8 234) | DINO antes (75_objetivo) | **DINO pad (75_objetivo_pad)** | ResNet50 | Ensamble antes → ahora |
|---|---|---|---|---|
| suma_conf sin `_fp` (oficial) | 0.741 | **0.788** (+4.6) | 0.827 (sin cambios ✔) | 0.844 → **0.850** |
| voto sin `_fp` | 0.722 | 0.771 | 0.815 | 0.824 → 0.835 |
| suma_conf con `_fp` | 0.675 | 0.713 | 0.759 | 0.767 → 0.771 |

- La distancia con ResNet baja de 8.6 a **3.9 puntos**. ResNet coincide con el valor de ayer (0.8266 frente a 0.8267), lo que confirma que el cruce empareja bien.
- Por especie (`campo_por_especie.csv`): mejoran 39, se quedan igual 11 y empeoran 15. DINO gana a ResNet en 14 especies (antes 12) y pierde en 32 (antes 47).
- Mayores mejoras: N. gregaria 0.34→0.82, N. fonticola 0.48→0.90, Fistulifera 0.19→0.46, P. lanceolatum 0.47→0.74, N. desertorum 0.53→0.79.
- **Regresiones a investigar** (confusiones entre especies parecidas):
  Stephanodiscus_hantzschii 0.64→0.18 (→ Cyclotella_atomus), A. eutrophilum 0.84→0.46 (→ A. rivulare),
  Encyonopsis_minuta 0.40→0.04 (→ A. pyrenaicum), Fragilaria_vaucheriae 0.80→0.50 (→ N. fonticola).
