# 10 — Registro de decisiones (ADR ligero)

Formato: `D-NNN | fecha | decisión | motivo | consecuencias`. Añadir al final; no editar las antiguas
(si se revierten, crear una nueva que las sustituya).

## Decisiones históricas (reconstruidas del código, notas.md y commits)

**D-001 | ~2026-07 | DINOv2 congelado + MLP sobre embeddings precalculados.**
Motivo: pocas imágenes por clase; fine-tuning/entrenar desde cero inviable y caro.
Consecuencia: cualquier cambio de preprocesado/augmentation exige recalcular `.pt`.

**D-002 | ~2026-07 | Embedding = `pooler_output` (CLS) normalizado L2.**
Motivo: CLS pensado para clasificación; norma 1 estabiliza el MLP. (Mean pooling posible alternativa.)

**D-003 | ~2026-07 | MLP 768→512→256 con ReLU + Dropout(0.3/0.2), init Xavier.**
Motivo: una sola capa lineal no separaba bien especies parecidas.

**D-004 | ~2026-07 | Split fijo 70/15/15 estratificado guardado en `.txt`, semilla 42.**
Motivo: comparabilidad entre experimentos.

**D-005 | ~2026-07 | Selección de modelo y early stopping por macro-F1 en val.**
Motivo: desbalance; la accuracy engaña.

**D-006 | ~2026-07/09 | Desbalance: sampler + pesos de loss con el mismo suavizado `1/n^EXP`, EXP=0.3; copias extra continuas (mediana/n − 1, máx 3).**
Motivo: tres mecanismos sumados sobrecorregían (p. ej. *Gomphonema pumilum* recall 0.97 / precision 0.76).

**D-007 | 2026-09 | Temperature scaling horneado en los pesos de `cabeza_especie`.**
Motivo: confianza pegada a 0.99; así cualquier consumidor del `.pth` recibe probabilidades calibradas.

**D-008 | 2026-09 | Detección de desconocidas por distancia a centroide (p95 en val).**
Estado: rendimiento pobre (16 % rechazo). Ver deuda B3.

**D-009 | 2026-09 | `UMBRAL_CONF = 0.80` para revisión manual.** Validado sobre val.

**D-010 | 2026-09/10 | Pérdida de género y Center Loss desactivadas por defecto (peso 0).**
Motivo: sin evidencia de mejora; se conservan como ablaciones.

**D-011 | 2026-10 | Preprocesado `dinov2-pad-square-v1`: padding cuadrado con color medio del borde antes del processor.**
Motivo: el center-crop del AutoImageProcessor cortaba extremos de ROIs alargados. Requiere reentrenar (pendiente, deuda B4).

**D-012 | 2026-10 | El experimento DINOv2 se fija en exactamente 77 clases; ampliaciones solo como experimento nuevo tras `auditoria_dataset.py`.**

**D-013 | 2026-10 | Inferencia de producción = YOLO obligatorio + DINOv2 y ResNet50 en el mismo comando, salida Excel.**

**D-014 | 2026-10-06 | Adoptar metodología SDD con contexto en `contexto_IA/`.**

## Nuevas decisiones

**D-015 | 2026-10-06 | En la evaluación por imagen, las predicciones `*_fp` (posición pleural), `Debris` y `Fragments` no votan; la regla oficial es `suma_conf` y el voto desempata por suma de confianza.**
Motivo: `_fp` no es una especie. Excluirlas sube DINOv2 0.675→0.741 y ResNet 0.762→0.827 (spec 001). La métrica de campo oficial es `src/evaluar_campo.py`.

<!-- D-016 | AAAA-MM-DD | ... -->
