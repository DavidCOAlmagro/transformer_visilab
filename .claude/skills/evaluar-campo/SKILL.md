---
name: evaluar-campo
description: Evalúa en campo una inferencia nueva (Excel de Inferir/infer_and_split_resnet_single_folder.py) con src/evaluar_campo.py, la compara con las referencias vigentes y registra el resultado. Úsala cuando el usuario diga que ha terminado una inferencia o pase Excel de resultados.
---

# Evaluar una inferencia de campo

## 1. Localizar los datos (sin pisar nada)
- Excel de la inferencia: `classification_results.xlsx` (combinado, sirve para `--dino` y `--resnet`) o los `_dinov2` / `_resnet`.
  Comprueba la fecha (`ls -la --time-style=full-iso`) para saber si es una inferencia nueva o una antigua.
- Etiquetas: hoja `Por_imagen` de `C:\VISILAB\cruce_ground_truth.xlsx` (no depende del modelo).
- **Nunca** sobrescribas Excel ni informes de referencia: la salida va a una carpeta nueva o al scratch.

## 2. Ejecutar
```bash
python src/evaluar_campo.py --dino <excel> --resnet <excel> --etiquetas C:/VISILAB/cruce_ground_truth.xlsx --salida <carpeta_nueva>/informe_campo.xlsx
```
Si la inferencia solo tiene un modelo, pasa solo ese argumento.

## 3. Comprobar que el cruce es válido
- `n_imagenes` debe ser **8 234**.
- Un modelo que no ha cambiado debe dar lo mismo que en la referencia (p. ej. ResNet 0.827); si no, el cruce no
  empareja bien (rutas de la columna `image` distintas de `imagen`) → no sacar conclusiones.

## 4. Comparar con las referencias
- Métrica oficial: fila `suma_conf` con `excluye_fp = True` (D-015).
- Referencias vigentes: tabla "Resultado en campo" de `contexto_IA/12_estado_actual.md`
  (a 2026-10-07: DINO-pad 0.788, ResNet 0.827, ensamble 0.850).
- Por especie: une la hoja `Por_especie` nueva con la del informe de referencia y cuenta especies que mejoran,
  se quedan igual y empeoran (umbral ±0.02), y frente a ResNet. Para las mayores regresiones, mira en `Por_imagen` a qué se predicen ahora.

## 5. Registrar
- Tabla resumen y mejoras/regresiones en `tareas.md` de la spec correspondiente (y CSV por especie en su carpeta).
- Actualiza `contexto_IA/12_estado_actual.md` (sección del modelo y tabla de referencias si cambia la mejor).
- Si cambia el modelo por defecto, añade una decisión en `contexto_IA/10_decisiones.md`.
- Responde al usuario con la tabla, la diferencia con ResNet y el siguiente paso propuesto.
