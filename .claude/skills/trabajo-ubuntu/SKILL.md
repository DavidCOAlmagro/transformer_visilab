---
name: trabajo-ubuntu
description: Prepara un trabajo (entrenamiento, inferencia, script largo) para que el usuario lo lance a mano en el Ubuntu del trabajo (Quadro P4000) por AnyDesk, y le da los comandos listos. Úsala siempre que haya que ejecutar algo en Ubuntu.
---

# Trabajo para el Ubuntu del trabajo

**Contexto:** `visilab@visilab-Precision-Tower-7810`, repo en `~/Escritorio/david/transformer_visilab`. GPU **Quadro P4000**
(8 GB, Pascal 6.1). El usuario entra por AnyDesk y **no quiere instalar nada** allí (ni Claude Code, ni SSH, ni `pip`):
yo no tengo acceso; preparo el script aquí, lo subo con git y él lo lanza y me pega la salida.

## Requisitos del script
- **Rutas relativas**: los splits de `data/splits/` tienen rutas absolutas de Windows. Usa la lista
  `contexto_IA/specs/006-finetuning-parcial-dinov2/splits_75_objetivo_relativos.txt.gz` (`split<TAB>ruta bajo imagenes_visilab(raw)`).
- **Comprobar las imágenes al empezar** y abortar si falta > 1 % (mostrando ejemplos). En Ubuntu falta `Seleccion_5_especies_por_especie` (127 imágenes): se omiten.
- **Reanudable** (checkpoint por época) y que imprima progreso legible (una línea cada ~100 pasos y una por época).
- **Precisión según la GPU**: bf16 si capacidad ≥ 8, fp16 si 7.x, **fp32 en Pascal** (la P4000). Batch 32 cabe en 8 GB.
- Opción para desactivar cualquier freno térmico: el usuario lo quiere **sin freno** en esa torre.
- Probar antes en Windows (prueba de humo con pocas imágenes) para no gastar horas de la torre en un fallo.
- Al terminar, una línea final clara para pegar (métricas + ruta de lo generado). Salidas en carpetas nuevas; no sobrescribir referencias.

## Comandos que doy (uno por bloque `bash`)
1. Actualizar: si `git status` muestra cambios, `git stash push -m "cambios locales ubuntu"` → `git fetch && git switch <rama>` (o `git pull`) → `git stash pop`.
2. Comprobar que no hay otros procesos en la GPU: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`; parar solo **por PID**.
3. Lanzar en **primer plano** con log (no `nohup`): `python3 <script> ... 2>&1 | tee log_<nombre>.txt`.
   Recordar: no cerrar la terminal ni pulsar Ctrl+C; si se corta, relanzar el mismo comando.
4. Decir exactamente **qué líneas pegar de vuelta** y cuánto tardará (estimación por paso o época).

## Ficheros que hay que llevar a mano (no están en git)
Pesos `.pth` (p. ej. `modelos/75_objetivo_pad/modelo_75_objetivo_pad.pth`), carpetas de `data/` que falten y Excel de `C:\VISILAB\`.
Para carpetas grandes, preparar un zip con la ruta relativa dentro (`tar.exe -a -cf ...zip <ruta>`) y dar el `unzip -d "data/imagenes_visilab(raw)"`.
