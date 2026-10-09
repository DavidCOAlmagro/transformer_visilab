"""
Tareas del proyecto en un solo comando (Windows y Ubuntu, sin instalar nada).

    python tareas.py estado                     modelo por defecto, pesos, GPU
    python tareas.py dividir-datos              reparto train/val/test -> recursos/
    python tareas.py dividir-campo --etiquetas cruce_ground_truth.xlsx
    python tareas.py seleccionar-campo --dino X --resnet X --etiquetas X   recortes de campo -> recursos/recortes_campo.csv
    python tareas.py extraer-campo [--entrada DIR]   recorta esas cajas de las fotos -> data/.../campo_pool/
    python tareas.py entrenar [opciones]        fine-tuning (opciones de src/entrenar.py)
    python tareas.py inferir [entrada] [--prueba NOMBRE] [--classifier both|dinov2|resnet]
                                                -> Inferir/resultados_inferencia/<modelo>/<prueba>/
    python tareas.py evaluar --resultados DIR --etiquetas cruce_ground_truth.xlsx
    python tareas.py test
    python tareas.py limpiar [--si]             lista (y con --si borra) lo regenerable

En Ubuntu hay un Makefile equivalente: make estado, make inferir, make evaluar...
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PY = sys.executable
MODELO = RAIZ / "modelos" / "75_objetivo_ft" / "modelo_75_objetivo_ft.pth"
TESTS = sorted(p.stem for p in (RAIZ / "tests").glob("test_*.py"))
DATOS = RAIZ / "data"
# Regenerable u obsoleto (spec 012). Nunca: imágenes de entrenamiento, data/metadata, zips originales
# ni resultados de inferencia (ni los antiguos dentro de Inferir/imagenes_inferencia).
LIMPIABLE = [
    DATOS / "embeddings_procesado",
    DATOS / "splits",
    DATOS / "imagenes_visilab(raw)" / "dataset_aq_dbo5_agrupado.zip",
    DATOS / "imagenes_visilab(raw)" / "Seleccion_5_especies_por_especie",
    RAIZ / ".venv",
]
CARPETAS_CODIGO = ("src", "tests", "Inferir")


def ejecutar(*argumentos: object) -> int:
    comando = [str(a) for a in argumentos]
    print("->", " ".join(comando), flush=True)
    return subprocess.call(comando, cwd=RAIZ)


def tamano(ruta: Path) -> int:
    if ruta.is_file():
        return ruta.stat().st_size
    return sum(f.stat().st_size for f in ruta.rglob("*") if f.is_file())


def candidatos_limpieza() -> list[Path]:
    """Lo de LIMPIABLE que existe + pesos de otros experimentos + cachés de Python."""
    rutas = [r for r in LIMPIABLE if r.exists()]
    modelos = RAIZ / "modelos"
    if modelos.is_dir():
        rutas += [d for d in modelos.iterdir() if d.is_dir() and d.name != "75_objetivo_ft"]
    # Solo en carpetas de código: recorrer data/ entero (millones de imágenes) es lentísimo
    rutas += [p for c in CARPETAS_CODIGO for p in (RAIZ / c).rglob("__pycache__")]
    rutas += [p for p in [RAIZ / "__pycache__"] if p.exists()]
    return rutas


def limpiar(confirmar: bool) -> int:
    rutas = candidatos_limpieza()
    if not rutas:
        print("Nada que limpiar.")
        return 0
    total = 0
    for ruta in rutas:
        t = tamano(ruta)
        total += t
        print(f"{t / 1e6:10.1f} MB  {ruta.relative_to(RAIZ)}")
    print(f"{total / 1e9:10.2f} GB  en total")
    if not confirmar:
        print("Simulación: no se ha borrado nada. Para borrarlo: python tareas.py limpiar --si")
        return 0
    for ruta in rutas:
        shutil.rmtree(ruta) if ruta.is_dir() else ruta.unlink()
    print("Borrado.")
    return 0


def estado() -> int:
    print(f"Modelo por defecto: {MODELO.relative_to(RAIZ)} -> {'OK' if MODELO.is_file() else 'NO ENCONTRADO'}")
    meta = MODELO.parent / "metadatos_modelo.json"
    print(f"Metadatos (clases): {'OK' if meta.is_file() else 'NO ENCONTRADO'}")
    for nombre in ("splits_75_objetivo_relativos.txt.gz", "campo_test.txt", "campo_pool.txt"):
        print(f"recursos/{nombre}: {'OK' if (RAIZ / 'recursos' / nombre).is_file() else 'falta'}")
    try:
        import torch
        gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "sin CUDA"
        print(f"GPU: {gpu}")
    except ImportError:
        print("GPU: torch no instalado en este Python")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="tarea", required=True)
    sub.add_parser("estado")
    sub.add_parser("dividir-datos")
    p = sub.add_parser("dividir-campo")
    p.add_argument("--etiquetas", type=Path, required=True)
    sub.add_parser("seleccionar-campo", help="Acepta las opciones de src/seleccionar_recortes_campo.py")
    sub.add_parser("extraer-campo", help="Acepta las opciones de src/extraer_recortes_campo.py")
    sub.add_parser("entrenar", help="Acepta las opciones de src/entrenar.py")
    p = sub.add_parser("inferir")
    p.add_argument("entrada", nargs="?", type=Path, default=RAIZ / "Inferir" / "imagenes_inferencia")
    p.add_argument("--prueba", help="Nombre de la ejecución (por defecto, fecha y hora)")
    p.add_argument("--lista", type=Path, help="Solo estas imágenes, p. ej. recursos/campo_test.txt")
    p.add_argument("--classifier", default="both", choices=("both", "dinov2", "resnet"))
    p = sub.add_parser("evaluar")
    p.add_argument("--resultados", type=Path, required=True,
                   help="Carpeta de la inferencia (con classification_results.xlsx)")
    p.add_argument("--resnet", type=Path, help="Excel de ResNet si no está en --resultados")
    p.add_argument("--etiquetas", type=Path, required=True)
    p.add_argument("--todo", action="store_true", help="Evaluar todas las imágenes, no solo campo_test")
    p.add_argument("--lista", type=Path, default=RAIZ / "recursos" / "campo_test.txt",
                   help="Subconjunto a evaluar (p. ej. recursos/campo_test_aqualitas.txt)")
    sub.add_parser("test")
    p = sub.add_parser("limpiar")
    p.add_argument("--si", action="store_true", help="Borrar de verdad (por defecto solo lista)")
    args, extra = parser.parse_known_args()

    if args.tarea == "estado":
        return estado()
    if args.tarea == "dividir-datos":
        return ejecutar(PY, "src/dividir_datos.py", *extra)
    if args.tarea == "dividir-campo":
        return ejecutar(PY, "src/dividir_campo.py", "--etiquetas", args.etiquetas, *extra)
    if args.tarea == "seleccionar-campo":
        return ejecutar(PY, "src/seleccionar_recortes_campo.py", *extra)
    if args.tarea == "extraer-campo":
        return ejecutar(PY, "src/extraer_recortes_campo.py", *extra)
    if args.tarea == "entrenar":
        return ejecutar(PY, "src/entrenar.py", *extra)
    if args.tarea == "inferir":
        prueba = ["--run-name", args.prueba] if args.prueba else []
        prueba += ["--image-list", args.lista] if args.lista else []
        return ejecutar(PY, "Inferir/infer_and_split_resnet_single_folder.py", args.entrada,
                        "--classifier", args.classifier, *prueba, *extra)
    if args.tarea == "evaluar":
        excel = args.resultados / "classification_results.xlsx"
        comando = [PY, "src/evaluar_campo.py", "--dino", excel, "--resnet", args.resnet or excel,
                   "--etiquetas", args.etiquetas, "--salida", args.resultados / "informe_campo.xlsx"]
        if not args.todo:
            comando += ["--subconjunto", args.lista]
        return ejecutar(*comando, *extra)
    if args.tarea == "test":
        return ejecutar(PY, "-m", "unittest", *(f"tests.{t}" for t in TESTS))
    if args.tarea == "limpiar":
        return limpiar(args.si)
    return 1


if __name__ == "__main__":
    sys.exit(main())
