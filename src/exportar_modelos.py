"""
--------------------------------------
Prepara la carpeta de modelos que usa ``clasificador_diatomeas`` (spec 014) y que se entrega aparte:

    <salida>/
      manifest.json             versión, peso de DINOv2 en la combinación, parámetros de YOLO y sha256 de cada fichero
      clases.json               las 77 clases de DINOv2 (orden del entrenamiento) y las 78 de ResNet
      yolo.pt  dinov2.pth  resnet50.pth
      dinov2_config/            config.json y preprocessor_config.json de facebook/dinov2-base (para no descargar nada)

No sobrescribe una carpeta que ya exista.

Uso:
    python src/exportar_modelos.py --dino modelos/77_campo/modelo_77_campo.pth --salida dist/modelos
--------------------------------------
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

from embeddings import resolver_modelo_dinov2

RAIZ = Path(__file__).resolve().parent.parent
YOLO = RAIZ / "Inferir" / "yolo_dinov2" / "yolo_best.pt"
RESNET = RAIZ / "Inferir" / "yolo_dinov2" / "resnet50_checkpoint_epoch50.pth"
CLASES_DINO = RAIZ / "Inferir" / "txt_classes" / "classes_77(dino).txt"
CLASES_RESNET = RAIZ / "Inferir" / "txt_classes" / "classes_78(resnet).txt"


def leer_lista(ruta: Path) -> list[str]:
    return [linea.strip() for linea in ruta.read_text(encoding="utf-8").splitlines() if linea.strip()]


def clases_dino(pesos: Path) -> list[str]:
    """Las del metadatos_modelo.json del entrenamiento; si no está, la lista de Inferir/txt_classes."""
    metadatos = pesos.parent / "metadatos_modelo.json"
    if metadatos.is_file():
        return json.loads(metadatos.read_text(encoding="utf-8"))["especies_filtradas"]
    return leer_lista(CLASES_DINO)


def sha256(ruta: Path) -> str:
    resumen = hashlib.sha256()
    with ruta.open("rb") as fichero:
        for bloque in iter(lambda: fichero.read(1 << 20), b""):
            resumen.update(bloque)
    return resumen.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Carpeta de modelos para clasificador_diatomeas.")
    parser.add_argument("--dino", type=Path, default=RAIZ / "modelos" / "77_campo" / "modelo_77_campo.pth")
    parser.add_argument("--salida", type=Path, default=RAIZ / "dist" / "modelos")
    parser.add_argument("--peso-dino", type=float, default=1.0,
                        help="peso de DINOv2 en la combinación con ResNet (1 = DINOv2 solo; spec 013)")
    parser.add_argument("--solo-cajas-diatom", action="store_true", help="descartar las cajas Not_diatom de YOLO")
    args = parser.parse_args()

    if args.salida.exists():
        raise SystemExit(f"{args.salida} ya existe: bórrala o elige otra --salida.")
    if not args.dino.is_file():
        raise SystemExit(f"No existe {args.dino}")
    carpeta_hf, local = resolver_modelo_dinov2()
    if not local:
        raise SystemExit("No está facebook/dinov2-base en la caché de Hugging Face (o en DINOV2_MODEL_PATH).")

    (args.salida / "dinov2_config").mkdir(parents=True)
    copias = {"yolo.pt": YOLO, "dinov2.pth": args.dino, "resnet50.pth": RESNET,
              "dinov2_config/config.json": Path(carpeta_hf) / "config.json",
              "dinov2_config/preprocessor_config.json": Path(carpeta_hf) / "preprocessor_config.json"}
    for destino, origen in copias.items():
        print(f"{origen} -> {destino}")
        shutil.copy2(origen, args.salida / destino)

    clases = {"dino": clases_dino(args.dino), "resnet": leer_lista(CLASES_RESNET)}
    (args.salida / "clases.json").write_text(json.dumps(clases, ensure_ascii=False, indent=1), encoding="utf-8")
    manifiesto = {
        "version_modelos": f"{args.dino.parent.name}_{date.today():%Y%m%d}",
        "peso_dino": args.peso_dino,
        "solo_cajas_diatom": args.solo_cajas_diatom,
        "yolo": {"confianza": 0.30, "tamano": 1024},
        "ficheros": {nombre: sha256(args.salida / nombre) for nombre in [*copias, "clases.json"]},
    }
    (args.salida / "manifest.json").write_text(json.dumps(manifiesto, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Modelos en {args.salida} ({manifiesto['version_modelos']}, peso DINOv2 {args.peso_dino})")


if __name__ == "__main__":
    main()
