"""Audita distribución, duplicados y posibles clases adicionales del dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

EXTENSIONES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def auditar_dataset(
    raiz: Path,
    clases_activas: set[str] | None = None,
    minimo_recomendado: int = 20,
) -> dict[str, object]:
    if not raiz.is_dir():
        raise FileNotFoundError(f"No existe la raíz del dataset: {raiz}")
    archivos = sorted(
        p for p in raiz.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONES
    )
    por_clase: Counter[str] = Counter()
    por_grupo: Counter[str] = Counter()
    hashes: defaultdict[str, list[str]] = defaultdict(list)
    for ruta in archivos:
        clase = ruta.parent.name
        por_clase[clase] += 1
        relativo = ruta.relative_to(raiz)
        if len(relativo.parts) >= 3:
            por_grupo[relativo.parts[0]] += 1
        digest = hashlib.sha256(ruta.read_bytes()).hexdigest()
        hashes[digest].append(str(relativo))
    duplicados = {digest: rutas for digest, rutas in hashes.items() if len(rutas) > 1}
    activas = sorted(clases_activas or set(por_clase))
    candidatas = sorted(
        clase for clase, cantidad in por_clase.items()
        if clase not in activas and cantidad >= minimo_recomendado
    )
    cantidades = list(por_clase.values())
    return {
        "raiz": str(raiz),
        "total_imagenes": len(archivos),
        "clases_activas": activas,
        "num_clases_activas": len(activas),
        "distribucion_clases": dict(sorted(por_clase.items())),
        "min_clase": min(cantidades) if cantidades else 0,
        "mediana_clase": median(cantidades) if cantidades else 0,
        "max_clase": max(cantidades) if cantidades else 0,
        "clases_con_pocos_ejemplos": sorted(
            clase for clase, cantidad in por_clase.items() if cantidad < minimo_recomendado
        ),
        "duplicados_exactos": duplicados,
        "num_grupos_inferidos": len(por_grupo),
        "imagenes_por_grupo": dict(sorted(por_grupo.items())),
        "clases_adicionales_recomendadas": candidatas,
        "recomendacion": (
            "Revisar manualmente las clases adicionales candidatas; no se incorporan "
            "automáticamente al experimento de 77 clases."
            if candidatas else
            "No hay clases adicionales con suficientes imágenes para recomendarlas."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raiz", type=Path, nargs="?", default=Path("data/imagenes_visilab(raw)"))
    parser.add_argument("--clases-activas", type=Path, help="JSON con especies_filtradas.")
    parser.add_argument("--minimo", type=int, default=20)
    parser.add_argument("--salida", type=Path)
    args = parser.parse_args()
    activas = None
    if args.clases_activas:
        datos = json.loads(args.clases_activas.read_text(encoding="utf-8"))
        activas = set(datos.get("especies_filtradas", datos.get("especies", [])))
    resultado = auditar_dataset(args.raiz, activas, args.minimo)
    texto = json.dumps(resultado, indent=2, ensure_ascii=False)
    print(texto)
    if args.salida:
        args.salida.write_text(texto + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
