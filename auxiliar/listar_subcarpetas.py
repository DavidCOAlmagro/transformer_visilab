from pathlib import Path


def obtener_subcarpetas(ruta: Path) -> list[str]:
    if not ruta.exists():
        print(f"[AVISO] No existe la carpeta: {ruta}")
        return []
    if not ruta.is_dir():
        print(f"[AVISO] La ruta no es una carpeta: {ruta}")
        return []

    return sorted([p.name for p in ruta.iterdir() if p.is_dir()])


def main() -> None:
    base = Path(__file__).resolve().parent.parent / "imagenes_visilab"

    carpetas_objetivo = [
        base / "Common_Species", base / "Unique_Species"
    ]

    salida = base / "subcarpetas_listado.txt"

    lineas: list[str] = []

    for carpeta in carpetas_objetivo:
        titulo = f"=== {carpeta.name} ==="
        print(titulo)
        lineas.append(titulo)

        subcarpetas = obtener_subcarpetas(carpeta)
        if not subcarpetas:
            print("(sin subcarpetas o carpeta no valida)")
            lineas.append("(sin subcarpetas o carpeta no valida)")
        else:
            for nombre in subcarpetas:
                print(nombre)
                lineas.append(nombre)

        print()
        lineas.append("")

    salida.write_text("\n".join(lineas), encoding="utf-8")
    print(f"Listado guardado en: {salida}")


if __name__ == "__main__":
    main()
