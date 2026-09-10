"""
--------------------------------------
Inferencia sobre imágenes nuevas usando el modelo DINOv2 + clasificador entrenado.
Permite clasificar una imagen suelta o una carpeta entera y exportar los resultados a Excel.
--------------------------------------
"""

from pathlib import Path
import torch
import pandas as pd
from modelo import cargar_modelo_entrenado
from constantes import VARIABLES_GLOBALES
from embeddings import inicializar_dinov2, get_embedding


def elegir_archivo() -> str:
    try:
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
        
        root = Tk()
        root.withdraw()  # oculta la ventana principal, solo queremos el diálogo
        ruta = askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        root.destroy()
        return ruta
    
    except Exception as e:
        print(f"Error al seleccionar archivo: {e}")
        print("Asegúrate de que tkinter esté correctamente instalado.(sudo apt install python3-tk)")
        return input("Inserta la ruta de la imagen manualmente: ").strip()

def elegir_carpeta() -> str:
    try:    
        from tkinter import Tk
        from tkinter.filedialog import askdirectory
        
        root = Tk()
        root.withdraw()
        ruta = askdirectory(title="Selecciona una carpeta de imágenes")
        root.destroy()
        return ruta
    
    except Exception as e:
        print(f"Error al seleccionar carpeta: {e}")
        print("Asegúrate de que tkinter esté correctamente instalado.(sudo apt install python3-tk)")
        return input("Inserta la ruta de la carpeta manualmente: ").strip()

@torch.no_grad()
def predecir_imagen(embedding: torch.Tensor,modelo: torch.nn.Module,
        especies_ordenadas: list[str],intervalo_confianza: float = VARIABLES_GLOBALES["UMBRAL_CONF"]) -> dict[str, object]:
    """
    Recibe el embedding de una imagen y devuelve un diccionario con:
    - especie_predicha: la clase con mayor probabilidad
    - confianza: probabilidad de la clase predicha (0-100%)
    - top3: lista de tuplas [(especie, probabilidad), ...] con las 3 más probables
    - revisar: True si la confianza está por debajo del umbral (imagen dudosa)
    """
    embedding = embedding.to(VARIABLES_GLOBALES["DEVICE"])

    # logits → probabilidades con softmax
    logits_especie, _,_ = modelo(embedding)                    # [1, n_clases]
    probs: torch.Tensor = torch.softmax(logits_especie, dim=1)          # [1, n_clases]

    # especie con mayor probabilidad y su confianza
    top3_probs, top3_indices = torch.topk(probs, k=3, dim=1)

    top3 = []
    # Con zip iteramos sobre los índices y probabilidades de las 3 clases más probables
    # Indice 0 -> probabilidad 0...
    for indice, probabilidad in zip(top3_indices[0], top3_probs[0]):
        especie = especies_ordenadas[indice.item()]
        porcentaje = round(probabilidad.item() * 100, 2)
        top3.append((especie, porcentaje))
    # top3[0] =    ("Nitzschia_inconspicua", 97.3)  → la tupla entera del 1º
    # top3[0][0] = "Nitzschia_inconspicua"          → el nombre (posición 0 de la tupla)
    # top3[0][1] = 97.3                             → la confianza (posición 1 de la tupla)
    especie_predicha: str = top3[0][0]
    confianza: float = top3[0][1]

    return {
        "especie_predicha": especie_predicha,
        "confianza":        confianza,           # en %
        "top3":             top3,
        "revisar":          confianza < (intervalo_confianza * 100)
    }
def guardar_resultado_txt(ruta_imagen: Path, resultado: dict[str, object]) -> Path:
    """Guarda el resultado de una imagen suelta en un .txt junto a ella."""
    ruta_txt = ruta_imagen.with_name(f"{ruta_imagen.stem}_prediccion.txt")
    lineas = [
        f"Imagen: {ruta_imagen.name}",
        f"Especie predicha: {resultado['especie_predicha']}",
        f"Confianza: {resultado['confianza']}%",
        "Top-3:",
    ]
    for i, (especie, prob) in enumerate(resultado["top3"], start=1):
        lineas.append(f"  {i}. {especie:40s} {prob}%")
    if resultado["revisar"]:
        lineas.append("Confianza baja: revisar manualmente")

    lineas.append("")

    ruta_txt.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta_txt

def inferir_imagen_suelta(ruta_imagen: str,modelo: torch.nn.Module, especies_ordenadas: list[str],
                          procesador: torch.nn.Module,dinov2: torch.nn.Module,device: torch.device,
                          augmentation: bool) -> None:
    """
    Modo consola: clasifica una sola imagen y muestra el resultado por pantalla.
    Uso: python inferencia.py imagen.jpg
    """
    ruta = Path(ruta_imagen)
    valid:bool = True
    # Comprobamos que el archivo existe y tiene una extensión válida
    try:
        if not ruta.exists():
            valid = False
            raise ValueError(f"Archivo no encontrado: {ruta_imagen}")
        if ruta.suffix.lower() not in VARIABLES_GLOBALES["EXTENSIONES_VALIDAS"]:
            valid = False
            raise ValueError(f"Extensión inválida para {ruta_imagen}.")
    except ValueError as e:
        valid = False
        print(f"Error: {e}")

    if valid:
        print(f"\nClasificando: {ruta.name}")

        embedding = get_embedding(ruta_imagen, procesador, dinov2, device, augmentation, is_train=False)
        resultado = predecir_imagen(embedding, modelo, especies_ordenadas)

        # Mostramos el resultado por consola
        print(f"\n{'='*50}")
        print(f"  Especie predicha : {resultado['especie_predicha']}")
        print(f"  Confianza        : {resultado['confianza']}%")
        print("  Top-3:")
        for i, (especie, prob) in enumerate(resultado["top3"], start=1):
            print(f"    {i}. {especie:40s} {prob}%")
        if resultado["revisar"]:
            print("Confianza baja — revisar manualmente")
        print(f"{'='*50}\n")
        ruta_txt = guardar_resultado_txt(ruta, resultado)
        print(f"Resultado guardado en: {ruta_txt}")

def inferir_carpeta(ruta_carpeta: str, modelo: torch.nn.Module, especies_ordenadas: list[str],
                    processor: torch.nn.Module, dinov2: torch.nn.Module, device: torch.device,
                    augmentation: bool) -> None:
    """
    Clasifica todas las imágenes de una carpeta y guarda
    los resultados en un Excel dentro de la misma carpeta.
    """
    valid:bool = True
    try:
        ruta = Path(ruta_carpeta)
        imagenes: list[Path] = []
        for archivo in ruta.iterdir():
            if archivo.suffix.lower() in VARIABLES_GLOBALES["EXTENSIONES_VALIDAS"]:
                imagenes.append(archivo)
        if not imagenes:
            print(f"No se encontraron imágenes en {ruta}")
            valid = False

    except FileNotFoundError as e:
        valid = False
        print(f"Error al procesar la carpeta: {e}")
    rows: list[dict[str, object]] = []
    if valid and imagenes:

        for imagen in imagenes:
            print(f"  Procesando: {imagen.name}")

            embedding = get_embedding(
                str(imagen), processor, dinov2, device, augmentation, is_train=False
            )
            resultado = predecir_imagen(embedding, modelo, especies_ordenadas)

            rows.append({
                "imagen":           imagen.name,
                "especie_predicha": resultado["top3"][0][0],
                "confianza_%":      resultado["top3"][0][1],
                "2a_opcion":        resultado["top3"][1][0],
                "confianza_2_%":   resultado["top3"][1][1],
                "3a_opcion":        resultado["top3"][2][0],
                "confianza_3_%":   resultado["top3"][2][1],
                "revisar":          "Revision" if resultado["revisar"] else "",
            })

        # Generamos el Excel
        df = pd.DataFrame(rows)
        ruta_excel = ruta / "predicciones.xlsx"
        df.to_excel(ruta_excel, index=False)

        total_revisar = sum(1 for f in rows if f["revisar"]) # si no es vacío, se suma 1 revision
        print(f"\nExcel guardado en: {ruta_excel}")
        print(f"Imágenes a revisar: {total_revisar}/{len(imagenes)}")
    else:
        print("No se procesaron imágenes debido a errores previos.")
        
def main() -> None:
    """
    Punto de entrada por consola. Pregunta al usuario qué quiere hacer
    y ejecuta el modo correspondiente.
    """
    try:
        modelo, especies_ordenadas = cargar_modelo_entrenado()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        
    print("Cargando DINOv2 (solo una vez)...")
    processor, dinov2, device, augmentation = inicializar_dinov2()
    
    print("=== Inferencia de diatomeas ===\n")
    print("1. Clasificar una imagen suelta")
    print("2. Clasificar una carpeta entera")

    valid = True
    while valid:
        opcion = input("\nElige una opción (1/2): ").strip()
    
        if opcion == "1":
            ruta = elegir_archivo()
            if not ruta:
                print("No se seleccionó ninguna imagen.")
            else:
                print(f"Imagen seleccionada: {ruta}")
                inferir_imagen_suelta(ruta, modelo, especies_ordenadas, processor, dinov2, device, augmentation)
            valid = False
            
        elif opcion == "2":
            ruta = elegir_carpeta()
            if not ruta:
                print("No se seleccionó ninguna carpeta.")
            else:
                print(f"Carpeta seleccionada: {ruta}")
                inferir_carpeta(ruta, modelo, especies_ordenadas, processor, dinov2, device, augmentation)
            valid = False

        else:
            print("Opción no válida. Escribe 1 o 2.")


if __name__ == "__main__":
    main()
