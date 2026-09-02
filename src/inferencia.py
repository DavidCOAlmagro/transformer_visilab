"""
--------------------------------------
Inferencia sobre imágenes nuevas usando el modelo DINOv2 + clasificador entrenado.
Permite clasificar una imagen suelta o una carpeta entera y exportar los resultados a Excel.
--------------------------------------
"""

from pathlib import Path
import torch
import pandas as pd
from clasificador import ClasificadorDiatomeas
from constantes import VARIABLES_GLOBALES
from embeddings import inicializar_dinov2, get_embedding
from preparar_datos import construir_numero_genero

def cargar_modelo() -> tuple[ClasificadorDiatomeas, list[str]]:
    """
    Carga el modelo entrenado desde disco y devuelve también la lista ordenada
    de especies (índice 0 → especie 0, índice 1 → especie 1...).
    """
    especies_numero: list[str] = sorted(VARIABLES_GLOBALES["ESPECIES_FILTRADAS"])
    num_clases = len(especies_numero)
    numero_genero = construir_numero_genero(VARIABLES_GLOBALES["ESPECIES_FILTRADAS"])
    num_generos = len(numero_genero)
    modelo = ClasificadorDiatomeas(num_clases, num_generos).to(VARIABLES_GLOBALES["DEVICE"])

    ruta_pesos=VARIABLES_GLOBALES["RUTA_MODELOS"]/VARIABLES_GLOBALES["PRUEBA"] / "mejor_modelo.pth"
    if not ruta_pesos.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado en: {ruta_pesos}\n"
            "Entrena el modelo antes de ejecutar la inferencia."
        )
    
    modelo.load_state_dict(torch.load(ruta_pesos, map_location=VARIABLES_GLOBALES["DEVICE"], weights_only=True))
    modelo.eval()

    return modelo, especies_numero

def calcular_embedding_imagen(ruta_imagen: str) -> torch.Tensor:
    """
    Dado la ruta de una imagen nueva (no del dataset), inicializa DINOv2,
    extrae su embedding y lo devuelve listo para pasarlo al clasificador.
    Sin augmentation porque es inferencia, no entrenamiento.
    """

    procesador, modelo_dino, device, augmentation = inicializar_dinov2()
    embedding = get_embedding(ruta_imagen, procesador, modelo_dino, device, augmentation,
                              is_train = False)

    return embedding

@torch.no_grad()
def predecir_imagen(embedding: torch.Tensor,modelo: ClasificadorDiatomeas,
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
    logits_especie, _ = modelo(embedding)                    # [1, n_clases]
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

def inferir_imagen_suelta(ruta_imagen: str) -> None:
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

        modelo, especies_ordenadas = cargar_modelo()
        embedding = calcular_embedding_imagen(ruta_imagen)
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

def inferir_carpeta(ruta_carpeta: str) -> None:
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
        modelo, especies_ordenadas = cargar_modelo()
        processor, dinov2, device, augmentation = inicializar_dinov2()

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
    print("=== Inferencia de diatomeas ===\n")
    print("1. Clasificar una imagen suelta")
    print("2. Clasificar una carpeta entera")

    opcion = input("\nElige una opción (1/2): ").strip()

    if opcion == "1":
        ruta = input("Ruta de la imagen: ").strip()
        inferir_imagen_suelta(ruta)

    elif opcion == "2":
        ruta = input("Ruta de la carpeta: ").strip()
        inferir_carpeta(ruta)

    else:
        print("Opción no válida. Escribe 1 o 2.")


if __name__ == "__main__":
    main()
