from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
rutas = [
("Navicula_cryptotenella", "/home/visilab/Escritorio/david/transformer_visilab/data/imagenes_visilab(raw)/UDE_Diatoms_84k_normalizadas_reinhard/Navicula_cryptotenella/HF_Tisa_000449a_20220607_Naphrax_60x.x_18820.y_32374.png"),
    ("Navicula_sp", "/home/visilab/Escritorio/david/transformer_visilab/data/imagenes_visilab(raw)/UDE_Diatoms_84k_normalizadas_reinhard/Navicula_sp/Israel_IADA_S01_20220501.x_14746.y_39723.png"),
    ("Navicula_sp", "/home/visilab/Escritorio/david/transformer_visilab/data/imagenes_visilab(raw)/UDE_Diatoms_84k_normalizadas_reinhard/Navicula_sp/DV_Sava_SL1_489a_20230418.x_2415.y_2398.png"),
]

TAM_MINIATURA = (300, 300)
COLUMNAS = 4

miniaturas = []
for etiqueta, ruta in rutas:
    img = Image.open(ruta).convert("RGB")
    img.thumbnail(TAM_MINIATURA)
    miniaturas.append((etiqueta, img))

filas = (len(miniaturas) + COLUMNAS - 1) // COLUMNAS
ancho_celda, alto_celda = TAM_MINIATURA[0], TAM_MINIATURA[1] + 25  # espacio para la etiqueta

hoja = Image.new("RGB", (ancho_celda * COLUMNAS, alto_celda * filas), "white")
draw = ImageDraw.Draw(hoja)

for i, (etiqueta, img) in enumerate(miniaturas):
    x = (i % COLUMNAS) * ancho_celda
    y = (i // COLUMNAS) * alto_celda
    hoja.paste(img, (x, y))
    draw.text((x + 5, y + TAM_MINIATURA[1] + 5), etiqueta, fill="black")

hoja.save("comparacion_imagenes.png")
print("Guardado en comparacion_imagenes.png")