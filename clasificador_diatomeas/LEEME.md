# Clasificador de diatomeas

Detecta las diatomeas de una foto de microscopio y dice de qué especie es cada una (conoce 77 especies).
De cada diatomea da la especie más probable y las tres más probables con su porcentaje, y saca un Excel con todo.

## Instalación

Necesita Python 3.13 y las librerías de `requirements.txt`:

    pip install -r requirements.txt

La carpeta `modelos` (unos 490 MB) va aparte. Funciona con el procesador y no necesita internet.

## Uso

    from clasificador_diatomeas import Clasificador

    clf = Clasificador("ruta/a/modelos")   # tarda unos segundos, mejor una sola vez al abrir la app

    resultados = []
    for foto in fotos:
        resultado = clf.clasificar(foto)
        resultados.append(resultado)
        clf.anotar(foto, resultado).save("anotada.jpg")   # la foto con los recuadros, si la quieres

    clf.guardar_excel(resultados, "resultados.xlsx")

Cada `resultado` tiene un `estado` ("ok", "sin detecciones" o "error" si la foto no se puede abrir, con el motivo en
`mensaje`) y una lista `recortes` con una entrada por diatomea:

- `especie`: el nombre, por ejemplo "Navicula gregaria"
- `tipo`: "especie", "vista pleural" (la diatomea está de lado y solo se sabe el género) o "resto" (restos o fragmentos)
- `caja`: (x1, y1, x2, y2) en píxeles de la foto
- `top3`: las tres más probables, como [(clase, probabilidad), ...]

Si prefieres procesar una carpeta entera de golpe está `clf.procesar_carpeta("fotos", "resultados")`, que guarda el
Excel y las fotos anotadas. Para probarlo sin la app:

    python -m clasificador_diatomeas carpeta_fotos carpeta_resultados --modelos ruta/a/modelos

## El Excel

Tiene tres hojas: Recortes (una fila por diatomea con la especie y el top-3 con porcentajes), Imagenes (una fila por
foto con su estado) e Info (versiones y fecha).

## PyInstaller

Funciona añadiendo estas opciones:

    --onedir --collect-data ultralytics --copy-metadata torch

Mejor `--onedir` que `--onefile`, porque con `--onefile` tiene que descomprimir unos 900 MB cada vez que se abre.
Constrúyelo en una ruta corta (tipo `C:\build`), que Windows da problemas con rutas muy largas. La carpeta `modelos`
déjala fuera del exe. Si ofuscas con PyArmor, deja fuera este paquete.

## A tener en cuenta

- En un portátil normal tarda unos 3 s en buscar las diatomeas de una foto y medio segundo más por cada una, así que una
  foto con 10 diatomeas son unos 8 s. Con un procesador de 2 núcleos, más o menos el doble.
- Solo conoce 77 especies: si en la foto hay otra, te dirá la más parecida.
- Está probado con fotos de Aqualitas y DBO5. Con otro microscopio o aumento puede funcionar peor.
