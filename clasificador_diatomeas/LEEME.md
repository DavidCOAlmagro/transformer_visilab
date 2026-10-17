# Clasificador de diatomeas

Le pasas una foto del microscopio, recorta las diatomeas y dice de qué especie es cada una.

## Qué hace falta

- Python 3.13 y las librerías: `pip install -r requirements.txt`
- La carpeta `modelos` (`yolo.pt`, `dinov2.pth` y `clases.txt`)

## Cómo se usa

```python
from pathlib import Path
from clasificador_diatomeas import Clasificador

clf = Clasificador("modelos")   # carga los modelos: hazlo una sola vez, al abrir la app

fotos = sorted(Path("carpeta_fotos").glob("*.tif"))
resultados = []
for foto in fotos:
    resultado = clf.clasificar(foto)
    resultados.append(resultado)
    if resultado.estado != "error":   # opcional: guardar la foto con los recuadros
        clf.anotar(foto, resultado).save(f"anotada_{foto.stem}.jpg")

clf.guardar_excel(resultados, "resultados.xlsx")
```

- `resultado.estado` es `"ok"`, `"sin detecciones"` o `"error"` (si la foto no se puede abrir).
- `resultado.recortes` tiene una entrada por diatomea con su `especie`, su `tipo` (especie, vista pleural o resto),
  su `caja` (x1, y1, x2, y2) y `top3`: las tres especies más probables con su porcentaje.

Para probarlo sin la app:

```
python -m clasificador_diatomeas carpeta_fotos carpeta_resultados --modelos modelos
```

## Qué hay en cada fichero

Solo hace falta usar `Clasificador`. Lo demás funciona por dentro y no hay que tocarlo.

- `clasificador.py`: la clase `Clasificador` con `clasificar`, `anotar` y `guardar_excel`. Es lo único que usa la app.
- `fotos.py`: abre las fotos y dibuja los recuadros con el nombre de cada especie.
- `excel.py`: pone los nombres de las especies como se escriben y genera el Excel.
- `dinov2.py`: el modelo que reconoce la especie de cada diatomea.
- `__main__.py`: el comando para probarlo sin la app.

## Si haces un .exe con PyInstaller

Añade `--onedir --collect-data ultralytics --copy-metadata torch` y constrúyelo en una ruta corta (como `C:\build`),
porque Windows falla con rutas muy largas. La carpeta `modelos` va fuera del exe y, si ofuscas con PyArmor,
deja fuera este paquete.

## A tener en cuenta

- Va con el procesador: unos 8 segundos por foto con 10 diatomeas.
- Solo conoce 77 especies. Si en la foto hay otra, dirá la más parecida.
