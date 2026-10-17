# Clasificador de diatomeas

Le pasas una foto del microscopio y te dice qué diatomeas hay y de qué especie es cada una.

## Qué hace falta

- Python 3.13 y las librerías: `pip install -r requirements.txt`
- La carpeta `modelos` (`yolo.pt`, `dinov2.pth` y `clases.txt`)

## Cómo se usa

```python
from clasificador_diatomeas import Clasificador

clf = Clasificador("modelos")   # carga los modelos: hazlo una sola vez, al abrir la app

resultados = []
for foto in fotos:
    resultado = clf.clasificar(foto)
    resultados.append(resultado)
    clf.anotar(foto, resultado).save("anotada.jpg")   # opcional: la foto con los recuadros

clf.guardar_excel(resultados, "resultados.xlsx")
```

- `resultado.estado` es `"ok"`, `"sin detecciones"` o `"error"` (si la foto no se puede abrir).
- `resultado.recortes` tiene una entrada por diatomea con su `especie`, su `tipo` (especie, vista pleural o resto),
  su `caja` (x1, y1, x2, y2) y `top3`: las tres especies más probables con su porcentaje.

Para probarlo sin la app:

```
python -m clasificador_diatomeas carpeta_fotos carpeta_resultados --modelos modelos
```

## Si haces un .exe con PyInstaller

Añade `--onedir --collect-data ultralytics --copy-metadata torch`, constrúyelo en una ruta corta (como `C:\build`)
y deja la carpeta `modelos` fuera del exe. Si ofuscas con PyArmor, deja fuera este paquete.

## A tener en cuenta

- Va con el procesador: unos 8 segundos por foto con 10 diatomeas.
- Solo conoce 77 especies. Si en la foto hay otra, dirá la más parecida.
