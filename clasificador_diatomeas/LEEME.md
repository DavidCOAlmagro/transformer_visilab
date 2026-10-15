# Clasificador de diatomeas

Busca las diatomeas de una foto de microscopio y dice de qué especie es cada una (77 especies).
Para cada diatomea da la especie más probable y las tres más probables con su porcentaje.

## Qué necesitas

- La carpeta `clasificador_diatomeas` (este código).
- La carpeta `modelos` (unos 490 MB, te la paso aparte). No cambies lo que hay dentro.
- Python 3.13 (probado con 3.13.4) y las librerías de `requirements_clasificador.txt`:

```
python -m pip install -r requirements_clasificador.txt
```

Funciona solo con el procesador (CPU) y sin internet.

## Uso

Carga los modelos **una sola vez** (unos 6 s; la primera vez después de encender el ordenador puede tardar hasta 1 min)
y después pásale las fotos que quieras.
Si no quieres que la app se quede parada al arrancar, crea el `Clasificador` en un hilo aparte mientras se abre la ventana.

Foto a foto, leyendo de una carpeta:

```python
from pathlib import Path
from clasificador_diatomeas import Clasificador

clf = Clasificador("ruta/a/modelos")              # una vez, al abrir la app

resultados = []
for foto in sorted(Path("fotos").glob("*.tif")):
    resultado = clf.clasificar(foto)               # unos 3-5 s por foto
    resultados.append(resultado)
    clf.anotar(foto, resultado).save(f"anotadas/{foto.stem}.jpg")   # la foto con los recuadros (opcional)

clf.guardar_excel(resultados, "resultados.xlsx")
```

O todo de una vez (busca también en subcarpetas y escribe el Excel y las fotos anotadas):

```python
excel = clf.procesar_carpeta("fotos", "resultados",
                             al_avanzar=lambda hechas, total, foto: print(f"{hechas}/{total}"))
```

`resultado.estado` es `"ok"`, `"sin detecciones"` o `"error"` (la foto no se puede abrir; el motivo está en
`resultado.mensaje`). Cada elemento de `resultado.recortes` es una diatomea:

| Campo | Qué es |
|---|---|
| `especie` | Nombre de la especie más probable, p. ej. `Navicula gregaria` |
| `tipo` | `especie`, `vista pleural` (la diatomea se ve de lado y solo se sabe el género) o `resto` (restos o fragmentos) |
| `caja` | `(x1, y1, x2, y2)` en píxeles de la foto |
| `top3` | Las tres más probables: `[(clase, probabilidad de 0 a 1), ...]` |

Si falta algo en la carpeta de modelos, `Clasificador(...)` lanza `ErrorModelos`.

## El Excel

- **Recortes**: una fila por diatomea: foto, número, caja, especie, tipo, top1 / % top1, top2 / % top2, top3 / % top3.
- **Imagenes**: una fila por foto, con su estado (`ok`, `sin detecciones` o `error`) y el motivo si falla.
- **Info**: versión del clasificador y de los modelos, y fecha.

## Probarlo sin la app

```
python -m clasificador_diatomeas carpeta_de_fotos carpeta_de_salida --modelos ruta/a/modelos
```

## Si conviertes la app en .exe (PyInstaller)

Probado con PyInstaller 6.22. Añade estas opciones a tu comando:

```
--onedir --collect-data ultralytics --copy-metadata torch
```

- Usa `--onedir`, no `--onefile`: con `--onefile` la app descomprime unos 900 MB cada vez que arranca.
- Construye en una carpeta con una ruta corta (por ejemplo `C:\build`): Windows falla con rutas de más de
  260 caracteres.
- La carpeta `modelos` no va dentro del .exe: déjala al lado de la app y pásale su ruta a `Clasificador`.
- Si ofuscas con PyArmor, mejor deja fuera `clasificador_diatomeas`.
- Ocupa unos 900 MB la app empaquetada y 490 MB los modelos.

## Limitaciones

- Solo conoce 77 especies. Si en la foto hay otra, dirá la más parecida de esas 77 (no avisa de que no la conoce).
- Está probado con fotos de Aqualitas y DBO5. Con otro microscopio o aumento puede funcionar peor.
