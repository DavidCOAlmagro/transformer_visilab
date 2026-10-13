# Clasificador de diatomeas

Busca las diatomeas de una foto de microscopio y dice de qué especie es cada una (77 especies).
Para cada diatomea da la especie más probable y las tres más probables con su porcentaje.

## Qué necesitas

- La carpeta `clasificador_diatomeas` (este código).
- La carpeta `modelos` (unos 490 MB, te la paso aparte). No la cambies de contenido.
- Python 3.13 y las librerías de `requirements_clasificador.txt`:

```
python -m pip install -r requirements_clasificador.txt
```

Funciona solo con el procesador (CPU) y sin internet.

## Uso

```python
from clasificador_diatomeas import Clasificador

clf = Clasificador("ruta/a/modelos")   # tarda entre 10 s y 1 min: hazlo una vez, al abrir la app

# Una foto
resultado = clf.clasificar("foto.tif")
for diatomea in resultado.recortes:
    print(diatomea.especie, diatomea.caja, diatomea.top3)
imagen = clf.anotar("foto.tif", resultado)   # la foto con los recuadros, para mostrarla en la app

# Una carpeta entera: escribe resultados.xlsx y las fotos anotadas
excel = clf.procesar_carpeta("fotos", "resultados",
                             al_avanzar=lambda hechas, total, foto: print(f"{hechas}/{total}"))
```

`resultado.estado` es `"ok"` o `"sin detecciones"`. Cada elemento de `resultado.recortes` tiene:

| Campo | Qué es |
|---|---|
| `especie` | Nombre de la especie más probable, p. ej. `Navicula gregaria` |
| `tipo` | `especie`, `vista pleural` (la diatomea se ve de lado y solo se sabe el género) o `resto` (restos o fragmentos) |
| `caja` | `(x1, y1, x2, y2)` en píxeles de la foto |
| `top3` | Las tres más probables: `[(clase, probabilidad de 0 a 1), ...]` |

Si una foto no se puede abrir, `clasificar` lanza `ErrorEntrada`. Si falta algo en la carpeta de modelos,
`Clasificador(...)` lanza `ErrorModelos`. `procesar_carpeta` no se para por una foto mala: la apunta con estado `error`.

## El Excel (`resultados.xlsx`)

- **Recortes**: una fila por diatomea: foto, número, caja, especie, tipo, top1 / % top1, top2 / % top2, top3 / % top3.
- **Imagenes**: una fila por foto, con su estado (`ok`, `sin detecciones` o `error`) y el motivo si falla.
- **Info**: versión del clasificador y de los modelos, y fecha.

Las fotos anotadas quedan en `resultados/anotadas/`.

## Probarlo sin la app

```
python -m clasificador_diatomeas carpeta_de_fotos carpeta_de_salida --modelos ruta/a/modelos
```

## Si conviertes la app en .exe (PyInstaller)

Probado con PyInstaller 6.22. Añade estas opciones a tu comando:

```
--onedir --collect-data clasificador_diatomeas --collect-data ultralytics
--collect-submodules transformers.models.dinov2 --collect-submodules transformers.models.bit
--copy-metadata transformers --copy-metadata torch
```

- Usa `--onedir`, no `--onefile`: con `--onefile` la app descomprime unos 900 MB cada vez que arranca.
- Construye en una carpeta con una ruta corta (por ejemplo `C:\build`). Windows falla con rutas de más de
  260 caracteres y algunas de transformers son muy largas.
- La carpeta `modelos` no va dentro del .exe: déjala al lado de la app y pásale su ruta a `Clasificador`.
- Ocupa unos 900 MB la app empaquetada y 490 MB los modelos.

## Tiempos

En un portátil, unos 5-6 s por foto con 10 diatomeas. Cargar los modelos, entre 10 s y 1 min la primera vez.

## Limitaciones

- Solo conoce 77 especies. Si en la foto hay otra, dirá la más parecida de esas 77 (no avisa de que no la conoce).
- Está probado con fotos de Aqualitas y DBO5. Con otro microscopio o aumento puede funcionar peor.
