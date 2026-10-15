"""
--------------------------------------
Clasificador de especies de diatomeas en fotos de microscopio.

    from clasificador_diatomeas import Clasificador

    clf = Clasificador("modelos")                    # se carga una vez
    resultado = clf.clasificar("foto.tif")           # especie y top-3 de cada diatomea
    clf.procesar_carpeta("fotos", "resultados")      # Excel + fotos anotadas
--------------------------------------
"""
from .clasificador import VERSION as __version__
from .clasificador import Clasificador, Recorte, Resultado
from .imagenes import ErrorEntrada
from .modelos import ErrorModelos

__all__ = ["Clasificador", "ErrorEntrada", "ErrorModelos", "Recorte", "Resultado", "__version__"]
