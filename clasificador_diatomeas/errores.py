class ErrorModelos(Exception):
    """Falta un fichero en la carpeta de modelos o no es válido."""


class ErrorEntrada(Exception):
    """Una foto no se puede leer (fichero dañado o formato no soportado)."""
