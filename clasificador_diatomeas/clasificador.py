"""
--------------------------------------
Clasificador de diatomeas para la app.

Se crea una vez (carga los modelos) y después se le pasan las fotos:
YOLO encuentra las diatomeas y, para cada una, el modelo da la especie
más probable y las tres más probables con su probabilidad.
--------------------------------------
"""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .excel import guardar_excel, nombre_legible, tipo_clase
from .imagenes import ErrorEntrada, abrir_imagen, anotar, buscar_imagenes, nombre_fichero
from .modelos import Detector, ErrorModelos, ModeloDino, ModeloResNet, combinar, indices_resnet, top3

VERSION = "1.0.0"


@dataclass
class Recorte:
    """Una diatomea encontrada en la foto."""
    numero: int                        # orden en que la encontró YOLO (1, 2, ...)
    caja: tuple[int, int, int, int]    # x1, y1, x2, y2 en píxeles de la foto
    top3: list[tuple[str, float]]      # (clase, probabilidad de 0 a 1), de más a menos probable
    clase_detector: str = ""           # lo que dijo YOLO: Diatom o Not_diatom

    @property
    def codigo(self) -> str:
        return self.top3[0][0]

    @property
    def especie(self) -> str:
        return nombre_legible(self.codigo)

    @property
    def tipo(self) -> str:
        return tipo_clase(self.codigo)


@dataclass
class Resultado:
    """Resultado de una foto. El estado es "ok", "sin detecciones" o "error"."""
    imagen: str
    estado: str
    recortes: list[Recorte] = field(default_factory=list)
    mensaje: str = ""
    probabilidades: dict | None = field(default=None, repr=False)  # solo si se piden (uso interno)


def leer_json(ruta: Path) -> dict:
    if not ruta.is_file():
        raise ErrorModelos(f"No existe {ruta}")
    return json.loads(ruta.read_text(encoding="utf-8"))


def comprobar_modelos(carpeta: str | Path) -> list[str]:
    """
    Comprueba con el sha256 del manifiesto que los ficheros de la carpeta de
    modelos están completos. Devuelve los que faltan o están dañados.
    """
    carpeta = Path(carpeta)
    malos = []
    for nombre, sha256 in leer_json(carpeta / "manifest.json").get("ficheros", {}).items():
        ruta = carpeta / nombre
        if not ruta.is_file() or hashlib.sha256(ruta.read_bytes()).hexdigest() != sha256:
            malos.append(nombre)
    return malos


class Clasificador:
    """
    Carga los modelos de la carpeta indicada una sola vez. Después se puede
    usar con todas las fotos que hagan falta.
    """

    def __init__(self, carpeta_modelos: str | Path, cargar_resnet: bool | None = None):
        self.carpeta = Path(carpeta_modelos)
        manifiesto = leer_json(self.carpeta / "manifest.json")
        clases = leer_json(self.carpeta / "clases.json")
        self.clases: list[str] = clases["dino"]
        self.version_modelos: str = manifiesto.get("version_modelos", "")
        # Peso de DINOv2 al mezclarlo con ResNet (1 = solo DINOv2)
        self.peso_dino = float(manifiesto.get("peso_dino", 1.0))
        # Si es True se descartan las cajas que YOLO marca como Not_diatom
        self.solo_diatomeas = bool(manifiesto.get("solo_cajas_diatom", False))

        yolo = manifiesto.get("yolo", {})
        self.detector = Detector(self.carpeta / "yolo.pt", yolo.get("confianza", 0.30), yolo.get("tamano", 1024))
        self.dino = ModeloDino(self.carpeta / "dinov2.pth", self.carpeta / "dinov2_config", len(self.clases))

        # ResNet solo se carga si entra en la mezcla (o si se piden sus probabilidades)
        if cargar_resnet is None:
            cargar_resnet = self.peso_dino < 1
        self.resnet = None
        self.indices = []
        if cargar_resnet:
            self.resnet = ModeloResNet(self.carpeta / "resnet50.pth", len(clases["resnet"]))
            self.indices = indices_resnet(clases["resnet"], self.clases)

    def clasificar(self, foto: str | Path | Image.Image, guardar_probabilidades: bool = False) -> Resultado:
        """
        Clasifica una foto (ruta o imagen de PIL). Si no se puede abrir, el
        resultado sale con estado "error" y el motivo en "mensaje".
        """
        if isinstance(foto, Image.Image):
            nombre = Path(getattr(foto, "filename", "") or "imagen").name
        else:
            nombre = Path(foto).name
        try:
            imagen = abrir_imagen(foto)
        except ErrorEntrada as error:
            return Resultado(nombre, "error", mensaje=str(error))
        return self._clasificar(imagen, nombre, guardar_probabilidades)

    def anotar(self, foto: str | Path | Image.Image, resultado: Resultado) -> Image.Image:
        """La foto con un recuadro en cada diatomea y el nombre de su especie."""
        cajas = [recorte.caja for recorte in resultado.recortes]
        nombres = [recorte.especie for recorte in resultado.recortes]
        return anotar(abrir_imagen(foto), cajas, nombres)

    def guardar_excel(self, resultados: list[Resultado], ruta: str | Path) -> Path:
        """Escribe el Excel (hojas Recortes, Imagenes e Info) con los resultados de clasificar."""
        return guardar_excel(resultados, Path(ruta), VERSION, self.version_modelos)

    def procesar_carpeta(self, entrada: str | Path, salida: str | Path, anotar_fotos: bool = True,
                         al_avanzar: Callable[[int, int, str], None] | None = None,
                         guardar_probabilidades: bool = False) -> Path:
        """
        Clasifica una foto o todas las de una carpeta (con subcarpetas) y escribe en "salida":
            resultados.xlsx            una fila por diatomea y otra por foto
            anotadas/<foto>.jpg        si anotar_fotos es True
        Si una foto falla sale con estado "error" y se sigue con las demás.
        al_avanzar(hechas, total, foto) se llama después de cada foto (para una barra de progreso).
        Devuelve la ruta del Excel.
        """
        entrada = Path(entrada)
        salida = Path(salida)
        if entrada.is_file():
            fotos = [entrada]
            base = entrada.parent
        elif entrada.is_dir():
            fotos = buscar_imagenes(entrada, excluir=salida)
            base = entrada
        else:
            raise ErrorEntrada(f"No existe {entrada}")
        if not fotos:
            raise ErrorEntrada(f"No hay fotos en {entrada}")

        resultados = []
        for hechas, foto in enumerate(fotos, start=1):
            relativa = foto.relative_to(base)
            try:
                imagen = abrir_imagen(foto)
                resultado = self._clasificar(imagen, relativa.as_posix(), guardar_probabilidades)
                if anotar_fotos:
                    carpeta_anotadas = salida / "anotadas"
                    carpeta_anotadas.mkdir(parents=True, exist_ok=True)
                    cajas = [recorte.caja for recorte in resultado.recortes]
                    nombres = [recorte.especie for recorte in resultado.recortes]
                    anotar(imagen, cajas, nombres).save(carpeta_anotadas / f"{nombre_fichero(relativa)}.jpg", quality=90)
            except Exception as error:  # una foto que falla no para todo el lote
                resultado = Resultado(relativa.as_posix(), "error", mensaje=str(error))
            resultados.append(resultado)
            if al_avanzar is not None:
                al_avanzar(hechas, len(fotos), relativa.as_posix())

        if guardar_probabilidades:
            guardar_npz(resultados, salida / "probabilidades.npz")
        return self.guardar_excel(resultados, salida / "resultados.xlsx")

    def _clasificar(self, imagen: Image.Image, nombre: str, guardar_probabilidades: bool) -> Resultado:
        # 1. YOLO busca las diatomeas
        detecciones = self.detector.detectar(imagen)
        if self.solo_diatomeas:
            detecciones = [deteccion for deteccion in detecciones if deteccion[2] == "Diatom"]
        if not detecciones:
            return Resultado(nombre, "sin detecciones")

        # 2. Cada diatomea se recorta y se clasifica
        recortes = [imagen.crop(caja) for _, caja, _ in detecciones]
        p_dino = self.dino.probabilidades(recortes)
        p_resnet = self.resnet.probabilidades(recortes) if self.resnet is not None else None
        probabilidades = combinar(p_dino, p_resnet, self.indices, self.peso_dino)

        # 3. Top-3 de cada recorte
        resultado = Resultado(nombre, "ok")
        for (numero, caja, clase_detector), mejores in zip(detecciones, top3(probabilidades, self.clases)):
            resultado.recortes.append(Recorte(numero, caja, mejores, clase_detector))
        if guardar_probabilidades:
            resultado.probabilidades = {"dino": p_dino, "resnet": p_resnet}
        return resultado


def guardar_npz(resultados: list[Resultado], ruta: Path) -> None:
    """
    Guarda las probabilidades completas de cada recorte. Es de uso interno:
    sirve para decidir cómo mezclar DINOv2 y ResNet (spec 013).
    """
    con_datos = [resultado for resultado in resultados if resultado.probabilidades is not None]
    imagenes, numeros, cajas, clases_detector = [], [], [], []
    for resultado in con_datos:
        for recorte in resultado.recortes:
            imagenes.append(resultado.imagen)
            numeros.append(recorte.numero)
            cajas.append(recorte.caja)
            clases_detector.append(recorte.clase_detector)

    datos = {
        "imagen": np.array(imagenes),
        "recorte": np.array(numeros),
        "caja": np.array(cajas).reshape(-1, 4),
        "clase_detector": np.array(clases_detector),
        "p_dino": np.concatenate([r.probabilidades["dino"] for r in con_datos]) if con_datos else np.zeros((0, 0)),
    }
    if con_datos and all(r.probabilidades["resnet"] is not None for r in con_datos):
        datos["p_resnet"] = np.concatenate([r.probabilidades["resnet"] for r in con_datos])
    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ruta, **datos)
