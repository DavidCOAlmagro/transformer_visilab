"""Clasificador: detecta las diatomeas de una foto y da la especie más probable de cada una, con su top-3."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .anotacion import anotar
from .combinacion import combinar, indices_resnet, top3
from .deteccion import Detector
from .errores import ErrorEntrada, ErrorModelos
from .imagenes import abrir_imagen, buscar_imagenes
from .modelos import ModeloDino, ModeloResNet
from .resultados import Recorte, Resultado
from .salida import guardar_excel


def leer_json(ruta: Path) -> dict:
    if not ruta.is_file():
        raise ErrorModelos(f"No existe {ruta}")
    return json.loads(ruta.read_text(encoding="utf-8"))


def comprobar_modelos(carpeta: str | Path) -> list[str]:
    """Ficheros de la carpeta de modelos que faltan o no coinciden con el manifiesto (vacía si todo está bien)."""
    carpeta = Path(carpeta)
    malos = []
    for nombre, sha256 in leer_json(carpeta / "manifest.json").get("ficheros", {}).items():
        ruta = carpeta / nombre
        if not ruta.is_file() or hashlib.sha256(ruta.read_bytes()).hexdigest() != sha256:
            malos.append(nombre)
    return malos


def nombre_fichero(relativa: Path) -> str:
    """``muestra 1/foto.tif`` -> ``muestra_1__foto``: único por foto y válido en cualquier sistema."""
    partes = [re.sub(r"[^0-9A-Za-zÀ-ÿ.-]+", "_", parte).strip(" ._") or "imagen" for parte in relativa.with_suffix("").parts]
    return "__".join(partes) or "imagen"


class Clasificador:
    """Carga los modelos de ``carpeta_modelos`` una sola vez; después se usa con tantas fotos como haga falta."""

    def __init__(self, carpeta_modelos: str | Path, cargar_resnet: bool | None = None) -> None:
        self.carpeta = Path(carpeta_modelos)
        manifiesto = leer_json(self.carpeta / "manifest.json")
        clases = leer_json(self.carpeta / "clases.json")
        self.clases: list[str] = clases["dino"]
        self.version_modelos = manifiesto.get("version_modelos", "")
        self.peso_dino = float(manifiesto.get("peso_dino", 1.0))
        self.solo_diatomeas = bool(manifiesto.get("solo_cajas_diatom", False))
        yolo = manifiesto.get("yolo", {})
        self.detector = Detector(self.carpeta / "yolo.pt", yolo.get("confianza", 0.30), yolo.get("tamano", 1024))
        self.dino = ModeloDino(self.carpeta / "dinov2.pth", self.carpeta / "dinov2_config", len(self.clases))
        # ResNet solo se carga si entra en la combinación (o si se piden sus probabilidades)
        usar_resnet = self.peso_dino < 1 if cargar_resnet is None else cargar_resnet
        self.resnet = ModeloResNet(self.carpeta / "resnet50.pth", len(clases["resnet"])) if usar_resnet else None
        self.indices = indices_resnet(clases["resnet"], self.clases) if usar_resnet else []

    def clasificar(self, foto: str | Path | Image.Image, guardar_probabilidades: bool = False) -> Resultado:
        """Clasifica una foto (ruta o ``PIL.Image``). Si no se puede abrir, el resultado sale con estado ``error``."""
        if isinstance(foto, Image.Image):
            nombre = Path(getattr(foto, "filename", "") or "imagen").name
        else:
            nombre = Path(foto).name
        try:
            imagen = abrir_imagen(foto)
        except ErrorEntrada as error:
            return Resultado(nombre, "error", mensaje=str(error))
        return self._clasificar(imagen, nombre, guardar_probabilidades)

    def guardar_excel(self, resultados: list[Resultado], ruta: str | Path) -> Path:
        """Escribe el Excel (hojas Recortes, Imagenes e Info) con los resultados de ``clasificar``."""
        return guardar_excel(resultados, Path(ruta), self.version_modelos)

    def anotar(self, foto: str | Path | Image.Image, resultado: Resultado) -> Image.Image:
        """La foto con un recuadro en cada diatomea y el nombre de su especie."""
        return anotar(abrir_imagen(foto), resultado.recortes)

    def procesar_carpeta(
        self,
        entrada: str | Path,
        salida: str | Path,
        anotar_fotos: bool = True,
        al_avanzar: Callable[[int, int, str], None] | None = None,
        guardar_probabilidades: bool = False,
    ) -> Path:
        """Clasifica una foto o todas las de una carpeta (con subcarpetas) y escribe en ``salida``:

        - ``resultados.xlsx``: una fila por diatomea (hoja Recortes) y una por foto (hoja Imagenes);
        - ``anotadas/<foto>.jpg`` si ``anotar_fotos``.

        Una foto que falla sale en la hoja Imagenes con estado ``error`` y no para el resto.
        ``al_avanzar(hechas, total, foto)`` se llama tras cada foto, por ejemplo para una barra de progreso.
        Devuelve la ruta del Excel.
        """
        entrada, salida = Path(entrada), Path(salida)
        if entrada.is_file():
            fotos, base = [entrada], entrada.parent
        elif entrada.is_dir():
            fotos, base = buscar_imagenes(entrada, excluir=salida), entrada
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
                    (salida / "anotadas").mkdir(parents=True, exist_ok=True)
                    anotar(imagen, resultado.recortes).save(salida / "anotadas" / f"{nombre_fichero(relativa)}.jpg", quality=90)
            except Exception as error:  # una foto que falla no para el lote
                resultado = Resultado(relativa.as_posix(), "error", mensaje=str(error))
            resultados.append(resultado)
            if al_avanzar is not None:
                al_avanzar(hechas, len(fotos), relativa.as_posix())
        if guardar_probabilidades:
            guardar_npz(resultados, salida / "probabilidades.npz")
        return self.guardar_excel(resultados, salida / "resultados.xlsx")

    def _clasificar(self, imagen: Image.Image, nombre: str, guardar_probabilidades: bool) -> Resultado:
        detecciones = self.detector.detectar(imagen)
        if self.solo_diatomeas:
            detecciones = [d for d in detecciones if d[2] == "Diatom"]
        if not detecciones:
            return Resultado(nombre, "sin detecciones")
        recortes = [imagen.crop(caja) for _, caja, _ in detecciones]
        p_dino = self.dino.probabilidades(recortes)
        p_resnet = self.resnet.probabilidades(recortes) if self.resnet is not None else None
        probabilidades = combinar(p_dino, p_resnet, self.indices, self.peso_dino)
        resultado = Resultado(nombre, "ok", [
            Recorte(numero, caja, mejores, clase)
            for (numero, caja, clase), mejores in zip(detecciones, top3(probabilidades, self.clases))
        ])
        if guardar_probabilidades:
            resultado.probabilidades = {"dino": p_dino, "resnet": p_resnet}
        return resultado


def guardar_npz(resultados: list[Resultado], ruta: Path) -> None:
    """Probabilidades completas de cada recorte (uso interno: elegir la combinación, spec 013)."""
    con_datos = [r for r in resultados if r.probabilidades is not None]
    recortes = [(r, c) for r in con_datos for c in r.recortes]
    datos = {
        "imagen": np.array([r.imagen for r, _ in recortes]),
        "recorte": np.array([c.numero for _, c in recortes]),
        "caja": np.array([c.caja for _, c in recortes]).reshape(-1, 4),
        "clase_detector": np.array([c.clase_detector for _, c in recortes]),
        "p_dino": np.concatenate([r.probabilidades["dino"] for r in con_datos]) if con_datos else np.zeros((0, 0)),
    }
    if con_datos and all(r.probabilidades["resnet"] is not None for r in con_datos):
        datos["p_resnet"] = np.concatenate([r.probabilidades["resnet"] for r in con_datos])
    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ruta, **datos)
