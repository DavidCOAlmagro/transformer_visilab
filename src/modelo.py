"""Carga segura del clasificador DINOv2 entrenado."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from clasificador import ClasificadorDiatomeas
from constantes import VARIABLES_GLOBALES
from preparar_datos import construir_numero_genero, obtener_especies_activas


def _leer_clases_metadatos(ruta: Path) -> list[str]:
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de metadatos: {ruta}")
    with ruta.open(encoding="utf-8") as archivo:
        datos = json.load(archivo)
    clases = datos.get("especies_filtradas") or datos.get("especies")
    if not isinstance(clases, list) or not clases or not all(
        isinstance(clase, str) and clase.strip() for clase in clases
    ):
        raise ValueError(f"Metadatos sin lista de clases válida: {ruta}")
    clases = [clase.strip() for clase in clases]
    if len(set(clases)) != len(clases):
        raise ValueError(f"Hay clases duplicadas en los metadatos: {ruta}")
    return sorted(clases)


def cargar_modelo_entrenado(
    ruta_pesos: Path | None = None,
    ruta_metadatos: Path | None = None,
    clases: list[str] | None = None,
) -> tuple[ClasificadorDiatomeas, list[str]]:
    """Carga pesos DINOv2 y devuelve las clases en el orden de sus logits."""
    carpeta = VARIABLES_GLOBALES["RUTA_MODELOS"] / VARIABLES_GLOBALES["PRUEBA"]
    ruta_pesos = ruta_pesos or carpeta / f"modelo_{VARIABLES_GLOBALES['PRUEBA']}.pth"
    ruta_metadatos = ruta_metadatos or carpeta / "metadatos_modelo.json"
    especies_metadatos = _leer_clases_metadatos(ruta_metadatos)
    if clases is not None and list(clases) != especies_metadatos:
        raise ValueError(
            "El archivo de clases no coincide exactamente con el orden de "
            "metadatos_modelo.json."
        )
    especies = especies_metadatos
    if len(especies) != 77:
        raise ValueError(f"DINOv2 requiere exactamente 77 clases; se recibieron {len(especies)}.")
    if len(set(especies)) != len(especies):
        raise ValueError("La lista de clases DINOv2 contiene duplicados.")
    especies = sorted(especies)

    if not ruta_pesos.is_file():
        raise FileNotFoundError(f"No se encontró el modelo entrenado: {ruta_pesos}")
    numero_genero = construir_numero_genero(set(especies))
    modelo = ClasificadorDiatomeas(len(especies), len(numero_genero)).to(
        VARIABLES_GLOBALES["DEVICE"]
    )
    estado = torch.load(ruta_pesos, map_location=VARIABLES_GLOBALES["DEVICE"], weights_only=True)
    if not isinstance(estado, dict):
        raise ValueError(f"Los pesos no contienen un state_dict: {ruta_pesos}")
    try:
        resultado = modelo.load_state_dict(estado, strict=False)
    except RuntimeError as error:
        raise ValueError(f"Pesos incompatibles con 77 clases: {ruta_pesos}") from error
    opcionales = {"centros", "umbral_distancia"}
    faltantes = [clave for clave in resultado.missing_keys if clave not in opcionales]
    if faltantes or resultado.unexpected_keys:
        raise ValueError(
            f"State_dict incompatible. Faltan: {faltantes}; sobrantes: {resultado.unexpected_keys}"
        )
    modelo.eval()
    return modelo, especies
