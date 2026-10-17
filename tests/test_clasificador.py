"""
--------------------------------------
Tests de clasificador_diatomeas que no necesitan los pesos.
Los que comparan con transformers se saltan si no está instalado.
--------------------------------------
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "src"))
import preprocesado as preprocesado_entrenamiento  # noqa: E402
from clasificador_diatomeas import excel, fotos  # noqa: E402
from clasificador_diatomeas.clasificador import Recorte, Resultado  # noqa: E402
from clasificador_diatomeas.dinov2 import Dinov2, preparar, rellenar_cuadrado, traducir_nombre  # noqa: E402

CLASES = (RAIZ / "Inferir" / "txt_classes" / "classes_77(dino).txt").read_text(encoding="utf-8").split()


def foto_ruido(ancho, alto, semilla=0):
    return Image.fromarray(np.random.default_rng(semilla).integers(0, 256, (alto, ancho, 3), dtype=np.uint8))


class NombresTests(unittest.TestCase):
    def test_ejemplos(self):
        self.assertEqual(excel.nombre_legible("Cocconeis_placentula_var_euglypta"), "Cocconeis placentula var. euglypta")
        self.assertEqual(excel.nombre_legible("Gomphonema_parvulum_f_saprophilum"), "Gomphonema parvulum f. saprophilum")
        self.assertEqual(excel.nombre_legible("Reimeria_sp"), "Reimeria sp.")
        self.assertEqual(excel.nombre_legible("Gomphonema_fp"), "Gomphonema (vista pleural)")
        self.assertEqual(excel.nombre_legible("Debris"), "Restos")
        self.assertEqual(excel.nombre_legible("Fragments"), "Fragmento")

    def test_todas_las_clases(self):
        tipos = [excel.tipo(clase) for clase in CLASES]
        self.assertEqual(tipos.count("resto"), 2)
        self.assertEqual(tipos.count("vista pleural"), 9)
        for clase in CLASES:
            self.assertNotIn("_", excel.nombre_legible(clase))


class RellenoTests(unittest.TestCase):
    def test_igual_que_en_el_entrenamiento(self):
        for semilla, (ancho, alto) in enumerate([(50, 200), (300, 40), (128, 128), (1, 7)]):
            recorte = foto_ruido(ancho, alto, semilla)
            referencia = preprocesado_entrenamiento.preparar_para_dinov2(recorte)
            self.assertEqual(rellenar_cuadrado(recorte).tobytes(), referencia.tobytes())


def carpeta_dinov2():
    """Carpeta de dinov2-base en la caché de Hugging Face, o None."""
    try:
        from embeddings import resolver_modelo_dinov2
        ruta, local = resolver_modelo_dinov2()
        return Path(ruta) if local else None
    except Exception:
        return None


@unittest.skipUnless(importlib.util.find_spec("transformers") and carpeta_dinov2(), "hace falta transformers")
class IgualQueTransformersTests(unittest.TestCase):
    """Nuestro DINOv2 da exactamente lo mismo que el de transformers con los mismos pesos."""

    def test_preparar(self):
        from transformers import AutoImageProcessor
        recortes = [foto_ruido(60, 180, 0), foto_ruido(300, 90, 1), foto_ruido(224, 224, 2)]
        procesador = AutoImageProcessor.from_pretrained(carpeta_dinov2(), local_files_only=True)
        esperado = procesador(images=[rellenar_cuadrado(r) for r in recortes], return_tensors="pt")["pixel_values"]
        self.assertTrue(torch.equal(preparar(recortes), esperado))

    def test_dinov2(self):
        from transformers import Dinov2Config, Dinov2Model
        torch.manual_seed(0)
        referencia = Dinov2Model(Dinov2Config.from_json_file(carpeta_dinov2() / "config.json")).eval()  # pesos al azar
        pesos = {}
        for nombre, valor in referencia.state_dict().items():
            if nombre != "embeddings.mask_token":
                pesos[traducir_nombre("backbone." + nombre).removeprefix("dinov2.")] = valor
        nuestro = Dinov2().eval()
        nuestro.load_state_dict(pesos, strict=True)
        pixeles = torch.randn(2, 3, 224, 224)
        with torch.inference_mode():
            self.assertTrue(torch.equal(nuestro(pixeles), referencia(pixel_values=pixeles).pooler_output))


class FotosTests(unittest.TestCase):
    def test_16_bits_no_se_satura(self):
        valores = np.linspace(0, 4000, 64 * 64, dtype=np.uint16).reshape(64, 64)
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "foto16.tif"
            Image.fromarray(valores).save(ruta)
            rgb = np.asarray(fotos.abrir(ruta))
        self.assertEqual(rgb.max(), 255)
        self.assertLess(np.mean(rgb), 200)  # convertida sin más saldría casi entera a 255

    def test_orientacion_exif(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "girada.jpg"
            exif = Image.Exif()
            exif[0x0112] = 6  # girada 90 grados
            foto_ruido(40, 20).save(ruta, exif=exif)
            self.assertEqual(fotos.abrir(ruta).size, (20, 40))

    def test_fichero_danado(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "mala.tif"
            ruta.write_bytes(b"esto no es una imagen")
            with self.assertRaises(OSError):
                fotos.abrir(ruta)

    def test_etiquetas_no_se_pisan(self):
        ocupados = []
        for caja in [(700, 120, 1100, 260), (720, 100, 900, 250), (710, 110, 1000, 240)]:  # tres cajas juntas
            ocupados.append(fotos.sitio_etiqueta(caja, 800, 60, 2048, 1536, ocupados))
        for i, a in enumerate(ocupados):
            for b in ocupados[i + 1:]:
                self.assertFalse(fotos.se_pisan(a, b))

    def test_anotar(self):
        foto = foto_ruido(200, 120)
        recortes = resultados_ejemplo()[0].recortes
        self.assertNotEqual(fotos.anotar(foto, recortes).tobytes(), foto.tobytes())
        self.assertEqual(fotos.anotar(foto, []).tobytes(), foto.tobytes())


def resultados_ejemplo():
    recorte = Recorte(1, (10, 10, 60, 40), "Gomphonema (vista pleural)", "vista pleural",
                      [("Gomphonema (vista pleural)", 70.0), ("Restos", 20.0), ("Navicula gregaria", 5.0)])
    return [Resultado("a.tif", "ok", [recorte]), Resultado("b.tif", "sin detecciones"),
            Resultado("c.tif", "error", mensaje="No se puede abrir c.tif")]


class ExcelTests(unittest.TestCase):
    def test_hojas_y_columnas(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "resultados.xlsx"
            excel.guardar(resultados_ejemplo(), ruta, "prueba")
            hojas = pd.read_excel(ruta, sheet_name=None)
        self.assertEqual(list(hojas), ["Recortes", "Imagenes", "Info"])
        self.assertEqual(list(hojas["Recortes"].columns), ["imagen", "recorte", "x1", "y1", "x2", "y2", "especie", "tipo",
                                                            "top1", "% top1", "top2", "% top2", "top3", "% top3"])
        fila = hojas["Recortes"].iloc[0]
        self.assertEqual((fila["especie"], fila["tipo"], fila["top2"], fila["% top1"]),
                         ("Gomphonema (vista pleural)", "vista pleural", "Restos", 70.0))
        self.assertEqual(hojas["Imagenes"].estado.tolist(), ["ok", "sin detecciones", "error"])


if __name__ == "__main__":
    unittest.main()
