"""
--------------------------------------
Tests de clasificador_diatomeas (spec 014) que no necesitan los pesos.
Los de DINOv2 propio se saltan si no está transformers instalado.
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
from clasificador_diatomeas.clasificador import Recorte, Resultado  # noqa: E402
from clasificador_diatomeas.dinov2 import rellenar_cuadrado  # noqa: E402
from clasificador_diatomeas.excel import COLUMNAS_IMAGENES, COLUMNAS_RECORTES, guardar_excel, nombre_legible, tipo_clase  # noqa: E402
from clasificador_diatomeas.imagenes import (ErrorEntrada, abrir_imagen, anotar, buscar_imagenes,  # noqa: E402
                                             nombre_fichero, posicion_etiqueta)
from clasificador_diatomeas.modelos import ErrorModelos, combinar, indices_resnet, top3  # noqa: E402

CLASES_DINO = (RAIZ / "Inferir" / "txt_classes" / "classes_77(dino).txt").read_text(encoding="utf-8").split()
CLASES_RESNET = (RAIZ / "Inferir" / "txt_classes" / "classes_78(resnet).txt").read_text(encoding="utf-8").split()


def foto_ruido(ancho, alto, semilla=0):
    return Image.fromarray(np.random.default_rng(semilla).integers(0, 256, (alto, ancho, 3), dtype=np.uint8))


class NombresTests(unittest.TestCase):
    def test_ejemplos(self):
        self.assertEqual(nombre_legible("Cocconeis_placentula_var_euglypta"), "Cocconeis placentula var. euglypta")
        self.assertEqual(nombre_legible("Gomphonema_parvulum_f_saprophilum"), "Gomphonema parvulum f. saprophilum")
        self.assertEqual(nombre_legible("Reimeria_sp"), "Reimeria sp.")
        self.assertEqual(nombre_legible("Gomphonema_fp"), "Gomphonema (vista pleural)")
        self.assertEqual(nombre_legible("Debris"), "Restos")
        self.assertEqual(nombre_legible("Fragments"), "Fragmento")

    def test_todas_las_clases(self):
        tipos = [tipo_clase(clase) for clase in CLASES_DINO]
        self.assertEqual(tipos.count("resto"), 2)
        self.assertEqual(tipos.count("vista pleural"), 9)
        for clase in CLASES_DINO:
            self.assertTrue(nombre_legible(clase))
            self.assertNotIn("_", nombre_legible(clase))


class RellenoTests(unittest.TestCase):
    def test_igual_que_en_el_entrenamiento(self):
        for semilla, (ancho, alto) in enumerate([(50, 200), (300, 40), (128, 128), (1, 7)]):
            recorte = foto_ruido(ancho, alto, semilla)
            nuestro = rellenar_cuadrado(recorte)
            referencia = preprocesado_entrenamiento.preparar_para_dinov2(recorte)
            self.assertEqual(nuestro.size, referencia.size)
            self.assertEqual(nuestro.tobytes(), referencia.tobytes())


def _carpeta_config_dinov2():
    """Carpeta de dinov2-base en la caché de Hugging Face (con config.json y preprocessor_config.json), o None."""
    try:
        from embeddings import resolver_modelo_dinov2
        ruta, local = resolver_modelo_dinov2()
    except Exception:
        return None
    return Path(ruta) if local else None


@unittest.skipUnless(importlib.util.find_spec("transformers") and _carpeta_config_dinov2(),
                     "hace falta transformers y dinov2-base en la caché para comparar")
class Dinov2PropioTests(unittest.TestCase):
    """El DINOv2 del paquete da exactamente lo mismo que el de transformers con los mismos pesos."""

    def test_preprocesado_identico(self):
        from transformers import AutoImageProcessor
        from clasificador_diatomeas.dinov2 import Preprocesado
        carpeta = _carpeta_config_dinov2()
        recortes = [foto_ruido(ancho, alto, semilla) for semilla, (ancho, alto) in enumerate([(60, 180), (300, 90), (224, 224)])]
        procesador_hf = AutoImageProcessor.from_pretrained(carpeta, local_files_only=True)
        esperado = procesador_hf(images=[rellenar_cuadrado(r) for r in recortes], return_tensors="pt")["pixel_values"]
        obtenido = Preprocesado(carpeta / "preprocessor_config.json")(recortes)
        self.assertTrue(torch.equal(obtenido, esperado))

    def test_modelo_identico(self):
        from transformers import Dinov2Config, Dinov2Model
        from clasificador_diatomeas.dinov2 import Dinov2
        config = _carpeta_config_dinov2() / "config.json"
        torch.manual_seed(0)
        referencia = Dinov2Model(Dinov2Config.from_json_file(config)).eval()  # pesos aleatorios
        nuestro = Dinov2(config).eval()
        nuestro.load_state_dict(referencia.state_dict(), strict=True)
        pixeles = torch.randn(2, 3, 224, 224)
        with torch.inference_mode():
            self.assertTrue(torch.equal(nuestro(pixeles), referencia(pixel_values=pixeles).pooler_output))


class CombinacionTests(unittest.TestCase):
    def test_alineacion_con_resnet(self):
        indices = indices_resnet(CLASES_RESNET, CLASES_DINO)
        self.assertEqual(len(indices), 77)
        self.assertNotIn(CLASES_RESNET.index("Planothidium_fp"), indices)
        self.assertEqual(CLASES_RESNET[indices[CLASES_DINO.index("Denticula_tenuis")]], "Denticula_tenius")

    def test_falta_una_clase(self):
        with self.assertRaises(ErrorModelos):
            indices_resnet(CLASES_RESNET[:-5], CLASES_DINO)

    def test_combinar_y_top3(self):
        rng = np.random.default_rng(1)
        p_dino = rng.dirichlet(np.ones(77), 4)
        p_resnet = rng.dirichlet(np.ones(78), 4)
        indices = indices_resnet(CLASES_RESNET, CLASES_DINO)
        np.testing.assert_array_equal(combinar(p_dino, p_resnet, indices, 1.0), p_dino)
        mezcla = combinar(p_dino, p_resnet, indices, 0.6)
        np.testing.assert_allclose(mezcla, 0.6 * p_dino + 0.4 * p_resnet[:, indices])
        mejores = top3(mezcla, CLASES_DINO)
        self.assertEqual(len(mejores), 4)
        self.assertEqual(mejores[0][0][0], CLASES_DINO[int(np.argmax(mezcla[0]))])
        for fila in mejores:
            self.assertGreaterEqual(fila[0][1], fila[1][1])
            self.assertGreaterEqual(fila[1][1], fila[2][1])


class ImagenesTests(unittest.TestCase):
    def test_16_bits_no_se_satura(self):
        valores = np.linspace(0, 4000, 64 * 64, dtype=np.uint16).reshape(64, 64)
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "foto16.tif"
            Image.fromarray(valores).save(ruta)
            rgb = np.asarray(abrir_imagen(ruta))
        self.assertEqual(rgb.shape, (64, 64, 3))
        self.assertEqual(rgb.max(), 255)
        self.assertLess(np.mean(rgb), 200)  # convertida sin más saldría casi entera a 255

    def test_orientacion_exif(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "girada.jpg"
            exif = Image.Exif()
            exif[0x0112] = 6  # girada 90 grados
            foto_ruido(40, 20).save(ruta, exif=exif)
            self.assertEqual(abrir_imagen(ruta).size, (20, 40))

    def test_fichero_danado(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "mala.tif"
            ruta.write_bytes(b"esto no es una imagen")
            with self.assertRaises(ErrorEntrada):
                abrir_imagen(ruta)

    def test_buscar_imagenes_no_se_salta_carpetas(self):
        with tempfile.TemporaryDirectory() as carpeta:
            raiz = Path(carpeta)
            for relativa in ("a.tif", "results/b.png", "muestra 1/c.JPG", "salida/d.jpg", "notas.txt"):
                (raiz / relativa).parent.mkdir(parents=True, exist_ok=True)
                (raiz / relativa).write_bytes(b"x")
            encontradas = [p.relative_to(raiz).as_posix() for p in buscar_imagenes(raiz, excluir=raiz / "salida")]
        self.assertEqual(encontradas, ["a.tif", "muestra 1/c.JPG", "results/b.png"])

    def test_nombre_fichero(self):
        self.assertEqual(nombre_fichero(Path("muestra 1/foto 2.tif")), "muestra_1__foto_2")


def resultados_ejemplo():
    recorte = Recorte(1, (10, 10, 60, 40), [("Gomphonema_fp", 0.7), ("Debris", 0.2), ("Navicula_gregaria", 0.05)], "Diatom")
    return [Resultado("a.tif", "ok", [recorte]),
            Resultado("b.tif", "sin detecciones"),
            Resultado("c.tif", "error", mensaje="No se puede abrir c.tif")]


class SalidaTests(unittest.TestCase):
    def test_excel(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = guardar_excel(resultados_ejemplo(), Path(carpeta) / "resultados.xlsx", "1.0.0", "prueba")
            hojas = pd.read_excel(ruta, sheet_name=None)
        self.assertEqual(list(hojas), ["Recortes", "Imagenes", "Info"])
        recortes = hojas["Recortes"]
        imagenes = hojas["Imagenes"]
        self.assertEqual(list(recortes.columns), COLUMNAS_RECORTES)
        self.assertEqual(list(imagenes.columns), COLUMNAS_IMAGENES)
        fila = recortes.iloc[0]
        self.assertEqual((fila["especie"], fila["tipo"], fila["top2"], fila["% top1"]),
                         ("Gomphonema (vista pleural)", "vista pleural", "Restos", 70.0))
        self.assertEqual(imagenes.estado.tolist(), ["ok", "sin detecciones", "error"])

    def test_etiquetas_no_se_pisan(self):
        ocupadas = []
        for caja in [(700, 120, 1100, 260), (720, 100, 900, 250), (710, 110, 1000, 240)]:  # tres cajas juntas arriba
            ocupadas.append(posicion_etiqueta(caja, 800, 60, 2048, 1536, ocupadas))
        for i, a in enumerate(ocupadas):
            for b in ocupadas[i + 1:]:
                se_pisan = a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
                self.assertFalse(se_pisan, (a, b))

    def test_anotar(self):
        foto = foto_ruido(200, 120)
        recortes = resultados_ejemplo()[0].recortes
        anotada = anotar(foto, [r.caja for r in recortes], [r.especie for r in recortes])
        self.assertEqual(anotada.size, foto.size)
        self.assertNotEqual(anotada.tobytes(), foto.tobytes())
        self.assertEqual(anotar(foto, [], []).tobytes(), foto.tobytes())


if __name__ == "__main__":
    unittest.main()
