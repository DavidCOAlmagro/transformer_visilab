import unittest

from PIL import Image

from src.auditoria_dataset import auditar_dataset
from src.preprocesado import pad_to_square


class PreprocesadoTests(unittest.TestCase):
    def test_padding_conserva_roi_y_no_deforma(self):
        imagen = Image.new("RGB", (100, 40), (10, 20, 30))
        salida = pad_to_square(imagen, color=(1, 2, 3))
        self.assertEqual(salida.size, (100, 100))
        self.assertEqual(salida.getpixel((0, 0)), (1, 2, 3))
        self.assertEqual(salida.getpixel((0, 30)), (10, 20, 30))

    def test_color_borde_es_determinista(self):
        imagen = Image.new("RGB", (10, 20), (40, 50, 60))
        self.assertEqual(pad_to_square(imagen).getpixel((0, 0)), (40, 50, 60))

    def test_auditoria_detecta_duplicados_y_candidata(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            for grupo, clase in (("g1", "Activa"), ("g2", "Nueva")):
                carpeta = raiz / grupo / clase
                carpeta.mkdir(parents=True)
                Image.new("RGB", (2, 2), (1, 2, 3)).save(carpeta / f"{grupo}.png")
            informe = auditar_dataset(raiz, {"Activa"}, minimo_recomendado=1)
            self.assertEqual(informe["total_imagenes"], 2)
            self.assertEqual(informe["clases_adicionales_recomendadas"], ["Nueva"])
            self.assertEqual(len(informe["duplicados_exactos"]), 1)


if __name__ == "__main__":
    unittest.main()
