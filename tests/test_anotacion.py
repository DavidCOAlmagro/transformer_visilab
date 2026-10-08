import sys
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Inferir"))
from infer_and_split_resnet_single_folder import (  # noqa: E402
    ANNOTATION_COLOR, _annotate_image, _annotation_label, _font,
)


class AnotacionTests(unittest.TestCase):
    def test_etiqueta_solo_especie(self):
        prediccion = {"especie_mas_parecida": "Gomphonema_rhombicum", "confianza": 42.0, "revisar": True}
        etiqueta = _annotation_label(prediccion)
        self.assertEqual(etiqueta, "Gomphonema rhombicum")
        self.assertNotIn("%", etiqueta)
        self.assertNotIn("REVISION", etiqueta)

    def test_recuadro_verde_y_fuente_escalable(self):
        imagen = Image.new("RGB", (2000, 1500), (128, 128, 128))
        prediccion = {"especie_mas_parecida": "Nitzschia_palea", "confianza": 90.0, "revisar": False}
        anotada = _annotate_image(imagen, [(500, 600, 900, 1000)], [prediccion], "dinov2")
        self.assertEqual(anotada.getpixel((500, 800)), ANNOTATION_COLOR)  # borde izquierdo del recuadro
        self.assertGreaterEqual(_font(50).size, 50)

    def test_etiquetas_cercanas_no_se_pisan(self):
        imagen = Image.new("RGB", (2000, 1500), (128, 128, 128))
        cajas = [(500, 600, 900, 1000), (700, 620, 1100, 1020)]
        predicciones = [{"especie_mas_parecida": "A_b", "confianza": 1.0, "revisar": False}] * 2
        # No debe fallar y ambas cajas deben quedar dibujadas
        anotada = _annotate_image(imagen, cajas, predicciones, "resnet")
        self.assertEqual(anotada.getpixel((1100, 900)), ANNOTATION_COLOR)


if __name__ == "__main__":
    unittest.main()
