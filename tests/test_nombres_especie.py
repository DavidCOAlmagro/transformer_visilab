import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from preparar_datos import normalizar_nombre_especie  # noqa: E402


class NombresEspecieTests(unittest.TestCase):
    def test_espacios_y_puntos(self):
        self.assertEqual(normalizar_nombre_especie("Fistulifera saprophila"), "Fistulifera_saprophila")
        self.assertEqual(normalizar_nombre_especie("Cymbella excisa var. excisa"), "Cymbella_excisa_var_excisa")

    def test_nombre_correcto_no_cambia(self):
        self.assertEqual(normalizar_nombre_especie("Mayamaea_permitis"), "Mayamaea_permitis")


if __name__ == "__main__":
    unittest.main()
