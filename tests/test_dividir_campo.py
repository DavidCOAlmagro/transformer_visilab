import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dividir_campo import clave_grupo, dividir  # noqa: E402
from evaluar_campo import filtrar_subconjunto  # noqa: E402


class DividirCampoTests(unittest.TestCase):
    def test_grupos(self):
        self.assertEqual(clave_grupo("Diatomeas DBO5 GT/x/Achnanthidium minutissimum_98865_DC_07.tif"), "dbo5:98865")
        self.assertEqual(clave_grupo("Imagenes Aqualitas/Melosira varians/Melosira_varians_89.2.tif"),
                         clave_grupo("Imagenes Aqualitas/Melosira varians/Melosira_varians_89.tif"))
        self.assertNotEqual(clave_grupo("Imagenes Aqualitas/A b/A_b_89.tif"),
                            clave_grupo("Imagenes Aqualitas/C d/C_d_89.tif"))

    def test_sin_fuga_entre_test_y_pool(self):
        filas = [(f"Imagenes Aqualitas/{e}/{e}_{n}{s}.tif", e)
                 for e in ("A_a", "B_b") for n in range(30) for s in ("", ".2")]
        etiquetas = pd.DataFrame(filas, columns=["imagen", "etiqueta"])
        test, pool = dividir(etiquetas)
        self.assertFalse({clave_grupo(i) for i in test} & {clave_grupo(i) for i in pool})
        self.assertEqual(len(test) + len(pool), len(etiquetas))
        self.assertTrue(0.25 < len(test) / len(etiquetas) < 0.42)

    def test_filtrar_subconjunto(self):
        etiquetas = pd.DataFrame({"imagen": ["a/1.tif", "a/2.tif"], "etiqueta": ["X", "Y"]})
        with tempfile.TemporaryDirectory() as carpeta:
            lista = Path(carpeta) / "test.txt"
            lista.write_text("a/2.tif\n", encoding="utf-8")
            self.assertEqual(filtrar_subconjunto(etiquetas, lista).imagen.tolist(), ["a/2.tif"])


if __name__ == "__main__":
    unittest.main()
