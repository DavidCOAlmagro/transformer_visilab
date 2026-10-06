import unittest

import pandas as pd

from src.evaluar_campo import agregar_imagen, clase_excluida, evaluar


class AgregacionTests(unittest.TestCase):
    def test_excluye_no_especies_y_posicion_pleural(self):
        self.assertTrue(clase_excluida("Debris", excluir_fp=False))
        self.assertFalse(clase_excluida("Gomphonema_fp", excluir_fp=False))
        self.assertTrue(clase_excluida("Gomphonema_fp", excluir_fp=True))
        self.assertTrue(clase_excluida(float("nan"), excluir_fp=False))

    def test_voto_desempata_por_suma_de_confianza(self):
        recortes = [("A", 40.0, 1), ("B", 90.0, 1), ("A", 10.0, 1), ("B", 5.0, 1)]
        self.assertEqual(agregar_imagen(recortes, "voto"), "B")

    def test_reglas(self):
        recortes = [("A", 30.0, 100), ("A", 30.0, 10), ("B", 80.0, 50), ("Gomphonema_fp", 99.0, 500)]
        self.assertEqual(agregar_imagen(recortes, "mayor"), "Gomphonema_fp")
        self.assertEqual(agregar_imagen(recortes, "mayor", excluir_fp=True), "A")
        self.assertEqual(agregar_imagen(recortes, "suma_conf", excluir_fp=True), "B")
        self.assertEqual(agregar_imagen([("Debris", 99.0, 1)], "voto"), "")

    def test_evaluar_cuenta_imagen_sin_recortes_como_fallo(self):
        recortes = pd.DataFrame({
            "image": ["i1", "i1", "i2"], "top1": ["A", "A_fp", "B"],
            "conf1": [60.0, 90.0, 70.0], "area": [1, 1, 1],
        })
        etiquetas = pd.DataFrame({"imagen": ["i1", "i2", "i3", "i4"],
                                  "etiqueta": ["A", "A", "A", "Z"]})
        resumen, _ = evaluar({"m": recortes}, etiquetas, {"A", "B"})
        fila = resumen[(resumen.regla == "suma_conf") & resumen.excluye_fp].iloc[0]
        self.assertEqual(fila.n_imagenes, 3)  # i4 no es evaluable
        self.assertAlmostEqual(fila.acc_m, round(1 / 3, 4))


if __name__ == "__main__":
    unittest.main()
