import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Inferir"))
import infer_and_split_resnet_single_folder as inferir  # noqa: E402


class ResultadosInferenciaTests(unittest.TestCase):
    def test_ruta_modelo_y_prueba_sin_sobrescribir(self):
        with tempfile.TemporaryDirectory() as carpeta:
            raiz = Path(carpeta)
            primera = inferir.results_dir("75_objetivo_ft", "prueba", root=raiz)
            self.assertEqual(primera, raiz / "75_objetivo_ft" / "prueba")
            primera.mkdir(parents=True)
            self.assertEqual(inferir.results_dir("75_objetivo_ft", "prueba", root=raiz).name, "prueba_2")

    def test_nombre_del_modelo(self):
        args = inferir.build_parser().parse_args(["--classifier", "both"])
        self.assertEqual(inferir.model_name(args), "75_objetivo_ft+resnet50")
        args = inferir.build_parser().parse_args(["--classifier", "resnet"])
        self.assertEqual(inferir.model_name(args), "resnet50")

    def test_por_defecto_dentro_de_resultados_inferencia(self):
        self.assertEqual(inferir.RESULTS_ROOT, inferir.ROOT / "Inferir" / "resultados_inferencia")


if __name__ == "__main__":
    unittest.main()
