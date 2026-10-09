import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "Inferir"))
from dividir_datos import dividir_campo  # noqa: E402
from seleccionar_recortes_campo import grupo_de_recorte, nombre_recorte, seleccionar  # noqa: E402
import infer_and_split_resnet_single_folder as inferir  # noqa: E402


def recorte(item, etiqueta, d, cd, r, cr, d3=("", ""), r3=("", "")):
    return {"image": f"Imagenes Aqualitas/{etiqueta}/{etiqueta}_{item}.tif", "item": item,
            "x1": 0, "y1": 0, "x2": 10, "y2": 10, "etiqueta": etiqueta,
            "top1_d": d, "conf1_d": cd, "top2_d": d3[0], "top3_d": d3[1],
            "top1_r": r, "conf1_r": cr, "top2_r": r3[0], "top3_r": r3[1]}


class SeleccionarTests(unittest.TestCase):
    clases = {"Gom_a", "Gom_b", "Gom_fp", "Debris"}

    def test_criterio(self):
        filas = pd.DataFrame([
            recorte(1, "Gom_a", "Gom_a", 90, "Gom_b", 50, r3=("Gom_a", "x")),  # DINO seguro + ResNet en top-3: sí
            recorte(2, "Gom_a", "Gom_a", 90, "Gom_b", 50),                      # el otro no la tiene en top-3: no
            recorte(3, "Gom_a", "Gom_b", 40, "Gom_a", 85, d3=("Gom_a", "x")),  # ResNet seguro, DINO falla: sí (difícil)
            recorte(4, "Gom_a", "Gom_fp", 70, "Gom_fp", 70),                    # pleural del mismo género: Gom_fp
            recorte(5, "Gom_a", "Debris", 99, "Debris", 99),                    # resto: no
            recorte(6, "Otra_x", "Otra_x", 99, "Otra_x", 99),                   # especie fuera de las clases: no
        ])
        r = seleccionar(filas, self.clases)
        self.assertEqual(sorted(zip(r.item, r.especie)), [(1, "Gom_a"), (3, "Gom_a"), (4, "Gom_fp")])
        self.assertEqual(r.set_index("item").dificil.to_dict(), {1: False, 3: True, 4: False})

    def test_tope_prioriza_dificiles(self):
        filas = pd.DataFrame([recorte(i, "Gom_a", "Gom_a", 95, "Gom_a", 95) for i in range(10)]
                             + [recorte(100, "Gom_a", "Gom_b", 30, "Gom_a", 95, d3=("Gom_a", "x"))])
        r = seleccionar(filas, self.clases, tope=3)
        self.assertEqual(len(r), 3)
        self.assertIn(100, r.item.tolist())


class RepartoCampoTests(unittest.TestCase):
    def test_una_foto_nunca_se_parte(self):
        recortes = []
        for foto in range(40):
            grupo = f"Gom_a:{foto}"
            for item in range(3):
                nombre = nombre_recorte(f"Imagenes Aqualitas/Gom a/Gom_a_{foto}.tif", item, grupo)
                recortes.append((Path("campo_pool") / "Gom_a" / nombre, "Gom_a"))
        partes = dividir_campo(recortes)
        grupos = {s: {grupo_de_recorte(p.name) for p, _ in items} for s, items in partes.items()}
        self.assertFalse(grupos["train"] & grupos["val"])
        self.assertEqual(len(partes["train"]) + len(partes["val"]), len(recortes))
        self.assertTrue(partes["val"])


class ListaImagenesTests(unittest.TestCase):
    def test_solo_las_de_la_lista(self):
        with tempfile.TemporaryDirectory() as carpeta:
            raiz = Path(carpeta) / "entrada"
            for r in ("A/a_1.tif", "A/a_2.tif", "B/b 1.tif"):
                (raiz / r).parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (4, 4)).save(raiz / r)
            lista = Path(carpeta) / "lista.txt"
            lista.write_text("A/a_2.tif\nB" + "\\" + "b 1.tif\n", encoding="utf-8")
            encontradas = inferir.discover_images(raiz, (), inferir.read_image_list(lista))
            self.assertEqual([p.relative_to(raiz).as_posix() for p in encontradas], ["A/a_2.tif", "B/b 1.tif"])


if __name__ == "__main__":
    unittest.main()
