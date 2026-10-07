import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Inferir"))
from infer_and_split_resnet_single_folder import backbone_state  # noqa: E402


class BackboneStateTests(unittest.TestCase):
    def test_checkpoint_antiguo_sin_backbone(self):
        estado = {"tronco.0.weight": torch.zeros(1), "cabeza_especie.bias": torch.zeros(1)}
        self.assertEqual(backbone_state(estado), {})

    def test_checkpoint_ajustado_quita_prefijo(self):
        estado = {"backbone.encoder.layer.0.norm1.weight": torch.ones(2), "tronco.0.weight": torch.zeros(1)}
        self.assertEqual(list(backbone_state(estado)), ["encoder.layer.0.norm1.weight"])


if __name__ == "__main__":
    unittest.main()
