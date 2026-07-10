from pathlib import Path
import runpy
import sys

script_path = Path(__file__).resolve().parent / \
    "diatomeas_DINOv2_ViT" / "auxiliar" / "nombrar_especies.py"

if not script_path.exists():
    raise FileNotFoundError(f"No se encontró el script: {script_path}")

sys.argv = [str(script_path)] + sys.argv[1:]
runpy.run_path(str(script_path), run_name="__main__")
