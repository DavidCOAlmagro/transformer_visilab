"""Entrena el MLP de 75_objetivo (sin augmentation) con embeddings center-crop vs pad-square y compara en test."""
import sys, json, torch
from pathlib import Path
from PIL import Image
import numpy as np, pandas as pd
from torch import nn
from sklearn.metrics import f1_score
RAIZ = Path(r"C:\VISILAB\transformer\proyecto_transformer_v2"); sys.path.insert(0, str(RAIZ / "src"))
from constantes import VARIABLES_GLOBALES as V
from preparar_datos import get_datos, codificacion, fijar_semilla, construir_numero_genero, etiquetas_a_generos, obtener_especies_activas
from clasificador import ClasificadorDiatomeas
from dataloader import crear_dataloaders, calcular_pesos_clases
from entrenamiento import entrenar_modelo
from CenterLoss import center_loss
from generar_leer_splits import leer_split
import main as M
V["NUM_WORKERS"] = 0; V["PERSISTENT_WORKERS"] = False
BASE = Path(sys.argv[1])
items_test = leer_split(RAIZ / "data/splits/75_objetivo/test.txt")
ar = []
for r, _ in items_test:
    with Image.open(r) as im: w, h = im.size
    ar.append(max(w, h) / max(1, min(w, h)))
ar = np.array(ar)
res = {}
for variante in ("centercrop", "pad"):
    V["RUTA_EMBEDDINGS"] = BASE / variante
    fijar_semilla(42)
    etr, ytr, ne = codificacion(get_datos("train")); eva, yva, _ = codificacion(get_datos("val")); ete, yte, _ = codificacion(get_datos("test"))
    ng = construir_numero_genero(obtener_especies_activas()); nc = len(ne)
    dl_tr, dl_va = crear_dataloaders(etr, ytr, etiquetas_a_generos(ytr, ne, ng), eva, yva, etiquetas_a_generos(yva, ne, ng), nc)
    modelo = ClasificadorDiatomeas(nc, len(ng)).to(V["DEVICE"])
    loss = nn.CrossEntropyLoss(label_smoothing=V["LABEL_SMOOTHING"], weight=calcular_pesos_clases(ytr, nc))
    cl = center_loss(nc, V["DIM_CAPA_2"], V["DEVICE"])
    opt = torch.optim.AdamW(list(modelo.parameters()) + list(cl.parameters()), lr=V["LEARNING_RATE"], weight_decay=V["WEIGHT_DECAY"])
    sch = torch.optim.lr_scheduler.LambdaLR(opt, M.lr_lambda)
    ruta = BASE / f"modelo_{variante}.pth"
    entrenar_modelo(modelo, dl_tr, dl_va, loss, nn.CrossEntropyLoss(), cl, opt, sch, ruta, V["num_epocas"], V["PACIENCIA"])
    modelo.load_state_dict(torch.load(ruta, weights_only=True)); modelo.eval()
    with torch.no_grad(): pred = modelo(ete.to(V["DEVICE"]))[0].argmax(1).cpu().numpy()
    y = yte.numpy(); ok = pred == y
    res[variante] = {"acc": float(ok.mean()), "macro_f1": float(f1_score(y, pred, average="macro"))}
    for lo, hi in [(1, 1.5), (1.5, 2), (2, 3), (3, 5), (5, 1e9)]:
        m = (ar >= lo) & (ar < hi); res[variante][f"acc_ar_{lo}-{hi:g}"] = (float(ok[m].mean()), int(m.sum()))
    f1c = f1_score(y, pred, average=None, labels=range(nc)); inv = {v: k for k, v in ne.items()}
    res[variante]["f1_especie"] = {inv[i]: float(f) for i, f in enumerate(f1c)}
    print(variante, {k: v for k, v in res[variante].items() if k != "f1_especie"}, flush=True)
json.dump(res, open(BASE / "resultados.json", "w"), indent=1)
