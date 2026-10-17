"""
--------------------------------------
El modelo que dice la especie: DINOv2 + una red pequeña (MLP).

DINOv2 convierte cada recorte en un vector de 768 números que describe
cómo es la diatomea, y la MLP pasa ese vector a una probabilidad para
cada especie.

Está escrito solo con torch para no depender de la librería transformers,
que tarda mucho en cargar. Hace las mismas operaciones, así que da los
mismos resultados que el modelo con el que se entrenó.
--------------------------------------
"""
from pathlib import Path

import torch
from PIL import Image, ImageStat
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.v2 import functional as TF

# Tamaños de dinov2-base
DIMENSION = 768
CAPAS = 12
CABEZAS = 12
PARCHE = 14              # DINOv2 trocea la imagen en parches de 14x14 píxeles
LADO = 224               # tamaño de la imagen que entra al modelo
LADO_PREENTRENADO = 518  # tamaño con el que se preentrenó DINOv2

# Normalización de colores de ImageNet. Está multiplicada por 255 porque se aplica
# directamente sobre píxeles de 0 a 255
MEDIA = torch.tensor([0.485, 0.456, 0.406]) * (1.0 / 0.00392156862745098)
DESVIACION = torch.tensor([0.229, 0.224, 0.225]) * (1.0 / 0.00392156862745098)


def color_borde(imagen: Image.Image) -> tuple[int, int, int]:
    """
    Color medio de una franja alrededor del borde de la imagen.
    """
    ancho, alto = imagen.size
    grosor = max(1, min(ancho, alto) // 20)
    pixeles = []
    pixeles.extend(imagen.crop((0, 0, ancho, min(grosor, alto))).getdata())            # arriba
    pixeles.extend(imagen.crop((0, max(0, alto - grosor), ancho, alto)).getdata())     # abajo
    pixeles.extend(imagen.crop((0, 0, min(grosor, ancho), alto)).getdata())            # izquierda
    pixeles.extend(imagen.crop((max(0, ancho - grosor), 0, ancho, alto)).getdata())    # derecha
    muestra = Image.new("RGB", (len(pixeles), 1))
    muestra.putdata(pixeles)
    media = ImageStat.Stat(muestra).mean
    return tuple(max(0, min(255, round(valor))) for valor in media)


def rellenar_cuadrado(imagen: Image.Image) -> Image.Image:
    """
    Pone el recorte en el centro de un cuadrado del color de su borde. Así la
    diatomea no se deforma ni se corta al redimensionarla (igual que al entrenar).
    """
    imagen = imagen.convert("RGB")
    ancho, alto = imagen.size
    lado = max(ancho, alto)
    cuadrado = Image.new("RGB", (lado, lado), color=color_borde(imagen))
    cuadrado.paste(imagen, ((lado - ancho) // 2, (lado - alto) // 2))
    return cuadrado


def preparar(recortes: list[Image.Image]) -> torch.Tensor:
    """
    Convierte los recortes en el tensor que entra a DINOv2:
    1. Se rellena cada recorte hasta hacerlo cuadrado
    2. Se pasa a 256x256 y se coge el centro de 224x224
    3. Se normalizan los colores
    """
    tensores = []
    for recorte in recortes:
        imagen = TF.pil_to_tensor(rellenar_cuadrado(recorte))
        imagen = TF.resize(imagen, [256, 256], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        imagen = TF.crop(imagen, 16, 16, LADO, LADO)
        tensores.append(imagen)
    lote = torch.stack(tensores).to(torch.float32)
    return TF.normalize(lote, MEDIA, DESVIACION)


class Bloque(nn.Module):
    """
    Un bloque del transformer: primero atención (cada parche mira a todos los
    demás) y después una MLP. Cada parte suma su resultado a la entrada.
    """

    def __init__(self):
        super().__init__()
        self.norma1 = nn.LayerNorm(DIMENSION, eps=1e-6)
        self.q = nn.Linear(DIMENSION, DIMENSION)
        self.k = nn.Linear(DIMENSION, DIMENSION)
        self.v = nn.Linear(DIMENSION, DIMENSION)
        self.proyeccion = nn.Linear(DIMENSION, DIMENSION)
        self.escala1 = nn.Parameter(torch.ones(DIMENSION))
        self.norma2 = nn.LayerNorm(DIMENSION, eps=1e-6)
        self.fc1 = nn.Linear(DIMENSION, 4 * DIMENSION)
        self.fc2 = nn.Linear(4 * DIMENSION, DIMENSION)
        self.escala2 = nn.Parameter(torch.ones(DIMENSION))

    def atencion(self, x: torch.Tensor) -> torch.Tensor:
        n, tokens, _ = x.shape
        # Se separa en 12 cabezas de 64 números cada una
        k = self.k(x).view(n, -1, CABEZAS, DIMENSION // CABEZAS).transpose(1, 2)
        v = self.v(x).view(n, -1, CABEZAS, DIMENSION // CABEZAS).transpose(1, 2)
        q = self.q(x).view(n, -1, CABEZAS, DIMENSION // CABEZAS).transpose(1, 2)
        salida = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
                                                scale=(DIMENSION // CABEZAS) ** -0.5, is_causal=False)
        # Se vuelven a juntar las 12 cabezas
        salida = salida.transpose(1, 2).contiguous().reshape(n, tokens, DIMENSION)
        return self.proyeccion(salida)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.atencion(self.norma1(x)) * self.escala1 + x
        x = self.fc2(F.gelu(self.fc1(self.norma2(x)))) * self.escala2 + x
        return x


class Dinov2(nn.Module):
    """
    DINOv2-base. De cada imagen devuelve el vector CLS (768 números), que
    resume la imagen entera.
    """

    def __init__(self):
        super().__init__()
        self.parches = nn.Conv2d(3, DIMENSION, kernel_size=PARCHE, stride=PARCHE)
        self.cls = nn.Parameter(torch.zeros(1, 1, DIMENSION))
        n_parches = (LADO_PREENTRENADO // PARCHE) ** 2
        self.posiciones = nn.Parameter(torch.zeros(1, n_parches + 1, DIMENSION))
        self.bloques = nn.ModuleList([Bloque() for _ in range(CAPAS)])
        self.norma_final = nn.LayerNorm(DIMENSION, eps=1e-6)

    def posiciones_224(self) -> torch.Tensor:
        """
        Las posiciones se aprendieron para una rejilla de 37x37 parches (518 píxeles).
        Aquí la imagen es de 224 (16x16 parches), así que se interpolan.
        """
        cls = self.posiciones[:, :1]
        lado_original = LADO_PREENTRENADO // PARCHE
        lado_nuevo = LADO // PARCHE
        parches = self.posiciones[:, 1:].reshape(1, lado_original, lado_original, DIMENSION).permute(0, 3, 1, 2)
        parches = F.interpolate(parches, size=(lado_nuevo, lado_nuevo), mode="bicubic", align_corners=False)
        parches = parches.permute(0, 2, 3, 1).view(1, -1, DIMENSION)
        return torch.cat((cls, parches), dim=1)

    def forward(self, pixeles: torch.Tensor) -> torch.Tensor:
        # Un vector por parche, más el CLS delante, más la posición de cada uno
        x = self.parches(pixeles).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.posiciones_224()
        for bloque in self.bloques:
            x = bloque(x)
        x = self.norma_final(x)
        return x[:, 0, :]


class ModeloEspecies(nn.Module):
    """
    DINOv2 + MLP: de un lote de imágenes ya preparadas a la probabilidad de cada especie.
    """

    def __init__(self, n_especies: int):
        super().__init__()
        self.dinov2 = Dinov2()
        # Misma MLP que en el entrenamiento: 768 -> 512 -> 256 -> especies
        self.mlp = nn.Sequential(
            nn.Linear(DIMENSION, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_especies),
        )

    def forward(self, pixeles: torch.Tensor) -> torch.Tensor:
        cls = self.dinov2(pixeles)
        # El vector se normaliza (longitud 1) antes de la MLP, igual que al entrenar
        cls = cls / cls.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.softmax(self.mlp(cls).float(), dim=1)


# Los pesos del .pth tienen los nombres que les dio el entrenamiento (los de transformers).
# Esta tabla los traduce a los nombres de aquí.
CAMBIOS_DE_NOMBRE = [
    ("backbone.embeddings.patch_embeddings.projection", "dinov2.parches"),
    ("backbone.embeddings.position_embeddings", "dinov2.posiciones"),
    ("backbone.embeddings.cls_token", "dinov2.cls"),
    ("backbone.encoder.layer", "dinov2.bloques"),
    ("backbone.layernorm", "dinov2.norma_final"),
    ("attention.attention.query", "q"),
    ("attention.attention.key", "k"),
    ("attention.attention.value", "v"),
    ("attention.output.dense", "proyeccion"),
    ("layer_scale1.lambda1", "escala1"),
    ("layer_scale2.lambda1", "escala2"),
    ("norm1", "norma1"),
    ("norm2", "norma2"),
    ("mlp.fc1", "fc1"),
    ("mlp.fc2", "fc2"),
    ("tronco.", "mlp."),
    ("cabeza_especie.", "mlp.6."),
]
# Pesos del .pth que no hacen falta para clasificar
NO_SE_USAN = ("backbone.embeddings.mask_token", "cabeza_genero", "centros", "umbral_distancia")


def traducir_nombre(nombre: str) -> str:
    for viejo, nuevo in CAMBIOS_DE_NOMBRE:
        nombre = nombre.replace(viejo, nuevo)
    return nombre


def cargar_modelo(ruta: Path, n_especies: int) -> ModeloEspecies:
    """
    Crea el modelo y le carga los pesos del entrenamiento.
    """
    pesos_pth = torch.load(ruta, map_location="cpu", weights_only=True)
    pesos = {}
    for nombre, valor in pesos_pth.items():
        if not nombre.startswith(NO_SE_USAN):
            pesos[traducir_nombre(nombre)] = valor

    modelo = ModeloEspecies(n_especies)
    modelo.load_state_dict(pesos, strict=True)  # strict: falla si sobra o falta algún peso
    modelo.eval()
    return modelo
