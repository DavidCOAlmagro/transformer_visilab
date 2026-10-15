"""
--------------------------------------
DINOv2-base y su preprocesado, escritos solo con torch y torchvision.

Es el mismo modelo que Dinov2Model de la librería transformers (mismas
operaciones y mismos nombres de los pesos), así que el .pth del
entrenamiento se carga tal cual y no hace falta instalar transformers,
que era lo que más tardaba al abrir la app.
--------------------------------------
"""
import json
from pathlib import Path

import torch
from PIL import Image, ImageStat
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.v2 import functional as tvF


def color_borde(imagen: Image.Image) -> tuple[int, int, int]:
    """
    Color medio de una franja del borde de la imagen. Se usa para rellenar
    el recorte hasta hacerlo cuadrado sin que se note el relleno.
    """
    rgb = imagen.convert("RGB")
    ancho, alto = rgb.size
    if ancho == 0 or alto == 0:
        return (128, 128, 128)
    grosor = max(1, min(ancho, alto) // 20)

    # Píxeles de las cuatro franjas: arriba, abajo, izquierda y derecha
    pixeles = []
    pixeles.extend(rgb.crop((0, 0, ancho, min(grosor, alto))).getdata())
    pixeles.extend(rgb.crop((0, max(0, alto - grosor), ancho, alto)).getdata())
    pixeles.extend(rgb.crop((0, 0, min(grosor, ancho), alto)).getdata())
    pixeles.extend(rgb.crop((max(0, ancho - grosor), 0, ancho, alto)).getdata())

    muestra = Image.new("RGB", (len(pixeles), 1))
    muestra.putdata(pixeles)
    media = ImageStat.Stat(muestra).mean
    if not media:
        return (128, 128, 128)
    return tuple(max(0, min(255, round(valor))) for valor in media)


def rellenar_cuadrado(imagen: Image.Image) -> Image.Image:
    """
    Centra el recorte en un lienzo cuadrado del color del borde. Así la
    diatomea no se deforma ni se corta al redimensionar. Es lo mismo que se
    hizo al entrenar.
    """
    rgb = imagen.convert("RGB")
    ancho, alto = rgb.size
    lado = max(ancho, alto)
    if lado <= 0:
        raise ValueError("La imagen debe tener dimensiones positivas.")
    lienzo = Image.new("RGB", (lado, lado), color=color_borde(rgb))
    lienzo.paste(rgb, ((lado - ancho) // 2, (lado - alto) // 2))
    return lienzo


class Preprocesado:
    """
    Convierte los recortes en el tensor que espera DINOv2: relleno a
    cuadrado, lado corto a 256 píxeles (bicúbico), recorte central de 224
    y normalización con la media y la desviación de ImageNet.
    Los valores salen de dinov2_config/preprocessor_config.json.
    """

    def __init__(self, ruta_config: Path):
        config = json.loads(ruta_config.read_text(encoding="utf-8"))
        self.lado_corto: int = config["size"]["shortest_edge"]
        self.alto_recorte: int = config["crop_size"]["height"]
        self.ancho_recorte: int = config["crop_size"]["width"]
        # Como en transformers, el paso de 0-255 a 0-1 va metido en la media y la desviación
        self.media = torch.tensor(config["image_mean"]) * (1.0 / config["rescale_factor"])
        self.desviacion = torch.tensor(config["image_std"]) * (1.0 / config["rescale_factor"])

    def __call__(self, recortes: list[Image.Image]) -> torch.Tensor:
        tensores = []
        for recorte in recortes:
            tensor = tvF.pil_to_tensor(rellenar_cuadrado(recorte))

            # Lado corto a 256 manteniendo la proporción (al ser cuadrado queda 256x256)
            alto, ancho = tensor.shape[-2:]
            corto, largo = (ancho, alto) if ancho <= alto else (alto, ancho)
            nuevo_largo = int(self.lado_corto * largo / corto)
            if ancho <= alto:
                tamano = [nuevo_largo, self.lado_corto]
            else:
                tamano = [self.lado_corto, nuevo_largo]
            tensor = tvF.resize(tensor, tamano, interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)

            # Recorte central de 224x224
            alto, ancho = tensor.shape[-2:]
            arriba = int((alto - self.alto_recorte) / 2.0)
            izquierda = int((ancho - self.ancho_recorte) / 2.0)
            tensores.append(tvF.crop(tensor, arriba, izquierda, self.alto_recorte, self.ancho_recorte))

        lote = torch.stack(tensores).to(dtype=torch.float32)
        return tvF.normalize(lote, self.media, self.desviacion)


# A partir de aquí, las piezas del modelo. Los nombres de los atributos (query, dense, lambda1...)
# son los de transformers para que las claves del .pth coincidan.

class _Proyeccion(nn.Module):
    """Parte la imagen en parches de 14x14 y convierte cada uno en un vector de 768."""

    def __init__(self, canales: int, dimension: int, parche: int):
        super().__init__()
        self.projection = nn.Conv2d(canales, dimension, kernel_size=parche, stride=parche)


class _Embeddings(nn.Module):
    """Token CLS + parches + posición de cada parche."""

    def __init__(self, config: dict):
        super().__init__()
        dimension = config["hidden_size"]
        self.parche: int = config["patch_size"]
        n_parches = (config["image_size"] // self.parche) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.mask_token = nn.Parameter(torch.zeros(1, dimension))  # solo se usa al preentrenar, pero está en el .pth
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_parches + 1, dimension))
        self.patch_embeddings = _Proyeccion(config["num_channels"], dimension, self.parche)

    def posiciones(self, alto: int, ancho: int) -> torch.Tensor:
        """
        DINOv2 se preentrenó con imágenes de 518 píxeles (37x37 parches) y aquí le
        pasamos 224 (16x16), así que las posiciones se interpolan a la rejilla nueva.
        """
        n_posiciones = self.position_embeddings.shape[1] - 1
        nuevo_alto = alto // self.parche
        nuevo_ancho = ancho // self.parche
        if nuevo_alto * nuevo_ancho == n_posiciones and alto == ancho:
            return self.position_embeddings

        posicion_cls = self.position_embeddings[:, :1]
        posiciones_parches = self.position_embeddings[:, 1:]
        lado = int(n_posiciones ** 0.5)
        dimension = posiciones_parches.shape[-1]
        posiciones_parches = posiciones_parches.reshape(1, lado, lado, dimension).permute(0, 3, 1, 2)
        posiciones_parches = F.interpolate(posiciones_parches.to(torch.float32), size=(nuevo_alto, nuevo_ancho),
                                           mode="bicubic", align_corners=False)
        posiciones_parches = posiciones_parches.to(dtype=self.position_embeddings.dtype)
        posiciones_parches = posiciones_parches.permute(0, 2, 3, 1).view(1, -1, dimension)
        return torch.cat((posicion_cls, posiciones_parches), dim=1)

    def forward(self, pixeles: torch.Tensor) -> torch.Tensor:
        _, _, alto, ancho = pixeles.shape
        parches = self.patch_embeddings.projection(pixeles).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(parches.shape[0], -1, -1)
        tokens = torch.cat((cls, parches), dim=1)
        return tokens + self.posiciones(alto, ancho)


class _QKV(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)


class _Salida(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dense = nn.Linear(dimension, dimension)


class _Atencion(nn.Module):
    """Atención de 12 cabezas: cada token mira a todos los demás."""

    def __init__(self, dimension: int, cabezas: int):
        super().__init__()
        self.attention = _QKV(dimension)
        self.output = _Salida(dimension)
        self.cabezas = cabezas
        self.dimension_cabeza = dimension // cabezas

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forma = (x.shape[0], -1, self.cabezas, self.dimension_cabeza)
        k = self.attention.key(x).view(*forma).transpose(1, 2)
        v = self.attention.value(x).view(*forma).transpose(1, 2)
        q = self.attention.query(x).view(*forma).transpose(1, 2)
        salida = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
                                                scale=self.dimension_cabeza ** -0.5, is_causal=False)
        # Se juntan otra vez las 12 cabezas en un solo vector por token
        salida = salida.transpose(1, 2).contiguous()
        salida = salida.reshape(*salida.shape[:-2], -1)
        return self.output.dense(salida)


class _Escala(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(dimension))


class _MLP(nn.Module):
    def __init__(self, dimension: int, ratio: int):
        super().__init__()
        self.fc1 = nn.Linear(dimension, dimension * ratio)
        self.fc2 = nn.Linear(dimension * ratio, dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _Bloque(nn.Module):
    """Un bloque del transformer: atención y MLP, cada uno con su normalización y su suma residual."""

    def __init__(self, config: dict):
        super().__init__()
        dimension = config["hidden_size"]
        self.norm1 = nn.LayerNorm(dimension, eps=config["layer_norm_eps"])
        self.attention = _Atencion(dimension, config["num_attention_heads"])
        self.layer_scale1 = _Escala(dimension)
        self.norm2 = nn.LayerNorm(dimension, eps=config["layer_norm_eps"])
        self.mlp = _MLP(dimension, config["mlp_ratio"])
        self.layer_scale2 = _Escala(dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(self.norm1(x)) * self.layer_scale1.lambda1 + x
        x = self.mlp(self.norm2(x)) * self.layer_scale2.lambda1 + x
        return x


class _Encoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.layer = nn.ModuleList([_Bloque(config) for _ in range(config["num_hidden_layers"])])


class Dinov2(nn.Module):
    """
    DINOv2 completo (como dinov2-base: sin SwiGLU). Devuelve el token CLS de
    cada imagen después de la última normalización: un vector de 768.
    """

    def __init__(self, ruta_config: Path):
        super().__init__()
        config = json.loads(ruta_config.read_text(encoding="utf-8"))
        if config.get("use_swiglu_ffn"):
            raise ValueError("Esta versión es para DINOv2 sin SwiGLU (dinov2-small, base o large).")
        self.hidden_size: int = config["hidden_size"]
        self.embeddings = _Embeddings(config)
        self.encoder = _Encoder(config)
        self.layernorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])

    def forward(self, pixeles: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(pixeles)
        for bloque in self.encoder.layer:
            x = bloque(x)
        x = self.layernorm(x)
        return x[:, 0, :]
