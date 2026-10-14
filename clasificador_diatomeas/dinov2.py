"""DINOv2-base y su preprocesado con solo torch y torchvision.

Reproduce ``Dinov2Model`` y ``BitImageProcessor`` de ``transformers`` 5.12 operación a operación (los nombres
de los parámetros son los mismos, así que el ``.pth`` del entrenamiento se carga tal cual). Así el paquete no
necesita ``transformers``, cuya importación es lo que más tarda al arrancar. Que da lo mismo lo comprueba
``scratchpad/paridad_clasificadores.py`` frente a ``transformers``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.v2 import functional as tvF


class Preprocesado:
    """Lado corto a 256 (bicúbico con antialias), recorte central de 224 y normalización ImageNet."""

    def __init__(self, ruta_config: Path) -> None:
        config = json.loads(ruta_config.read_text(encoding="utf-8"))
        self.lado_corto = config["size"]["shortest_edge"]
        self.recorte = (config["crop_size"]["height"], config["crop_size"]["width"])
        # Igual que transformers: el reescalado a [0, 1] va dentro de la media y la desviación
        self.media = torch.tensor(config["image_mean"]) * (1.0 / config["rescale_factor"])
        self.desviacion = torch.tensor(config["image_std"]) * (1.0 / config["rescale_factor"])

    def _uno(self, imagen: Image.Image) -> torch.Tensor:
        tensor = tvF.pil_to_tensor(imagen.convert("RGB"))
        alto, ancho = tensor.shape[-2:]
        corto, largo = (ancho, alto) if ancho <= alto else (alto, ancho)
        nuevo_largo = int(self.lado_corto * largo / corto)
        tamano = [nuevo_largo, self.lado_corto] if ancho <= alto else [self.lado_corto, nuevo_largo]
        tensor = tvF.resize(tensor, tamano, interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
        alto, ancho = tensor.shape[-2:]
        alto_r, ancho_r = self.recorte
        return tvF.crop(tensor, int((alto - alto_r) / 2.0), int((ancho - ancho_r) / 2.0), alto_r, ancho_r)

    def __call__(self, imagenes: list[Image.Image]) -> torch.Tensor:
        lote = torch.stack([self._uno(imagen) for imagen in imagenes]).to(dtype=torch.float32)
        return tvF.normalize(lote, self.media, self.desviacion)


class _Proyeccion(nn.Module):
    def __init__(self, canales: int, dim: int, parche: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(canales, dim, kernel_size=parche, stride=parche)


class _Embeddings(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        dim, parche = config["hidden_size"], config["patch_size"]
        n_parches = (config["image_size"] // parche) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))  # solo para preentrenar; está en el .pth
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_parches + 1, dim))
        self.patch_embeddings = _Proyeccion(config["num_channels"], dim, parche)
        self.parche = parche

    def posiciones(self, alto: int, ancho: int) -> torch.Tensor:
        """Embeddings de posición interpolados (bicúbico) a la rejilla de la imagen, como ``interpolate_pos_encoding``."""
        n_posiciones = self.position_embeddings.shape[1] - 1
        nuevo_alto, nuevo_ancho = alto // self.parche, ancho // self.parche
        if nuevo_alto * nuevo_ancho == n_posiciones and alto == ancho:
            return self.position_embeddings
        clase, parches = self.position_embeddings[:, :1], self.position_embeddings[:, 1:]
        lado, dim = int(n_posiciones ** 0.5), parches.shape[-1]
        parches = parches.reshape(1, lado, lado, dim).permute(0, 3, 1, 2)
        parches = F.interpolate(parches.to(torch.float32), size=(nuevo_alto, nuevo_ancho), mode="bicubic",
                                align_corners=False).to(dtype=self.position_embeddings.dtype)
        return torch.cat((clase, parches.permute(0, 2, 3, 1).view(1, -1, dim)), dim=1)

    def forward(self, pixeles: torch.Tensor) -> torch.Tensor:
        _, _, alto, ancho = pixeles.shape
        x = self.patch_embeddings.projection(pixeles).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        return x + self.posiciones(alto, ancho)


class _QKV(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query, self.key, self.value = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)


class _Salida(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(dim, dim)


class _Atencion(nn.Module):
    def __init__(self, dim: int, cabezas: int) -> None:
        super().__init__()
        self.attention, self.output = _QKV(dim), _Salida(dim)
        self.cabezas, self.dim_cabeza = cabezas, dim // cabezas

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forma = (x.shape[0], -1, self.cabezas, self.dim_cabeza)
        k = self.attention.key(x).view(*forma).transpose(1, 2)
        v = self.attention.value(x).view(*forma).transpose(1, 2)
        q = self.attention.query(x).view(*forma).transpose(1, 2)
        salida = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
                                                scale=self.dim_cabeza ** -0.5, is_causal=False)
        salida = salida.transpose(1, 2).contiguous()
        return self.output.dense(salida.reshape(*salida.shape[:-2], -1))


class _Escala(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(dim))


class _MLP(nn.Module):
    def __init__(self, dim: int, ratio: int) -> None:
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, dim * ratio), nn.Linear(dim * ratio, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _Bloque(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        dim, eps = config["hidden_size"], config["layer_norm_eps"]
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=eps), nn.LayerNorm(dim, eps=eps)
        self.attention = _Atencion(dim, config["num_attention_heads"])
        self.mlp = _MLP(dim, config["mlp_ratio"])
        self.layer_scale1, self.layer_scale2 = _Escala(dim), _Escala(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(self.norm1(x)) * self.layer_scale1.lambda1 + x
        return self.mlp(self.norm2(x)) * self.layer_scale2.lambda1 + x


class _Encoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.layer = nn.ModuleList(_Bloque(config) for _ in range(config["num_hidden_layers"]))


class Dinov2(nn.Module):
    """DINOv2 (sin SwiGLU ni drop path, como dinov2-base). ``forward`` devuelve el token CLS tras la última normalización."""

    def __init__(self, ruta_config: Path) -> None:
        super().__init__()
        config = json.loads(ruta_config.read_text(encoding="utf-8"))
        if config.get("use_swiglu_ffn"):
            raise ValueError("Esta implementación es para DINOv2 sin SwiGLU (dinov2-small/base/large).")
        self.embeddings = _Embeddings(config)
        self.encoder = _Encoder(config)
        self.layernorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.hidden_size = config["hidden_size"]

    def forward(self, pixeles: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(pixeles)
        for bloque in self.encoder.layer:
            x = bloque(x)
        return self.layernorm(x)[:, 0, :]
