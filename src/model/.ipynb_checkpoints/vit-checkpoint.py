import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor, einsum, nn
from torchsummary import summary


class ResidualAdd(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x: Tensor, **kwargs):
        res = x
        x = self.layer(x, **kwargs)
        x += res
        return x


class FeedForward(nn.Sequential):
    def __init__(self, embed_size: int, expansion: int = 4, dropout: float = 0.3) -> None:
        super().__init__(
            nn.Linear(embed_size, expansion * embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * embed_size, embed_size),
        )


class PatchEmbedding(nn.Module):
    """
    Patch Embedding block for ViT encoder.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        img_H: int,
        img_W: int,
    ) -> None:
        super().__init__()

        embed_dim = in_channels * (patch_size**2)
        self.patch_size = patch_size

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b c (h) (w) -> b (h w) c"),
        )
        self.position = nn.Parameter(torch.randn(img_H * img_W // (patch_size**2), embed_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)

        x += self.position

        return x


class TimeEncoding(nn.Module):
    encoding: Tensor

    def __init__(self, time: int, img_H: int, img_W: int) -> None:
        super().__init__()

        self.time = time

        pos = torch.arange(start=0, end=self.time)
        inv_freq = 1.0 / (10000.0 ** torch.arange(start=0, end=img_H * img_W // 2) / img_H * img_W)

        pe = torch.zeros(self.time, img_H * img_W)
        pe[:, 0::2] = torch.sin(pos[:, None] * inv_freq[None, :])
        pe[:, 1::2] = torch.cos(pos[:, None] * inv_freq[None, :])
        pe = rearrange(pe, "t (h w) -> t h w", h=img_H, w=img_W)
        self.register_buffer("encoding", pe)

    def forward(self, x: Tensor) -> Tensor:
        B, C, _, _, _ = x.shape
        x = x + repeat(self.encoding, "t h w -> b c t h w", b=B, c=C)
        return rearrange(x, "b c t h w -> b (c t) h w")


class SelfAttention(nn.Module):
    """
    Simple self attention module.
    """

    def __init__(
        self,
        embed_dim: int,
        heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.heads = heads
        self.inv_sqrt_dim: float = (embed_dim) ** (-0.5)

        self.weight_k = nn.Linear(embed_dim, embed_dim)
        self.weight_q = nn.Linear(embed_dim, embed_dim)
        self.weight_v = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)

        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        q: Tensor = rearrange(self.weight_q(x), "b n (h d) -> b h n d", h=self.heads)
        k: Tensor = rearrange(self.weight_q(x), "b n (h d) -> b h n d", h=self.heads)
        v: Tensor = rearrange(self.weight_v(x), "b n (h d) -> b h n d", h=self.heads)

        energy = einsum("bhqd, bhkd -> bhqk", q, k)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.masked_fill(mask, fill_value)

        attention = F.softmax(energy * self.inv_sqrt_dim, dim=-1)
        attention = self.attn_dropout(attention)
        out = einsum("bhqk, bhkv -> bhqv", attention, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.projection(out)


class TransformerBlock(nn.Sequential):
    def __init__(
        self, embed_size: int, dropout: float = 0.3, expansion: int = 4, forward_dropout: float = 0.3, **kwargs
    ) -> None:
        super().__init__(
            ResidualAdd(
                nn.Sequential(nn.LayerNorm(embed_size), SelfAttention(embed_size, **kwargs), nn.Dropout(dropout))
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(embed_size), FeedForward(embed_size, expansion, forward_dropout), nn.Dropout(dropout)
                )
            ),
        )


class Encoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerBlock(**kwargs) for _ in range(depth)])


class DWSeparable(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 1),
        )
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(x + self.block(x))

class Decoder(nn.Sequential):
    def __init__(self, in_channels: int, patch_size: int, img_H: int, img_W: int, time: int) -> None:
        super().__init__(
            Rearrange("b (h w) c -> b c h w", h=img_H // patch_size, w=img_W // patch_size),
            nn.ConvTranspose2d(
                in_channels=in_channels * (patch_size**2),
                out_channels=in_channels,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            DWSeparable(in_channels),
            DWSeparable(in_channels),
            Rearrange("b (c t) h w -> b c t h w", t=time),
        )


class Model(nn.Module):
    """
    Basic model using ViT as a base model
    It enables the full network to catch details,
    find out relationships between variables in ENSO images.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        time: int,
        img_H: int,
        img_W: int,
        depth: int,
        **kwargs,
    ) -> None:
        super().__init__()

        self.layer = nn.Sequential(
            TimeEncoding(time=time, img_H=img_H, img_W=img_W),
            PatchEmbedding(in_channels=in_channels * time, patch_size=patch_size, img_H=img_H, img_W=img_W),
            Encoder(depth, embed_size=in_channels * (patch_size) ** 2 * time, **kwargs),
            Decoder(in_channels=in_channels * time, patch_size=patch_size, img_H=img_H, img_W=img_W, time=time),
        )

    def forward(self, x: Tensor):
        return x + self.layer(x)

if __name__ == "__main__":
    model: nn.Module = Model(in_channels=9, patch_size=4, time=8, img_H=40, img_W=200, depth=6)

    summary(model, (9, 8, 40, 200), device="cpu")
