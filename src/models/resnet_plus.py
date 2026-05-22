import einops
from timm.models import efficientnet, resnet, resnetv2
from torch import nn

from .positional_encodings import PositionalEncodingPermute2D
from .transformer import MultiHeadAttention


class SimpleSigmoidAttention2d(nn.Module):
    def __init__(
        self,
        in_channels,
    ):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.proj1 = nn.Conv2d(in_channels, in_channels // 2, 1, 1, 0)
        self.proj2 = nn.Conv2d(in_channels // 2, 1, 1, 1, 0)

    def forward(self, x):
        inp = x
        x = self.norm(x)
        x = self.proj1(x)
        x = nn.ReLU()(x)
        x = self.proj2(x).sigmoid()

        return inp * x


class FeedForward2d(nn.Module):
    def __init__(self, in_channels, hidden_channels, norm=nn.BatchNorm2d):
        super().__init__()
        self.norm = norm(in_channels)
        self.proj1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.proj2 = nn.Conv2d(hidden_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        x = self.norm(x)
        x = self.proj1(x)
        x = nn.ReLU()(x)
        x = self.proj2(x)

        return x


class BasicBlockWithAttentionAndEmbedding(resnet.BasicBlock):
    def __init__(self, inplanes, planes, *args, attn_heads=8, **kwargs):
        super().__init__(inplanes, planes, *args, **kwargs)
        self.pos_emb = PositionalEncodingPermute2D(planes)

        self.ff = FeedForward2d(planes, planes // 4)
        self.attn = SimpleSigmoidAttention2d(planes)

    def forward(self, x):
        x = super().forward(x)
        skip = x

        x = x + self.pos_emb(x)
        x = self.ff(x)
        x = self.attn(x)

        return x + skip


def resnet10t_attn(pretrained=False, **kwargs):
    model_args = dict(
        block=BasicBlockWithAttentionAndEmbedding,
        layers=(1, 1, 1, 1),
        stem_width=32,
        stem_type="deep_tiered",
        avg_down=True,
    )
    return resnet._create_resnet("resnet10t", pretrained, **dict(model_args, **kwargs))
