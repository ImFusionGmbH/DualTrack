from src.models.raft.extractor import BasicEncoder
from torch import nn
import torch
import einops


class TwoDimensionalFeatureMapEmbedding(nn.Module):
    def __init__(self, height, width, dim):
        super().__init__()
        self.pos_embedding = torch.nn.Parameter(
            torch.randn(1, height * width, dim) * 0.2
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        x = x + self.pos_embedding.repeat([B, 1, 1])

        return x


class ClassToken(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._token = torch.nn.Parameter(torch.randn(1, 1, dim))

    def add_token(self, x):
        B, L, D = x.shape
        cls_token = self._token.repeat([B, 1, 1])
        return torch.cat([cls_token, x], dim=1)

    @staticmethod
    def remove_token(x):
        return x[:, 1:, :]

    @staticmethod
    def get_token(x):
        return x[:, 0, :]


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim

        self.to_query = nn.Linear(dim, dim)
        self.to_key = nn.Linear(dim, dim)
        self.to_value = nn.Linear(dim, dim)

        self.to_out = nn.Linear(dim, dim)

    def forward(self, query_x, target_x):
        B, L, D = query_x.shape
        B, S, D = target_x.shape

        queries = self.to_query(query_x)
        queries = einops.rearrange(queries, "b l (h d) -> b h l d", h=self.n_heads)
        keys = self.to_key(target_x)
        keys = einops.rearrange(keys, "b s (h d) -> b h s d", h=self.n_heads)
        values = self.to_value(target_x)
        values = einops.rearrange(values, "b s (h d) -> b h s d", h=self.n_heads)

        outputs = torch.nn.functional.scaled_dot_product_attention(
            queries, keys, values
        )
        outputs = einops.rearrange(outputs, "b h l d -> b l (h d)")
        outputs = self.to_out(outputs)
        return outputs


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.attn = MultiHeadAttention(dim, n_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x))
        x = self.norm2(x + self.mlp(x))
        return x


class TransformerBlockWithCrossAttention(nn.Module):
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.attn = MultiHeadAttention(dim, n_heads)
        self.cross_attn = MultiHeadAttention(dim, n_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x, target_x):
        x = self.norm1(x + self.attn(x, x))
        x = self.norm2(x + self.cross_attn(x, target_x))
        x = self.norm3(x + self.mlp(x))
        return x


class VisionAttentionNetwork(nn.Module):
    def __init__(self, img_size=(256, 256), n_attn_blocks=4):
        super().__init__()

        self.encoder = BasicEncoder(norm_fn="instance", in_chans=1)
        self.features_downsample = torch.nn.MaxPool2d((2, 2), (2, 2))
        self.position_encoder = TwoDimensionalFeatureMapEmbedding(
            img_size[0] // 16, img_size[1] // 16, 128
        )
        self.class_token = ClassToken(128)

        self.attn = MultiHeadAttention(dim=128)

        self.attn_blocks = nn.ModuleList(
            [TransformerBlock(128) for _ in range(n_attn_blocks)]
        )

        self.cross_attn_blocks = nn.ModuleList(
            [TransformerBlockWithCrossAttention(128) for _ in range(n_attn_blocks)]
        )

        self.fc = nn.Linear(128, 6)

    def forward(self, image_sequence):
        b, n, c, h, w = image_sequence.shape
        image_sequence = einops.rearrange(image_sequence, "b n c h w -> (b n) c h w")
        feats = self.encoder(image_sequence)
        feats = self.features_downsample(feats)
        feats = einops.rearrange(feats, "(b n) c h w -> b n c h w", b=b, n=n)

        feats_now = feats[:, 1:, ...]
        feats_prev = feats[:, :-1, ...]

        feats_now = einops.rearrange(feats_now, "b n c h w -> (b n) c h w")
        feats_prev = einops.rearrange(feats_prev, "b n c h w -> (b n) c h w")

        feats_now_emb = self.position_encoder(feats_now)
        feats_now_emb = self.class_token.add_token(feats_now_emb)

        feats_prev_emb = self.position_encoder(feats_prev)

        feats_now_emb = self.attn(feats_now_emb, feats_now_emb)
        feats_now_emb = self.attn(feats_now_emb, feats_prev_emb)

        for attn_block, cross_attn_block in zip(
            self.attn_blocks, self.cross_attn_blocks
        ):
            feats_prev_emb = attn_block(feats_prev_emb)
            feats_now_emb = cross_attn_block(feats_now_emb, feats_prev_emb)

        feats_now_emb = self.class_token.get_token(feats_now_emb)
        feats_now_emb = einops.rearrange(
            feats_now_emb, "(b n) d -> b n d", b=b, n=n - 1
        )

        return self.fc(feats_now_emb)
