"""
Implementation of transformers from the original paper:

@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}

Author: Paul Wilson
"""

from torch import nn
from .mlp import MLP
import torch
import einops
import typing as tp
import math


INF = torch.tensor(1e10)


def get_cos_sin_positonal_embedding(d_model, length):
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(
        (
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_k: int | None = None, d_v: int | None = None
    ):
        """
        Implements multi-head scaled dot product self attention.
        Args:
            d_model (int): the number of tokens for the input features.
            n_heads (int): the number of attention heads.
            d_k (optional, int): the dimension of key vectors for each head. Defaults
                to d_model divided by the number of heads.
            d_v (optional, int): the dimension of the value vectors for each head. Defaults
                to d_model divided by the number of heads.
        """
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_k = d_k = d_k or (d_model // n_heads)
        self.d_v = d_v = d_v or (d_model // n_heads)

        self.W_q = nn.Linear(d_model, d_k * n_heads)
        self.W_k = nn.Linear(d_model, d_k * n_heads)
        self.W_v = nn.Linear(d_model, d_v * n_heads)

        self.W_o = nn.Linear(d_v * n_heads, d_model)

        self.scale = 1 / torch.sqrt(torch.tensor(d_k).float())

    def forward(self, X, mask=None, return_attn=False):
        """
        X - tensor of shape batch_size, sequence_length, input_dim
        mask - tensor of shape batch_size, n_heads, sequence_length, sequence_length
        """

        B, n_tokens, d = X.shape

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        Q = einops.rearrange(Q, "b n (h d_k) -> b h n d_k", d_k=self.d_k)
        K_t = einops.rearrange(K, "b n (h d_k) -> b h d_k n", d_k=self.d_k)
        V = einops.rearrange(V, "b n (h d_v) -> b h n d_v", d_v=self.d_v)

        attn_scores = Q @ K_t * self.scale
        if mask is not None:
            attn_scores[mask == 0] = -INF.type(attn_scores.dtype)

        attn = attn_scores.softmax(-1)
        attn_weighted_values = attn @ V

        # concatenate across head dim
        attn_weighted_values = einops.rearrange(
            attn_weighted_values, "b h n d_v -> b n (h d_v)"
        )

        layer_output = self.W_o(attn_weighted_values)
        if return_attn:
            return layer_output, attn
        else:
            return layer_output


class MultiHeadAttentionWithRelativePositionalEmbeddings(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: int | None = None,
        d_v: int | None = None,
        max_distance=512,
        mode: tp.Literal["key", "value", "both"] = "both",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.mode = mode

        self.d_k = d_k = d_k or (d_model // n_heads)
        self.d_v = d_v = d_v or (d_model // n_heads)

        self.W_q = nn.Linear(d_model, d_k * n_heads)
        self.W_k = nn.Linear(d_model, d_k * n_heads)
        self.W_v = nn.Linear(d_model, d_v * n_heads)

        self.W_o = nn.Linear(d_v * n_heads, d_model)

        self.scale = 1 / torch.sqrt(torch.tensor(d_k).float())

        self.max_distance = max_distance
        self.num_rel_embeddings = max_distance + (max_distance - 1)

        if self.mode == "key" or self.mode == "both":
            self.P_k = nn.Parameter(
                data=torch.randn(self.num_rel_embeddings, self.n_heads)
            )
        else:
            self.P_k = None

        if self.mode == "value" or self.mode == "both":
            self.P_v = nn.Parameter(
                data=torch.randn(self.num_rel_embeddings, n_heads, self.d_v)
            )
        else:
            self.P_v = None

    def generate_rel_position_embedding_matrices(self, sequence_length):
        if sequence_length > self.max_distance:
            raise ValueError(
                f"Received a sequence length of {sequence_length},"
                f"but we are only equipped for tokens up to {self.max_distance} tokens apart."
            )
        zero_offset_idx = self.num_rel_embeddings // 2
        A_k = []
        A_v = []
        for i in range(sequence_length):
            if self.mode == "key" or self.mode == "both":
                A_k.append(
                    self.P_k[
                        zero_offset_idx - i : zero_offset_idx + sequence_length - i
                    ]
                )
            if self.mode == "value" or self.mode == "both":
                A_v.append(
                    self.P_v[
                        zero_offset_idx - i : zero_offset_idx + sequence_length - i
                    ]
                )

        if self.mode == "key" or self.mode == "both":
            A_k = torch.stack(A_k)
        if self.mode == "value" or self.mode == "both":
            A_v = torch.stack(A_v)

        return A_k, A_v

    def forward(self, X, mask=None, return_attn=False):
        """
        X - tensor of shape batch_size, sequence_length, input_dim
        mask - tensor of shape batch_size, n_heads, sequence_length, sequence_length
        """
        B, n_tokens, d = X.shape

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        Q = einops.rearrange(Q, "b n (h d_k) -> b h n d_k", d_k=self.d_k)
        K_t = einops.rearrange(K, "b n (h d_k) -> b h d_k n", d_k=self.d_k)
        V = einops.rearrange(V, "b n (h d_v) -> b h n d_v", d_v=self.d_v)

        A_k, A_v = self.generate_rel_position_embedding_matrices(n_tokens)
        if self.mode == "key" or self.mode == "both":
            A_k = einops.repeat(A_k, "n_out n_in h -> b n_out n_in h", b=B)
            A_k = einops.rearrange(A_k, "b n_out n_in h -> b h n_out n_in")

        attn_scores = Q @ K_t
        if self.mode == "key" or self.mode == "both":
            attn_scores = attn_scores + A_k
        attn_scores = attn_scores * self.scale

        if mask is not None:
            attn_scores[mask == 0] = -INF.type(attn_scores.dtype)

        attn = attn_scores.softmax(-1)
        attn_weighted_values = attn @ V  # b h n d_v

        if self.mode == "value" or self.mode == "both":
            # compute value components from relative position embeddings
            A_v = einops.repeat(A_v, "n_out n_in h d_k -> b n_out n_in h d_k", b=B)
            A_v = einops.rearrange(A_v, "b n_out n_in h d_k -> b h n_out n_in d_k")
            attn_broadcast = einops.repeat(attn, "b h n_out n_in -> b h n_out n_in 1")
            attn_weighted_abs_pos_value = (attn_broadcast * A_v).sum(-2)
            attn_weighted_values = attn_weighted_values + attn_weighted_abs_pos_value

        # concatenate across head dim
        attn_weighted_values = einops.rearrange(
            attn_weighted_values, "b h n d_v -> b n (h d_v)"
        )

        layer_output = self.W_o(attn_weighted_values)
        if return_attn:
            return layer_output, attn
        else:
            return layer_output


class TransformerEncoder(nn.Module):
    """
    Main body of the transformer encoder, which processes a sequence of tokens
    to a sequence of contextualized tokens of the same shape.
    """

    def __init__(
        self, n_layers=6, n_heads=8, d_model=512, d_feed_forward=768, dropout=0.1
    ):
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_feed_forward = d_feed_forward
        self.dropout = dropout

        self.attn_blocks = nn.ModuleList(
            [self.build_attn_layer() for _ in range(self.n_layers)]
        )
        self.attn_layernorms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(self.n_layers)]
        )
        self.feed_forward_blocks = nn.ModuleList(
            [MLP(d_model, d_feed_forward, d_model) for _ in range(self.n_layers)]
        )
        self.ff_layernorms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(self.n_layers)]
        )
        self.dropout = nn.Dropout(p=self.dropout)

    def build_attn_layer(self):
        return MultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
        )

    def forward(self, X, mask=None, return_attn=False):
        layer_outputs = []
        attentions = []

        for i in range(self.n_layers):
            if return_attn:
                attn_layer_out, attn = self.attn_blocks[i](
                    X, mask=mask, return_attn=True
                )
            else:
                attn_layer_out = self.attn_blocks[i](X, mask=mask)
                attn = None
            X = self.attn_layernorms[i](X + attn_layer_out)

            ff_out = self.feed_forward_blocks[i](X)
            X = self.ff_layernorms[i](X + ff_out)
            X = self.dropout(X)

            layer_outputs.append(X)
            attentions.append(attn)

        if return_attn:
            return X, {"attentions": attentions, "layer_outputs": layer_outputs}
        else:
            return X


class TransformerEncoderWithRelativePosEmbeddings(TransformerEncoder):
    def __init__(
        self,
        n_layers=6,
        n_heads=8,
        d_model=512,
        d_feed_forward=768,
        dropout=0.1,
        max_distance=512,
        rel_pos_emb_mode: tp.Literal["key", "value", "both"] = "both",
    ):
        self.max_distance = max_distance
        self.rel_pos_emb_mode = rel_pos_emb_mode
        super().__init__(n_layers, n_heads, d_model, d_feed_forward, dropout)

    def build_attn_layer(self):
        return MultiHeadAttentionWithRelativePositionalEmbeddings(
            self.d_model,
            self.n_heads,
            max_distance=self.max_distance,
            mode=self.rel_pos_emb_mode,
        )


class TransformerForSequenceGeneration(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_idx,
        n_layers,
        n_heads,
        d_model,
        d_feed_forward,
        dropout,
        max_len=500,
        pos_emb: tp.Literal["abs", "rel"] = "abs",
        rel_pos_emb_mode: tp.Literal["key", "value", "both", None] = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_feed_forward = d_feed_forward
        self.dropout = dropout

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pos_emb = pos_emb
        self.pad_idx = pad_idx

        self.token_embeddings = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_idx,
        )

        self.positional_embeddings = (
            torch.nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)
            if pos_emb == "abs"
            else None
        )

        self.transformer = (
            TransformerEncoder(n_layers, n_heads, d_model, d_feed_forward, dropout)
            if pos_emb == "abs"
            else TransformerEncoderWithRelativePosEmbeddings(
                n_layers,
                n_heads,
                d_model,
                d_feed_forward,
                dropout,
                max_distance=max_len,
                rel_pos_emb_mode=rel_pos_emb_mode,
            )
        )

        self.classifier = nn.Linear(d_model, self.vocab_size)

        self.register_buffer("position_indices", torch.tensor(range(max_len)))

    def forward(self, X):
        """
        input: batch of sequences of tokens, shape (B, N)
        output: batch of sequences of class scores, shape (B, n, vocab_size)
        """
        B, N = X.shape

        # we need to mask the future tokens
        mask = torch.tril(torch.ones(N, N)).repeat(B, self.n_heads, 1, 1)

        # extract token embeddings
        X = self.token_embeddings(X)
        B, N, D = X.shape

        # get positions - positions are 0, 1, ..., max_len but we only need 0, 1, ..., N
        position_indices = self.position_indices
        positions = position_indices[:N]
        positions = positions.repeat(B, 1)  # B, N positions

        if self.pos_emb == "abs":
            positional_embeddings = self.positional_embeddings(positions)
            X = X + positional_embeddings

        X = self.transformer(X, mask)
        X = self.classifier(X)

        return X


if __name__ == "__main__":
    model = TransformerForSequenceGeneration(
        vocab_size=1000,
        pad_idx=0,
        n_layers=6,
        n_heads=8,
        d_model=512,
        d_feed_forward=768,
        dropout=0.1,
    )
    input = torch.randint(0, 1000, (64, 400))
    import torchinfo

    torchinfo.summary(model, input_data=input)
    print(model(input).shape)
    model = TransformerForSequenceGeneration(
        vocab_size=1000,
        pad_idx=0,
        n_layers=6,
        n_heads=8,
        d_model=512,
        d_feed_forward=768,
        dropout=0.1,
    )
    input = torch.randint(0, 1000, (64, 400))
    import torchinfo

    torchinfo.summary(model, input_data=input)
    print(model(input).shape)
