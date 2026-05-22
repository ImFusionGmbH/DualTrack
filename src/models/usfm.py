# Copied from https://github.com/openmedlab/USFM under licence CC BY-NC 4.0.
# Adapted from the original code to fit the project structure, exposing an easy model 
# constructor and loading pre-trained weights.

import logging
import math
from functools import partial
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from scipy import interpolate
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import os
import numpy as np
import torch
import torch.distributed as dist
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator as RGI
from einops import rearrange


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the original BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        window_size=None,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (
                2 * window_size[1] - 1
            ) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads)
            )  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = (
                coords_flatten[:, :, None] - coords_flatten[:, None, :]
            )  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(
                1, 2, 0
            ).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = torch.zeros(
                size=(window_size[0] * window_size[1] + 1,) * 2,
                dtype=relative_coords.dtype,
            )
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)

            # trunc_normal_(self.relative_position_bias_table, std=.0)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1,
                -1,
            )  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        window_size=None,
        attn_head_dim=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values is not None:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones(dim), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones(dim), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(
                self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias)
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME: Relax size constraints to support variable input sizes
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding Extract feature map from CNN, flatten, project to embedding
    dim."""

    def __init__(
        self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME: This is a temporary workaround - implement proper dimension detection
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[
                    -1
                ]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (
            2 * window_size[1] - 1
        ) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1,
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class HVITBackbone4Seg(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=80,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=None,
        init_values=None,
        use_checkpoint=False,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        pretrained=None,
        out_indices=[3, 5, 7, 11],
    ):

        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        num_patches = self.patch_embed.num_patches
        self.out_indices = out_indices

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.patch_shape, num_heads=num_heads
            )
        else:
            self.rel_pos_bias = None

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=(
                        self.patch_embed.patch_shape if use_rel_pos_bias else None
                    ),
                )
                for i in range(depth)
            ]
        )

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        # trunc_normal_(self.mask_token, std=.02)
        self.out_indices = out_indices

        if patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.Identity()

            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif patch_size == 8:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Identity()

            self.fpn3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4),
            )
        # Georgejiao: add the weights
        self.apply(self._init_weights)
        self.fix_init_weight()
        if pretrained is not None:
            self.init_weights(pretrained)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        # NOTE pretrained weights will be loaded in the build function
        # if isinstance(pretrained, str):
        #     self.apply(_init_weights)
        #     load_pretrained(self, pretrained, strict=False)
        # elif pretrained is None:
        #     self.apply(_init_weights)
        # else:
        #     raise TypeError("pretrained must be a str or None")
        self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        features = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            else:
                x = blk(x, rel_pos_bias)
            if i in self.out_indices:
                xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                features.append(xp.contiguous())

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        return tuple(features)

    def forward(self, x):
        x = self.forward_features(x)
        return x


def get_usfm_backbone(
    pretrained_path=None,
    image_size=512,
):
    logging.info(
        f"Building USFM from path {pretrained_path} with overrides {image_size=}"
    )
    config = {
        "type": "HVITBackbone4Seg",
        "pretrained": pretrained_path,
        "img_size": image_size,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "drop_path_rate": 0.1,
        "attn_drop_rate": 0.0,
        "drop_rate": 0.0,
        "out_indices": [5, 7, 11],
        "use_abs_pos_emb": False,
        "use_rel_pos_bias": True,
    }

    model_cfg = config.copy()
    model_cfg.pop("type")
    model = HVITBackbone4Seg(**model_cfg)

    if pretrained_path:
        load_pretrained(
            OmegaConf.create(config), model, logging.getLogger("load_model")
        )

    return model


class USFMWrapperFor3DFeatureMaps(nn.Module):
    def __init__(
        self, usfm, projection_dim=None, chunk_size=None, output_axes="b c n h w"
    ):
        super().__init__()
        self.usfm = usfm
        self.chunk_size = chunk_size
        self.output_axes = output_axes

        if projection_dim is not None:
            self.projector = nn.Conv3d(768, projection_dim, 1, 1, 0)
        else:
            self.projector = nn.Identity()

    def forward(self, images):
        B, N, C, H, W = images.shape

        images = rearrange(images, "b n c h w -> (b n) c h w")

        if self.chunk_size is not None:
            all_fm_features = []
            n = images.shape[0]
            chunk_size = self.chunk_size
            n_chunks = (n // chunk_size) + 1
            for image_subsequence in torch.chunk(images, n_chunks):

                current_chunk_size = image_subsequence.shape[0]
                features = self.usfm(image_subsequence)[-1]
                features = features[:, :, None, :, :]  # (b n) c 1 h w
                features = self.projector(features)  # (b n) proj_dim 1 h w
                features = features[:, :, 0, :, :]  # (b n) proj_dim h w

                all_fm_features.append(features)

            features = torch.cat(all_fm_features, dim=0)
            features = rearrange(features, f"(b n) c h w -> {self.output_axes}", b=B)
            return features

        else:
            fm_features = self.usfm(images)[-1]
            fm_features = rearrange(fm_features, "(b n) c h w -> b c n h w", b=B)
            fm_features = self.projector(fm_features)
            fm_features = rearrange(
                fm_features, f"b c n h w -> {self.output_axes}", b=B
            )
            return fm_features


def remap_pretrained_keys_vit(model, checkpoint_model, logger):
    # Duplicate shared rel_pos_bias to each layer
    if (
        getattr(model, "use_rel_pos_bias", False)
        and "rel_pos_bias.relative_position_bias_table" in checkpoint_model
    ):
        logger.info(
            "Expand the shared relative position embedding to each transformer block."
        )
    num_layers = model.get_num_layers()
    rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
    for i in range(num_layers):
        checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = (
            rel_pos_bias.clone()
        )
    checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")

    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (
                dst_patch_shape[1] * 2 - 1
            )
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                logger.info(
                    "Position interpolate for %s from %dx%d to %dx%d"
                    % (key, src_size, src_size, dst_size, dst_size)
                )
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r**n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.090307:
                #     q = 1.090307

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                # logger.info("Original positions = %s" % str(x))
                # logger.info("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                xi, yi = np.meshgrid(dx, dy, indexing="ij")
                points = np.array([xi.ravel(), yi.ravel()]).T

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = RGI((x, y), z.T, method="cubic", bounds_error=False)
                    # 进行插值
                    all_rel_pos_bias.append(
                        torch.Tensor(f(points).reshape(xi.shape))
                        .contiguous()
                        .view(-1, 1)
                        .to(rel_pos_bias.device)
                    )

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                checkpoint_model[key] = new_rel_pos_bias

    return checkpoint_model


def load_pretrained(model_cfg, model, logger):
    logger.info(f">>>>>>>>>> Fine-tuned from {model_cfg.pretrained} ..........")
    checkpoint = torch.load(model_cfg.pretrained, map_location="cpu")
    checkpoint_model = checkpoint

    if "swin" in model_cfg.type.lower():
        raise NotImplementedError()
        # logger.info(">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
        # checkpoint_model = remap_pretrained_keys_swin(model, checkpoint_model, logger)
    elif "vit" in model_cfg.type.lower():
        logger.info(">>>>>>>>>> Remapping pre-trained keys for VIT ..........")
        checkpoint_model = remap_pretrained_keys_vit(model, checkpoint_model, logger)
    else:
        raise NotImplementedError

    msg = model.load_state_dict(checkpoint_model, strict=False)
    logger.info(msg)

    del checkpoint_model
    torch.cuda.empty_cache()
    logger.info(f">>>>>>>>>> loaded successfully '{model_cfg.pretrained}'")


if __name__ == "__main__":
    with torch.no_grad():
        from scripts.train_usfm_rn18_finetune import get_foundation_model

        model = get_foundation_model(projection_dim=1).cuda().eval()
        inp = torch.randn(1, 512, 3, 224, 224).cuda()
        out1 = model(inp)
        out2 = model(inp)
        print(torch.allclose(out1, out2))
        model.module.chunk_size = 4
        print(torch.allclose(out1, out2))
