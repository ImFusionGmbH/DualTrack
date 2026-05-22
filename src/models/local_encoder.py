import logging
from warnings import warn
from torch import nn
import einops
import torch

from src.engine.tracking_estimator import (
    BaseTrackingEstimator,
    LocalEncoderTrackingEstimator,
)
from src.models.utils import (
    FrozenModuleWrapper,
    apply_model_chunked,
    temporal_tiled_exact,
)
from src.models.spatio_temporal_attn import SimpleTemporalAttn
from .model_registry import get_model, register_model as _register_model
from src.models.video_resnet import VideoResnetWrapperForFeatureMaps


MODELS = []


def register_model(fn):
    MODELS.append(fn.__name__)
    return _register_model(fn)


class VideoResnetTrackingEstimator(LocalEncoderTrackingEstimator, nn.Module):
    def __init__(
        self,
        backbone,
        n_feats=512,
        features_only=False,
        max_subsequence_size=None,
        output_features="pooled",
    ):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(n_feats, 6)
        self.features_only = features_only
        self.max_subsequence_size = max_subsequence_size
        self.output_features = output_features

    def forward(self, x):

        B, N, C, H, W = x.shape
        x = einops.rearrange(x, "b n c h w -> b c n h w")

        if self.max_subsequence_size and N > self.max_subsequence_size:
            roi_t = self.max_subsequence_size
            halo_t = 32  # >= ~RF/2; bump to 48–64 if you still see seams
            features = temporal_tiled_exact(
                self.backbone, x, roi_t, halo_t, amp=False, cpu_agg=False
            )
        else:
            features = self.backbone(x)

        if self.features_only and not self.output_features == "pooled":
            return einops.rearrange(features, "b c n h w -> b n c h w")

        features_pooled = features.mean(
            (-2, -1)
        )  # TODO: Implement adaptive pooling based on input size
        features_pooled = einops.rearrange(features_pooled, "b c n -> b n c")
        if self.features_only:
            return features_pooled

        return self.fc(features_pooled)[:, 1:, :]


@register_model
def vidrn18_small_window_trck_reg(**kwargs):
    from src.models.video_resnet import video_rn18_small_temporal_context

    return VideoResnetTrackingEstimator(video_rn18_small_temporal_context(), **kwargs)


@register_model
def vidrn18_trck_reg(**kwargs):
    from src.models.video_resnet import video_rn18_base

    return VideoResnetTrackingEstimator(video_rn18_base(), **kwargs)


@register_model
def vidrn18_trck_reg_causal(**kwargs):
    from src.models.video_resnet import causal_video_rn18_base

    return VideoResnetTrackingEstimator(causal_video_rn18_base(), **kwargs)


@register_model
def vidrn18_small_window_trck_reg_causal(**kwargs):
    from src.models.video_resnet import causal_video_rn18_small_temporal_context

    return VideoResnetTrackingEstimator(
        causal_video_rn18_small_temporal_context(), **kwargs
    )


@register_model
def vidrn18_trck_reg_tiny(**kwargs):
    from src.models.video_resnet import tiny_resnet_base

    return VideoResnetTrackingEstimator(tiny_resnet_base(), n_feats=256, **kwargs)


class FeatureExtractorWithSpatialSelfAttentionV1(
    LocalEncoderTrackingEstimator, nn.Module
):
    def __init__(
        self,
        backbone,
        feature_map_size=16,
        num_features=512,
        patch_size=2,
        features_only=False,
        max_subsequence_size=None,
        hidden_size=64,
        num_hidden_layers=4,
        intermediate_size=32,
        num_attention_heads=4,
        backbone_features_key=None,
        input_type='frames',
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_map_size = feature_map_size
        self.patch_size = patch_size
        self.features_only = features_only

        if max_subsequence_size is not None: 
            warn('Setting max_subsequence size in the module uses an inexact chunking implementation.')
        self.max_subsequence_size = max_subsequence_size
        self.backbone_features_key = backbone_features_key
        self.input_type = input_type

        # use components from huggingface to create the self-attention layers
        from transformers.models.vit import ViTConfig, ViTModel

        vit_config = ViTConfig(
            image_size=feature_map_size,
            patch_size=patch_size,
            num_channels=num_features,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            **kwargs,
        )
        self.vit = ViTModel(
            vit_config,
            add_pooling_layer=False,
        )
        self.fc = torch.nn.Linear(hidden_size, 6)

    def forward(self, x):
        if self.input_type == 'frames':
            return self._forward_impl(frames=x)
        elif self.input_type == 'features':
            return self._forward_impl(backbone_feature_maps=x)
        else:
            raise ValueError(f"Unknown input type: {self.input_type}")

    def _forward_impl(self, frames=None, backbone_feature_maps=None):

        if backbone_feature_maps is None:
            assert frames is not None
            B, N, C, H, W = frames.shape
            if self.max_subsequence_size and N > self.max_subsequence_size:
                backbone_feature_maps = apply_model_chunked(
                    self.backbone, frames, self.max_subsequence_size, 1
                )
            else:
                backbone_feature_maps = self.backbone(frames)

        B, N, C, H, W = backbone_feature_maps.shape
        features = einops.rearrange(
            backbone_feature_maps, "b n c h w -> (b n) c h w"
        )  # fold sequence dim into batch dim

        vit_output = self.vit(features).last_hidden_state
        cls_tokens = vit_output[:, 0, :]

        cls_tokens = einops.rearrange(cls_tokens, "(b n) c -> b n c", b=B, n=N)
        if self.features_only:
            return cls_tokens

        outputs = self.fc(cls_tokens)
        return outputs[:, 1:, :]

    def predict(self, batch) -> torch.Tensor:

        if self.backbone_features_key is not None:
            backbone_features = batch[self.backbone_features_key].to(self.device)
            frames = None
        else:
            frames = batch["images"].to(self.device)
            backbone_features = None

        return self(frames, backbone_features)


@register_model
def cnn_sp_attn(
    *,
    backbone_cfg=dict(name="vidrn18_small_window_trck_reg"),
    freeze_backbone=False,
    implementation_version=0,
    **kwargs,
):
    backbone = get_model(**backbone_cfg).backbone
    
    if implementation_version == 0: 
        backbone = VideoResnetWrapperForFeatureMaps(backbone)
        backbone = FrozenModuleWrapper(backbone, frozen=freeze_backbone)

    else: 
        max_subsequence_size = kwargs.pop('max_subsequence_size')
        backbone = VideoResnetWrapperForFeatureMaps(backbone, max_subsequence_size=max_subsequence_size)
        backbone = FrozenModuleWrapper(backbone, frozen=freeze_backbone)

    model = FeatureExtractorWithSpatialSelfAttentionV1(backbone=backbone, **kwargs)
    return model


@register_model
def cnn_sp_attn_v2(
    *,
    backbone_cfg=dict(
        name="vidrn18_small_window_trck_reg",
        features_only=True,
        output_features="pre-pool",
    ),
    freeze_backbone=False,
    **kwargs,
):
    backbone = get_model(**backbone_cfg)
    backbone = FrozenModuleWrapper(backbone, frozen=freeze_backbone)

    model = FeatureExtractorWithSpatialSelfAttentionV1(backbone=backbone, **kwargs)
    return model


class LocalEncoderSPTAttn(
    LocalEncoderTrackingEstimator, nn.Module
):
    def __init__(
        self,
        backbone,
        temporal_encoder,
        features_only=False,    
    ):
        super().__init__()
        self.backbone = backbone
        self.temporal_encoder = temporal_encoder
        self.features_only = features_only

    # def forward_intermediates(self, x):
    #     if self.use_cache and hasattr(self, "_current_data_ids"):
    #         B = x.shape[0]
    #         outputs = []
    #         for i in range(B):
    #             if self._current_data_ids[i] in self._cache:
    #                 logging.debug(f"cache hit: {self._current_data_ids[i]}")
    #                 outputs.append(self._cache[self._current_data_ids[i]].to(x.device))
    #             else:
    #                 feat = self.backbone(x[[i]])
    #                 outputs.append(feat[0])
    #                 self._cache[self._current_data_ids[i]] = feat[0].detach().cpu()
# 
    #         return torch.stack(outputs)
# 
    #     else:
    #         x = self.backbone(x)
    #         return x

    def forward(self, x):
        x = self.backbone(x)
        
        if self.features_only:
            return x

        return self.temporal_encoder(x)


class TrackingEstimatorWithSequenceBackbone(nn.Module): 
    def __init__(self, backbone, embed_dim):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, 6)

    def forward(self, x):
        B, N, *_ = x.shape
        x = self.backbone(x)
        x = self.head(x[:, -(N-1):, ...])
        return x


@register_model
def cnn_sp_attn_then_temp_attn(
    *,
    backbone_cfg=dict(name="cnn_sp_attn"),
    freeze_backbone=True,
    keep_backbone_in_eval_mode=None,
    input_mode="features",
    features_only=False,
    hidden_size=64,
    **kwargs,
):
    """
    Attaches a temporal attention module to the output of a pooled  spatial attention module.

    Args:
        backbone_cfg (dict): Configuration for the backbone model (should be cnn).
        freeze_backbone (bool): Whether to freeze the backbone model.
        input_mode (str): Whether the input to the model is images or features. If images, the backbone is applied to the input. If features, the input is assumed to be the output of the backbone.
        **kwargs: Additional arguments to pass to the temporal attention module.
    """

    backbone_cfg = dict(**backbone_cfg)
    backbone_cfg['features_only'] = True
    backbone = get_model(**backbone_cfg)
    backbone = FrozenModuleWrapper(backbone, frozen=freeze_backbone, always_eval_mode=keep_backbone_in_eval_mode)
    temporal_encoder = SimpleTemporalAttn(
        **kwargs, hidden_size=hidden_size, features_only=features_only
    )

    return LocalEncoderSPTAttn(
        backbone, temporal_encoder
    )


@register_model
def cnn_sp_attn_then_roformer(
    *,
    backbone_cfg=dict(name="cnn_sp_attn"),
    freeze_backbone=True,
    roformer_model="roformer_small",
    **kwargs,
):
    """
    Attaches a temporal attention module to the output of a pooled  spatial attention module.

    Args:
        backbone_cfg (dict): Configuration for the backbone model (should be cnn).
        freeze_backbone (bool): Whether to freeze the backbone model.
        input_mode (str): Whether the input to the model is images or features. If images, the backbone is applied to the input. If features, the input is assumed to be the output of the backbone.
        **kwargs: Additional arguments to pass to the temporal attention module.
    """

    backbone_cfg = dict(**backbone_cfg)
    backbone_cfg['features_only'] = True
    backbone = get_model(**backbone_cfg) 
    backbone = FrozenModuleWrapper(backbone, frozen=freeze_backbone)
    
    class RoFormerWrapper(nn.Module): 
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x): 
            x = self.model(x)

            return x['x_norm_patchtokens']

    from .roformer import roformer as roformer_models
    roformer = roformer_models.__dict__[roformer_model](
        **kwargs
    )
    roformer.init_weights()
    embed_dim = roformer.embed_dim
    roformer = RoFormerWrapper(roformer)

    backbone = nn.Sequential(
        backbone, roformer
    )
    
    return TrackingEstimatorWithSequenceBackbone(backbone, embed_dim=embed_dim)


@register_model
def dualtrack_loc_enc_stg1(
    backbone_weights=None, cnn_name="vidrn18_small_window_trck_reg", **kwargs
):
    """Build the first stage (CNN) of the local encoder"""
    return get_model(cnn_name, **kwargs)


@register_model
def dualtrack_loc_enc_stg2(
    *,
    backbone_weights=None,
    cnn_name="vidrn18_small_window_trck_reg",
    freeze_backbone=True,
    **kwargs,
):
    """Build the second stage (CNN + vit) of the local encoder"""

    backbone_kw = kwargs.pop("backbone_cfg", {})

    return cnn_sp_attn(
        backbone_cfg=dict(
            name="dualtrack_loc_enc_stg1",
            checkpoint=backbone_weights,
            cnn_name=cnn_name,
            **backbone_kw,
        ),
        freeze_backbone=freeze_backbone,
        **kwargs,
    )


@register_model
def dualtrack_loc_enc_stg3(
    *,
    backbone_weights=None,
    features_only=False,
    input_mode="features",
    cnn_name="vidrn18_small_window_trck_reg",
    freeze_backbone=True,
    **kwargs,
):
    """Build the third stage (CNN + vit + small transformer) of the local encoder"""
    return cnn_sp_attn_then_temp_attn(
        backbone_cfg=dict(
            name="dualtrack_loc_enc_stg2",
            checkpoint=backbone_weights,
            cnn_name=cnn_name,
            **kwargs.pop("backbone_cfg", {}),
        ),
        input_mode=input_mode,
        features_only=features_only,
        freeze_backbone=freeze_backbone,
        **kwargs,
    )


@register_model
def dualtrack_loc_enc_stg3_legacy(
    *, backbone_weights=None, features_only=False, input_mode="features", **kwargs
):
    """Alias of dualtrack_loc_enc_stg3 that uses the cnn with the larger window.
    our old implementations used this before we found out that the smaller window
    version works just as well. We keep this version for compatibility."""

    return dualtrack_loc_enc_stg3(
        backbone_weights=backbone_weights,
        features_only=features_only,
        input_mode=input_mode,
        cnn_name="vidrn18_trck_reg",
        **kwargs,
    )
