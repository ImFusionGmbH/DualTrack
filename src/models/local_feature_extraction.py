from enum import StrEnum
from attr import dataclass
from src.models.video_resnet import VideoResnetFeaturesOnly
from src.utils.utils import load_model_weights
from .model_registry import register_model as _register_model, get_model
import torch 
from torch import nn 
from einops import rearrange
import einops


MODELS = []


def register_model(fn): 
    MODELS.append(fn.__name__)
    return _register_model(fn)


class VideoResnetWrapperForPooledFeatures(nn.Module):
    def __init__(self, backbone: VideoResnetFeaturesOnly, n_feats=512, features_only=False):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(n_feats, 6)
        self.features_only = features_only

    def forward(self, x):
        x = einops.rearrange(x, "b n c h w -> b c n h w")
        features = self.backbone(x)  # b c n h w
        features_pooled = features.mean(
            (-2, -1)
        )  # TODO: Implement adaptive pooling based on input size
        features_pooled = einops.rearrange(features_pooled, "b c n -> b n c")

        return features_pooled
        # if self.features_only: 
        #     return features_pooled
# 
        # return self.fc(features_pooled)[:, 1:, :]


VIDEO_RESNET_VERSIONS = [
    'rn18',
    'rn18_small_temporal', 
    'causal_rn18', 
    'causal_rn18_small_temporal', 
    'tinyrn'
]


@register_model
def video_resnet_feature_extractor(version="rn18", features_only=True, backbone_checkpoint=None):
    from src.models.video_resnet import (
        video_rn18_base,
        video_rn18_small_temporal_context,
        causal_video_rn18_base,
        causal_video_rn18_small_temporal_context,
        tiny_resnet_base,
    )

    if version == "rn18":
        backbone = video_rn18_base()
    elif version == "rn18_small_temporal":
        backbone = video_rn18_small_temporal_context()
    elif version == "causal_rn18":
        backbone = causal_video_rn18_base()
    elif version == "causal_rn18_small_temporal":
        backbone = causal_video_rn18_small_temporal_context()
    elif version == 'tinyrn': 
        backbone = tiny_resnet_base()
    else:
        raise ValueError()

    if 'rn18' in version: 
        num_features = 512 
    elif 'tiny' in version: 
        num_features = 256
    else: 
        raise ValueError()

    model = VideoResnetWrapperForPooledFeatures(backbone, n_feats=num_features, features_only=features_only)
    if backbone_checkpoint: 
        load_model_weights(model, backbone_checkpoint)
    return model


@register_model
def lyric_dragon_feature_extractor_no_temporal(backbone_checkpoint=None):
    module = get_model("spt_attn_v0", checkpoint="lyric-dragon")

    class SptAttnFeatureExtractionWrapper(torch.nn.Module): 
        def __init__(self, module): 
            super().__init__()
            self.module = module 

        def forward(self, images):
            features = self.module.backbone(images)
            B, N, C, H, W = features.shape
            features = einops.rearrange(
                features, "b n c h w -> (b n) c h w"
            )  # fold sequence dim into batch dim
            vit_output = self.module.vit(features).last_hidden_state
            cls_tokens = vit_output[:, 0, :]

            cls_tokens = einops.rearrange(cls_tokens, "(b n) c -> b n c", b=B, n=N)

            return cls_tokens

    return SptAttnFeatureExtractionWrapper(module)



@register_model
def lyric_dragon_feature_extractor(backbone_checkpoint=None):
    module = get_model("spt_attn_v0", checkpoint=backbone_checkpoint or "lyric-dragon")

    class SptAttnFeatureExtractionWrapper(torch.nn.Module): 
        def __init__(self, module): 
            super().__init__()
            self.module = module 

        def forward(self, images):
            features = self.module.backbone(images)
            B, N, C, H, W = features.shape
            features = einops.rearrange(
                features, "b n c h w -> (b n) c h w"
            )  # fold sequence dim into batch dim
            vit_output = self.module.vit(features).last_hidden_state
            cls_tokens = vit_output[:, 0, :]

            cls_tokens = einops.rearrange(cls_tokens, "(b n) c -> b n c", b=B, n=N)
            cls_tokens = self.module.bert(
                cls_tokens
            ).last_hidden_state

            return cls_tokens

    return SptAttnFeatureExtractionWrapper(module)


