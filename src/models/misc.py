from src.models.local_encoder import FeatureExtractorWithSpatialSelfAttentionV1
from src.models.utils import FrozenModuleWrapper
from src.models.video_resnet import (
    BaselineResnet,
    VideoResnetWrapperForSequenceRegression,
)
from src.models.causal_conv import ResidualBlock
from src.models.utils import TwoDModuleFor3DSequenceWrapper
import torch
from src.models.causal_conv import ResnetForTrackingEstimation, ConvThenS4Model
from src.utils import load_model_weights
from src.models.spatio_temporal_attn import (
    FeatureExtractorWithSPTAttnAndFMFeaturesLateConcat,
    FeatureExtractorWithSpatialSelfAttentionV0,
    FeatureExtractorWithSPTAttention,
    FeatureExtractorWithSpatialAttentionAndFMFeatures,
    FeatureExtractorWithSPTAttnAndFMFeatures,
    ViTForSpatialAttention,
)
from src.models.utils import FrozenModuleWrapper
from src.models.video_resnet import VideoResnetWrapperForFeatureMaps
from src.models.video_resnet import video_rn18_base, tiny_resnet_base
from src.models.utils import FrozenModuleWrapper
from src.models.video_resnet import VideoResnetWrapperForFeatureMaps
from src.models.model_registry import register_model
from torch import nn 
import os



@register_model
def two_frame_foundation_model(
    *, backbone_lr: float | None = None, freeze_backbone: bool = False
):
    return FoundationModelForTrackingEstimation(backbone_lr, freeze_backbone)


class FoundationModelForTrackingEstimation(nn.Module):
    def __init__(self, backbone_lr: float | None = None, freeze_backbone: bool = False):
        super().__init__()
        from prostNfound.sam_wrappers import build_medsam

        self.freeze_backbone = freeze_backbone

        sam = build_medsam()
        self.image_encoder = sam.image_encoder
        for param in self.image_encoder.parameters():
            if backbone_lr:
                param._optim = {"lr": backbone_lr}

        self.block1 = ResidualBlock(512, 512, 3, 2)
        self.block2 = ResidualBlock(512, 512, 3, 2)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 6)

        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        if x.ndim == 5:
            B, N, C, H, W = x.shape
            assert N == 2, f"Sequence should have length 2"
            assert C == 1, f"Sequence should have 1 channel"
            original_dim = 5
        elif x.ndim == 4:
            B, C, H, W = x.shape
            x = x[:, :, None, :, :]
            original_dim = 4
        else:
            raise ValueError(f"Input tensor must have 4 or 5 dimensions, got {x.ndim}")

        with torch.set_grad_enabled(not self.freeze_backbone):
            im1, im2 = x.unbind(1)  # B 1 H W, B 1 H W
            im1 = im1.repeat_interleave(3, dim=1)
            im2 = im2.repeat_interleave(3, dim=1)

            feat1 = self.image_encoder(im1)
            feat2 = self.image_encoder(im2)

        feats = torch.cat([feat1, feat2], dim=1)
        feats = feats[:, :, None, :, :]

        x = self.block1(feats)
        x = self.block2(x)

        x = x.mean((-1, -2))
        x = rearrange(x, "b c n -> b n c")

        x = self.act(self.fc1(x))
        x = self.fc2(x)

        return x[:, 0, :] if original_dim == 4 else x

    def forward_long_sequence(
        self, x: torch.Tensor, max_batch_size_for_computation: int = 100
    ):
        B, N, C, H, W = x.shape
        assert N > 2, f"Sequence should have length > 2"
        return TwoDModuleFor3DSequenceWrapper(
            self, max_batch_size_for_computation=max_batch_size_for_computation
        )(x)


@register_model
def causal_conv_rn():
    return ResnetForTrackingEstimation()


@register_model
def conv_rn():
    return ResnetForTrackingEstimation(conv_layer=nn.Conv3d)


@register_model
def conv_rn_groupnorm():
    return ResnetForTrackingEstimation(norm="group", conv_layer=nn.Conv3d)


@register_model
def conv_s4():
    return ConvThenS4Model()


@register_model
def conv_s4_control():
    """Create a conv s4 model with no sequence model, which should be very
    similar to just a 2-frame convolution model.
    """
    model = ConvThenS4Model()
    model.seq_model = nn.Linear(512, 6)

    return model


@register_model
def resnet10_baseline():
    return BaselineResnet


@register_model
def video_rn18_feature_extractor():

    return VideoResnetWrapperForFeatureMaps(video_rn18_base())


@register_model
def tiny_resnet_feature_extractor():
    return VideoResnetWrapperForFeatureMaps(tiny_resnet_base())


@register_model
def video_rn18():
    from src.models.video_resnet import video_rn18_base

    backbone = video_rn18_base()
    return VideoResnetWrapperForSequenceRegression(backbone)


@register_model 
def video_resnet_sequence_regressor(variant=None): 
    from src.models import video_resnet
    if variant is None: 
        backbone = video_resnet.video_rn18_base()
        num_features = 512
    elif variant == 'tiny': 
        backbone = video_resnet.tiny_resnet_base()
        num_features = 256 
    else: 
        raise ValueError(f'unknown variant {variant}')

    return VideoResnetWrapperForSequenceRegression(backbone, num_features=num_features)


@register_model
def video_rn18_frozen_backbone_classifier(backbone_weights_path=None):

    model = video_rn18()
    load_model_weights(model, backbone_weights_path)

    from src.models.video_resnet import VideoResnetWrapperForFeatureMaps

    feature_maps_model = VideoResnetWrapperForFeatureMaps(model.backbone)

    class FrozenBackboneClassifier(nn.Module):
        def __init__(self, feature_maps_model):
            super().__init__()
            self.feature_maps_model = feature_maps_model
            self.fc = nn.Linear(512, 6)

        def train(self, mode: bool = True):
            super().train(mode)
            self.feature_maps_model.eval()

        def forward(self, x):
            with torch.no_grad():
                x = self.feature_maps_model(x)
                x = x.mean(dim=(-2, -1))

            x = self.fc(x)[:, 1:, :]
            return x

    return FrozenBackboneClassifier(feature_maps_model)


@register_model
def video_rn18_plus_attention(
    backbone_weights_path=None, freeze_backbone=True, **kwargs
):

    model = video_rn18()
    load_model_weights(model, backbone_weights_path)

    backbone = model.backbone
    from src.models.video_resnet import VideoResnetWrapperForFeatureMaps

    backbone = VideoResnetWrapperForFeatureMaps(backbone)
    backbone = FrozenModuleWrapper(backbone, freeze_backbone)

    from src.models.spatio_temporal_attn import (
        FeatureExtractorWithSpatialSelfAttention,
    )

    return FeatureExtractorWithSpatialSelfAttention(backbone)


@register_model
def vision_attn_network_v0(img_size=(256, 256)):
    from src.models.new_attn_models import VisionAttentionNetwork

    return VisionAttentionNetwork(img_size=img_size)


@register_model
def usfm_backbone(
    pretrained_path=None, image_size=512, **kwargs
):
    from .usfm import get_usfm_backbone
    
    # Use environment variable or default path
    if pretrained_path is None:
        pretrained_path = os.environ.get("USFM_PRETRAINED_PATH", "trained_models/USFM_latest.pth")

    return get_usfm_backbone(pretrained_path, image_size, **kwargs)


@register_model
def lively_blaze_video_rn18(freeze: bool = False, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = os.environ.get("LIVELY_BLAZE_CHECKPOINT", "experiments/good_runs/lively-blaze-contd/checkpoint/best.pt")

    from src.models.model_registry import video_rn18
    from src.models.utils import FrozenModuleWrapper
    from src.models.video_resnet import VideoResnetWrapperForFeatureMaps

    model = video_rn18()
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)

    model = VideoResnetWrapperForFeatureMaps(model.backbone)
    model = FrozenModuleWrapper(model, frozen=freeze)

    return model


@register_model
def rn18_with_spatiotemporal_attn(pretrained=False):
    """ """

    from src.models.model_registry import video_rn18
    from src.models.utils import FrozenModuleWrapper
    from src.models.video_resnet import VideoResnetWrapperForFeatureMaps
    from src.models.spatio_temporal_attn import (
        FeatureExtractorWithSPTAttention,
    )

    # build backbone
    model = video_rn18()
    model = VideoResnetWrapperForFeatureMaps(model.backbone)
    model = FrozenModuleWrapper(model, frozen=False)
    backbone = model

    crop_size = 256, 256
    feature_map_size = 256 // 16, 256 // 16
    num_features = 512

    model = FeatureExtractorWithSPTAttention(backbone, feature_map_size, num_features)
    if not pretrained:
        return model

    if pretrained == True:
        path = os.environ.get("LYRIC_DRAGON_CHECKPOINT", "experiments/good_runs/lyric-dragon/checkpoint/best.pt")
    elif pretrained == "lyric_dragon":
        path = os.environ.get("LYRIC_DRAGON_CHECKPOINT", "experiments/good_runs/lyric-dragon/checkpoint/best.pt")
    elif os.path.exists(pretrained):
        path = pretrained
    else:
        raise ValueError(f"Cannot load pretrained {pretrained}")

    model.load_state_dict(
        torch.load(
            path,
            map_location="cpu",
        )
    )

    return model


@register_model
def cnn_with_spatial_attn(
    backbone_name="video_rn18",
    backbone_checkpoint=None,
    backbone_kwargs={},
    freeze_backbone=True,
    feature_map_size=16, 
    num_features=512,
    **kwargs
):
    """Builds a feature extractor that converts image sequence inputs to sequences of single feature extractors"""

    backbone = get_model(backbone_name, **backbone_kwargs)
    if backbone_checkpoint:
        state = torch.load(backbone_checkpoint, weights_only=True, map_location="cpu")
        backbone.load_state_dict(state)

    backbone = VideoResnetWrapperForFeatureMaps(backbone.backbone)
    backbone = FrozenModuleWrapper(backbone, frozen=freeze_backbone)

    model = FeatureExtractorWithSpatialSelfAttentionV1(
        backbone, 
        feature_map_size=feature_map_size, 
        num_features=num_features, 
        features_only=True, 
        **kwargs
    )

    model.num_features = 64
    model.sequence_output_type = 'feature'

    return model


@register_model
def spt_attn_v0(
    backbone_name="video_rn18",
    backbone_kwargs={},
    backbone_weights=None,
    freeze_backbone: bool = True,
    feature_map_size: int = 16,
    num_features: int = 512,
    **kwargs,
):
    if backbone_weights is None:
        backbone_weights = os.environ.get("LIVELY_BLAZE_CHECKPOINT", "experiments/good_runs/lively-blaze-contd/checkpoint/best.pt")
    
    backbone = get_model(backbone_name, **backbone_kwargs)
    if backbone_weights:
        state = torch.load(backbone_weights, weights_only=True, map_location="cpu")
        backbone.load_state_dict(state)

    backbone = VideoResnetWrapperForFeatureMaps(backbone.backbone)
    backbone = FrozenModuleWrapper(backbone, frozen=freeze_backbone)

    model = FeatureExtractorWithSPTAttention(
        backbone, feature_map_size, num_features, **kwargs
    )

    return model


@register_model
def spt_attn_fm_features(
    backbone_name="video_rn18",
    backbone_kwargs={},
    backbone_weights=None,
    fm_backbone_name=None,
    fm_backbone_kwargs={},
    fm_backbone_weights=None,
    freeze_backbone: bool = True,
    feature_map_size=16,
    fm_feature_map_size=14,
    num_features=512,
    num_fm_features=64,
    abs_pos_variant=False,
    **kwargs,
):
    if backbone_weights is None:
        backbone_weights = os.environ.get("LIVELY_BLAZE_CHECKPOINT", "experiments/good_runs/lively-blaze-contd/checkpoint/best.pt")
    
    cfg = {}
    if abs_pos_variant == "abs_pos":
        cfg["temporal_attn_type"] = "roformer"
        cfg["position_embedding_type"] = "absolute"

    cfg.update(kwargs)

    backbone = get_model(backbone_name, **backbone_kwargs)
    if backbone_weights:
        state = torch.load(backbone_weights, weights_only=True, map_location="cpu")
        backbone.load_state_dict(state)

    backbone = VideoResnetWrapperForFeatureMaps(backbone.backbone)
    backbone = FrozenModuleWrapper(backbone, frozen=freeze_backbone)

    if fm_backbone_name:
        fm_backbone = get_model(fm_backbone_name, **fm_backbone_kwargs)
        if fm_backbone_weights:
            state = torch.load(
                fm_backbone_weights, weights_only=True, map_location="cpu"
            )
            fm_backbone.load_state_dict(fm_backbone)
    else:
        fm_backbone = None

    model = FeatureExtractorWithSPTAttnAndFMFeatures(
        backbone,
        fm_backbone,
        feature_map_size=feature_map_size,
        num_features=num_features,
        num_fm_features=num_fm_features,
        fm_feature_map_size=fm_feature_map_size,
        **cfg,
    )

    return model


@register_model
def spt_attn_fm_features_abs_pos(**kwargs): ...


@register_model
def spt_attn_fm_feats_concat(
    backbone_name="video_rn18",
    backbone_kwargs={},
    backbone_weights=None,
    freeze_backbone: bool = True,
    feature_map_size=16,
    fm_feature_map_size=14,
    num_features=512,
    num_fm_features=64,
    abs_pos_variant=False,
): ...


@register_model
def spt_attn_sparse_fm(
    fm_backbone_name="usfm_for_3d_feature_maps",
    projection_dim=64,
    lora_rank=16,
    image_size=224,
    **kwargs,
):

    fm_backbone_kwargs = {
        "projection_dim": projection_dim,
        "lora_rank": lora_rank,
        "image_size": image_size,
    }
    fm_backbone_kwargs.update(kwargs.pop("fm_backbone_kwargs", {}))

    return spt_attn_fm_features(
        fm_backbone_name=fm_backbone_name,
        fm_backbone_kwargs=fm_backbone_kwargs,
        **kwargs,
    )


@register_model
def usfm_for_3d_feature_maps(
    image_size=512,
    projection_dim=None,
    lora_rank: int | None = None,
    weights_path=None,
    max_chunk_size: int | None = None,
    frozen=False,
):
    """Builds a foundation model wrapped appropriately to be able to provide 3d feature maps.

    The model receives as input B N C H W tracking sequence and returns B N C' H' W' feature map
    sequence.
    """
    
    if weights_path is None:
        weights_path = os.environ.get("USFM_PRETRAINED_PATH", "trained_models/USFM_latest.pth")

    from src.models.usfm import get_usfm_backbone, USFMWrapperFor3DFeatureMaps

    backbone = get_usfm_backbone(weights_path, image_size)

    if lora_rank:
        from src.models.utils import apply_lora_conversion

        apply_lora_conversion(backbone, lora_rank)

    fm_model = FrozenModuleWrapper(
        USFMWrapperFor3DFeatureMaps(
            backbone,
            output_axes="b n c h w",
            projection_dim=projection_dim,
        ),
        frozen=frozen,
    )
    fm_model.feature_map_size = image_size[0] // 16 if isinstance(image_size, Sequence) else image_size // 16
    fm_model.num_features = projection_dim or 768

    return fm_model


@register_model
def sam_for_3d_feature_maps(
    variant="medsam", projection_dim=None, frozen=False, lora_rank=None, image_size=None
):
    if lora_rank:
        warnings.warn(f"Lora rank set but not supported for this model")

    from prostNfound import sam_wrappers as sm

    class SamWrapperFor3dFeatureMaps(nn.Module):
        def __init__(self, sam, projection_dim=None):
            super().__init__()
            self.image_encoder = sam.image_encoder
            if projection_dim is not None:
                self.projector = nn.Conv2d(256, projection_dim, 1, 1, 0, bias=False)
            else:
                self.projector = None

        def forward(self, x):
            B, N, C, H, W = x.shape
            x = einops.rearrange(x, "b n c h w -> (b n) c h w")
            x = self.image_encoder(x)
            if self.projector is not None:
                x = self.projector(x)
            x = einops.rearrange(x, "(b n) c h w -> b n c h w", b=B)
            return x

    if variant == "medsam":
        sam = sm.build_medsam()
    elif variant == "medsam_adapter":
        sam = sm.build_adapter_medsam_256()
    elif variant == "sam":
        sam = sm.build_sam()
    elif variant == "sammed_2d":
        sam = sm.build_sammed_2d()
    else:
        raise NotImplementedError(variant)

    fm_model = FrozenModuleWrapper(
        SamWrapperFor3dFeatureMaps(sam, projection_dim=projection_dim), frozen=frozen
    )

    return fm_model


@register_model
def vit_for_spatial_attention(image_size=16, patch_size=1, **kwargs):
    return ViTForSpatialAttention(
        image_size=image_size, patch_size=patch_size, **kwargs
    )


@register_model
def ibot_vit_for_video_feature_extraction(pretrained_path=None, **kwargs):
    path = "external/ibot"
    sys.path.append(path)
    from models.vision_transformer import vit_tiny

    model = vit_tiny(**kwargs)
    state = torch.load(pretrained_path, map_location='cpu')
    state = state["teacher"]
    state = {
        k.replace("backbone.", ""): v
        for k, v in state.items()
        if k.startswith("backbone")
    }

    class ViTWrapper(nn.Module):
        def __init__(self, vit):
            super().__init__()
            self.vit = vit

        def forward(self, x):
            B, N, C, H, W = x.shape
            x = einops.rearrange(x, "b n ... -> (b n) ...")
            x = self.vit(x)  # (b n) d
            x = einops.rearrange(x, "(b n) ... -> b n ...", b=B)
            return x

    backbone = ViTWrapper(model)
    backbone.num_features = 192
    return backbone


@register_model 
def dinov2_pretrained_for_feature_extraction(): 
    
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    
    class ViTWrapper(nn.Module):
        def __init__(self, vit):
            super().__init__()
            self.vit = vit

        def forward(self, x):
            B, N, C, H, W = x.shape
            x = einops.rearrange(x, "b n ... -> (b n) ...")
            x = self.vit(x)  # (b n) d
            x = einops.rearrange(x, "(b n) ... -> b n ...", b=B)
            return x

    backbone = ViTWrapper(dinov2_vits14)
    backbone.num_features = 384 
    return backbone


@register_model 
def simple_sparse_tracking_estimator_v2(
    backbone='usfm', 
    image_size=224,
    hidden_size=256,
    pretrained=None,
    in_chans=3,
    backbone_path='auto',
    freeze_backbone=False,
):

    if pretrained == 'vital-butterfly':
        return get_model(
            'simple_sparse_tracking_estimator_v2',
            backbone="image_resnet_avgpool", 
            hidden_size = 512,
            checkpoint = "experiments/good_runs/vital-butterfly/checkpoint/best-hacked_for_other_model.pt"
        )

    from src.models.sparse_context_models import SimpleModelForSparseTrackingEstimation, SpatialMeanPooling

    if backbone == "usfm":
        backbone = usfm_for_3d_feature_maps(image_size, projection_dim=64, lora_rank=32)
        backbone = FeatureExtractorWithSpatialSelfAttentionV1(
            backbone,
            feature_map_size=backbone.feature_map_size,
            num_features=backbone.num_features,
            patch_size=2,
            features_only=True,
        )
        backbone.num_features = 64
    elif backbone == "ibot":
        if backbone_path == 'auto': 
            backbone_path = "external/ibot/logs/2025-02-08/checkpoint0160.pth"
        backbone = ibot_vit_for_video_feature_extraction(
            pretrained_path=backbone_path, in_chans=in_chans
        )
    elif backbone == "image_resnet_avgpool":
        from src.models.video_resnet import (
            VideoResnetWrapperForFeatureMaps,
            video_rn18_no_temporal,
        )

        backbone = VideoResnetWrapperForFeatureMaps(video_rn18_no_temporal())

        backbone = nn.Sequential(backbone, SpatialMeanPooling())
        backbone.num_features = 512
    elif backbone == "medsam":
        from src.models.model_registry import sam_for_3d_feature_maps

        backbone = sam_for_3d_feature_maps("medsam")
        backbone = nn.Sequential(backbone, SpatialMeanPooling())
        backbone.num_features = 256
    else:
        raise NotImplementedError(backbone)

    model = SimpleModelForSparseTrackingEstimation(backbone, hidden_size=hidden_size, proj_bias=True)
    return model


@register_model
def lyric_dragon_for_feature_extraction():
    base = get_model("spt_attn_v0", checkpoint="lyric-dragon")

    class LyricDragonWrapper(nn.Module):
        num_features = 64

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
            cls_tokens = einops.rearrange(cls_tokens, "(b n) ... -> b n ...", b=B)

            return cls_tokens

    return LyricDragonWrapper(base)