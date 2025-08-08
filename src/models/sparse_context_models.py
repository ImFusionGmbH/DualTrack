from dataclasses import dataclass
from turtle import back
from typing import Literal, Optional
import torch
from torch import nn
from src.models.bert import BertEncoder, BertConfig
import logging


import torch
from torch import nn
import einops

from src.models.model_registry import (
    dinov2_pretrained_for_feature_extraction,
    ibot_vit_for_video_feature_extraction,
    register_model,
    usfm_for_3d_feature_maps,
)
from src.models.local_encoder import FeatureExtractorWithSpatialSelfAttentionV1
from src.utils.utils import load_model_weights


class SimpleModelForSparseTrackingEstimation(nn.Module):

    def __init__(
        self,
        context_image_backbone=None,
        num_context_features=None,
        max_seq_length=1024,
        hidden_size=256,
        pred_mode="local",
        num_hidden_layers=8,
        features_only=False,
        proj_bias=False,
        position_embedding_type="absolute",
        **kwargs,
    ):
        super().__init__()

        self.num_features = hidden_size
        self.features_only = features_only
        self.pred_mode = pred_mode
        self.positon_embedding_type = position_embedding_type

        self.backbone = context_image_backbone
        self.num_context_features = (
            num_context_features or context_image_backbone.num_features
        )
        self.proj = nn.Linear(self.num_context_features, hidden_size, bias=proj_bias)
        self.pos_emb = (
            nn.Embedding(max_seq_length, hidden_size)
            if position_embedding_type == "absolute"
            else None
        )
        cfg = BertConfig(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=8,
            position_embedding_type=position_embedding_type,
            max_position_embeddings=1024,
            **kwargs,
        )
        self.bert = BertEncoder(cfg)
        self.fc = nn.Linear(hidden_size, 6)

    def forward(
        self,
        context_images=None,
        sample_indices=None,
        context_image_features=None,
        return_extras=False,
        **kwargs,
    ):

        assert sample_indices is not None

        if context_image_features is None:
            assert self.backbone is not None
            assert context_images is not None

            # might have to apply indexing
            if context_images.shape[1] != sample_indices.shape[1]:
                images_resampled = []
                for i in range(context_images.shape[0]):
                    images_resampled.append(context_images[i][sample_indices[i]])
                context_images = torch.stack(images_resampled, 0)

            feats = self.backbone(context_images)  # B N C H W
        else:
            if context_image_features.shape[1] != sample_indices.shape[1]:
                context_features_resampled = []
                for i in range(context_image_features.shape[0]):
                    context_features_resampled.append(
                        context_image_features[i][sample_indices[i]]
                    )
                context_image_features = torch.stack(context_features_resampled, 0)
            feats = context_image_features

        feats = self.proj(feats)

        if self.pos_emb is not None:
            pos_emb = self.pos_emb(sample_indices)
            feats = feats + pos_emb

        bert_outputs = self.bert(
            feats,
            position_ids=(
                sample_indices if self.positon_embedding_type != "absolute" else None
            ),
            **kwargs,
        )
        if return_extras:
            return bert_outputs

        bert_outputs = bert_outputs.last_hidden_state

        if self.features_only:
            return bert_outputs

        if self.pred_mode == "local":
            return self.fc(bert_outputs)[:, 1:, :]
        elif self.pred_mode == "global":
            out = {}
            pred = self.fc(bert_outputs)
            pred[:, 0, :] = 0
            out["global"] = pred[:, 1:, :]
            pred = pose_vector_to_rotation_matrix_torch(pred)
            pred = get_relative_transforms_torch(pred)
            pred = rotation_matrix_to_pose_vector_torch(pred)
            out["local"] = pred.detach()
            return out
        elif self.pred_mode == "absolute":
            out = {}
            pred = self.fc(bert_outputs)
            out["absolute"] = pred
            pred = pred.detach()
            pred = pose_vector_to_rotation_matrix_torch(pred)
            pred_glob = get_absolute_to_global_transforms_torch(pred)[..., 1:, :, :]
            pred_glob = rotation_matrix_to_pose_vector_torch(pred_glob)
            out["global"] = pred_glob
            pred_loc = get_relative_transforms_torch(pred)
            pred_loc = rotation_matrix_to_pose_vector_torch(pred_loc)
            out["local"] = pred_loc
            return out
        else:
            raise ValueError()

    def predict(self, batch, **kwargs):
        device = next(self.parameters()).device

        def _get_tensor(key):
            return batch[key].to(device) if key in batch else None

        inputs = dict(
            context_images=_get_tensor("images"),
            context_image_features=_get_tensor("context_features"),
            sample_indices=_get_tensor("sample_indices"),
            **kwargs,
        )

        return self(**inputs)


@dataclass
class ModelBuilder:
    """Model configuration options"""

    backbone: Literal["usfm", "ibot", "image_resnet_avgpool", "medsam", "medsam_batch_norm", "dinov2"] = (
        "usfm"
    )
    backbone_path: Optional[str] = None
    hidden_size: int = 512
    intermediate_size: Optional[int] = None 
    dropout: float = 0.1
    init_weights_path: Optional[str] = None
    in_channels: int = 1
    position_embedding_type: str = "absolute"

    def __post_init__(self): 
        if self.intermediate_size is None: 
            self.intermediate_size = self.hidden_size * 2

    def get_model(self, image_size=[224, 224]):
        backbone = self.get_backbone(image_size=image_size)
        model = SimpleModelForSparseTrackingEstimation(
            backbone,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            position_embedding_type=self.position_embedding_type,
        )
        if self.init_weights_path:
            load_model_weights(model, self.init_weights_path)

        return model

    def get_backbone(self, image_size=None):
        logging.info(f"Creating {self.backbone} backbone.")

        # get the backbone
        if self.backbone == "usfm":
            backbone = usfm_for_3d_feature_maps(
                image_size, projection_dim=64, lora_rank=32
            )
            backbone = FeatureExtractorWithSpatialSelfAttentionV1(
                backbone,
                feature_map_size=backbone.feature_map_size,
                num_features=backbone.num_features,
                patch_size=2,
                features_only=True,
            )
            backbone.num_features = 64
        elif self.backbone == "ibot":
            backbone = ibot_vit_for_video_feature_extraction(
                self.backbone_path
                or "external/ibot/logs/2025-02-08/checkpoint0160.pth",
                in_chans=self.in_channels,
            )
        elif self.backbone == "image_resnet_avgpool":
            from src.models.video_resnet import (
                VideoResnetWrapperForFeatureMaps,
                video_rn18_no_temporal,
            )

            backbone = VideoResnetWrapperForFeatureMaps(video_rn18_no_temporal())

            backbone = nn.Sequential(backbone, SpatialMeanPooling())
            backbone.num_features = 512
        elif self.backbone == "medsam":
            from src.models.model_registry import sam_for_3d_feature_maps

            backbone = sam_for_3d_feature_maps("medsam")
            backbone = nn.Sequential(backbone, SpatialMeanPooling())
            backbone.num_features = 256
        elif self.backbone == 'medsam_batch_norm':
            from src.models.medsam_wrapper import get_wrapped_medsam_encoder
            backbone = nn.Sequential(get_wrapped_medsam_encoder(norm='instance'), SpatialMeanPooling())
            backbone.num_features = 256
        elif self.backbone == "dinov2":
            backbone = dinov2_pretrained_for_feature_extraction()
        else:
            raise NotImplementedError(self.backbone)

        logging.info(f"Created {self.backbone}.")
        return backbone


class SpatialMeanPooling(nn.Module):
    def forward(self, x):
        return x.mean((-1, -2))


@register_model
def global_encoder_backbone(
    backbone_name: Literal[
        "usfm", "ibot", "image_resnet_avgpool", "medsam", "medsam_instance_norm", "dinov2", "identity", "none", "medsam_instance_norm"
    ] = "usfm",
    backbone_num_features: Optional[int] = None,
    backbone_path=None,
    in_channels=None,
    image_size=None,
):
    logging.info(f"Creating {backbone_name} backbone.")

    # get the backbone
    if backbone_name == "usfm":
        assert image_size is not None
        backbone = usfm_for_3d_feature_maps(image_size, projection_dim=64, lora_rank=32)
        backbone = FeatureExtractorWithSpatialSelfAttentionV1(
            backbone,
            feature_map_size=backbone.feature_map_size,
            num_features=backbone.num_features,
            patch_size=2,
            features_only=True,
        )
        backbone.num_features = 64
    elif backbone_name == "ibot":
        backbone = ibot_vit_for_video_feature_extraction(
            backbone_path or "external/ibot/logs/2025-02-08/checkpoint0160.pth",
            in_chans=in_channels,
        )
    elif backbone_name == "image_resnet_avgpool":
        from src.models.video_resnet import (
            VideoResnetWrapperForFeatureMaps,
            video_rn18_no_temporal,
        )

        backbone = VideoResnetWrapperForFeatureMaps(video_rn18_no_temporal())

        backbone = nn.Sequential(backbone, SpatialMeanPooling())
        backbone.num_features = 512
    elif backbone_name == "medsam":
        from src.models.model_registry import sam_for_3d_feature_maps

        backbone = sam_for_3d_feature_maps("medsam")
        backbone = nn.Sequential(backbone, SpatialMeanPooling())
        backbone.num_features = 256
    elif backbone_name == 'medsam_instance_norm': 
        from src.models.medsam_wrapper import get_wrapped_medsam_encoder
        backbone = nn.Sequential(get_wrapped_medsam_encoder(norm='instance'), SpatialMeanPooling())
        backbone.num_features = 256
    elif backbone_name == 'medsam_batch_norm': 
        from src.models.medsam_wrapper import get_wrapped_medsam_encoder
        backbone = nn.Sequential(get_wrapped_medsam_encoder(norm='batch'), SpatialMeanPooling())
        backbone.num_features = 256
    elif backbone_name == "dinov2":
        backbone = dinov2_pretrained_for_feature_extraction()
    elif backbone_name == "identity":
        backbone = nn.Identity()
        backbone.num_features = backbone_num_features
    elif backbone_name == "none":
        backbone = None
    else:
        raise NotImplementedError(backbone_name)

    logging.info(f"Created {backbone_name}.")
    return backbone


@register_model
def sparse_tracking_estimator_with_global_encoder(
    backbone_name: Literal[
        "usfm",
        "ibot",
        "image_resnet_avgpool",
        "medsam",
        "dinov2",
        "identity",
        "none",
    ] = "usfm",
    backbone_num_features: Optional[int] = None,
    backbone_path: Optional[str] = None,
    hidden_size: int = 512,
    dropout: float = 0.1,
    in_channels: int = 1,
    position_embedding_type: str = "absolute",
    image_size=None,
) -> SimpleModelForSparseTrackingEstimation:

    backbone = global_encoder_backbone(
        backbone_name=backbone_name,
        backbone_path=backbone_path,
        in_channels=in_channels,
        image_size=image_size,
        backbone_num_features=backbone_num_features,
    )
    model = SimpleModelForSparseTrackingEstimation(
        backbone,
        num_context_features=backbone_num_features,
        hidden_size=hidden_size,
        dropout=dropout,
        position_embedding_type=position_embedding_type,
    )

    return model
