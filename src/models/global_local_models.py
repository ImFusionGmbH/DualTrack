import argparse
from dataclasses import dataclass, field
import logging
import os
import warnings
from black import Mode
import einops
from omegaconf import OmegaConf
import torch
from torch import nn

from src.models.model_registry import get_model
from src.models.utils import BertWrapper, FrozenModuleWrapper, init_tracking_head
from src.transform import RandomSparseSampleTemporal
from src.utils.utils import UnstructuredArgsAction, load_model_weights

# from transformers.models.bert.modeling_bert import (
#     BertConfig,
#     BertEncoder,
#     BertLayer,
#     BertAttention,
#     BertIntermediate,
#     BertOutput,
# )
from src.models.bert import BertConfig, BertEncoder
from typing import *
from src.models.local_feature_extraction import MODELS
from src.models.model_registry import register_model, get_model
import src.models.sparse_context_models


class SparseSampler(nn.Module):

    def __init__(self, n_samples=128):
        super().__init__()
        self.n_samples = n_samples

    def __call__(self, images, features):
        if images is not None:
            B, N, C, H, W = images.shape
            device = images.device
        else:
            B, N, *_ = features.shape
            device = features.device
        samples = []
        for _ in range(B):
            samples.append(
                torch.sort(torch.randperm(N, device=device)[: self.n_samples]).values
            )
        return torch.stack(samples, 0)


class RegularGridSampler(nn.Module):

    def __init__(self, grid_spacing):
        super().__init__()
        self.grid_spacing = grid_spacing

    def __call__(self, images, features):
        if images is not None:
            B, N, C, H, W = images.shape
            device = images.device
        else:
            B, N, *_ = features.shape
            device = features.device
        samples = []
        for _ in range(B):
            samples.append(torch.arange(0, N, self.grid_spacing, device=device))
        return torch.stack(samples, 0)


class SimpleTemporalAttn(nn.Module):

    def __init__(
        self,
        hidden_size=64,
        num_hidden_layers=4,
        intermediate_size=32,
        num_attention_heads=4,
        features_only=False,
    ):
        super().__init__()
        self.encoder = BertEncoder(
            BertConfig(
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                position_embedding_type="relative_key",
                max_position_embeddings=1024,
            )
        )
        self.fc = nn.Linear(64, 6)
        self.features_only = features_only

    def forward(self, features, features_only=True):
        hidden_state = self.encoder(features).last_hidden_state
        if self.features_only:
            return hidden_state

        outputs = self.fc(hidden_state)[:, 1:, :]
        return outputs

    def predict(self, data, device):
        return self(data["pooled_cnn_features"].to(device))


class LocalTrackingEstimationHead(nn.Module):

    def __init__(self, in_features, apply_init_tracking_head=False):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, 6)
        if apply_init_tracking_head:
            init_tracking_head(self.fc)

    def forward(self, inputs):
        return self.fc(inputs)[:, 1:, :]



@dataclass
class BuildGlobalAndLocalContextFusionConfig:
    """Configure the model"""

    _registry = {}

    def __init_subclass__(cls, *args, name=None, **kwargs) -> None:
        cls._registry[name] = cls
        return super().__init_subclass__(*args, **kwargs)

    add_pos_embedding_before_decoder: bool = True
    decoder_hidden_size: int = 512
    upsample_global_context: str | None = None
    pos_emb_type: str = "relative_key"
    force_no_absolute_pos_emb_in_cross_attn: bool = False
    freeze_context_image_encoder: bool = False
    decoder_dropout: float = 0.1
    local_features_image_encoder: str = "lyric_dragon_feature_extractor_no_temporal"
    local_features_image_encoder_checkpoint: Optional[str] = None
    local_features_image_encoder_kwargs: Optional[dict] = field(default_factory=dict)
    local_features_encoder: Literal[
        "polished-snow", "rand-init-small-transformer", "identity"
    ] = "polished-snow"
    load_features_encoder_weights: bool = False
    local_features_dropout_prob: float = 0.0
    context_encoder_backbone: str = "vital-butterfly"
    context_encoder_hidden_size: int = 512
    context_encoder_sample_spacing: int = 16
    disable_context_encoder: bool = False
    disable_local_encoder: bool = False
    load_model_weights: Optional[str] = None


class GlobalAndLocalContextFusion(nn.Module):

    def __init__(
        self,
        local_image_features_encoder,
        local_features_encoder,
        sparse_context_encoder,
        tracking_decoder,
        head,
        decoder_hidden_size=512,
        upsample_global_context: str | None = None,
        add_pos_embedding_before_decoder: bool = True,
        disable_local_encoder=False, 
        sampler=SparseSampler(),
    ):
        super().__init__()
        self.upsample_global_context = upsample_global_context
        self.context_encoder = sparse_context_encoder
        self.features_encoder = local_features_encoder
        self.local_image_features_encoder = local_image_features_encoder
        self.sparse_sampler = sampler
        self.disable_local_encoder = disable_local_encoder
        self.positional_embedding = (
            torch.nn.Embedding(1024, decoder_hidden_size)
            if add_pos_embedding_before_decoder
            else None
        )
        self.tracking_decoder = tracking_decoder
        self.head = head

    def forward(
        self,
        images=None,
        context_image_features=None,
        features=None,
        images_for_features=None,
        position_indices=None,
        return_decoder_outputs=False,
    ):
        """
        Args:
            images - the context images (typically 224x224 downsampled)
            context_image_features - if not providing images, provide the context features directly.
            features - the local encoder outputs
            images_for_features - if not proving local encoder outputs, provide the images to create them.
        """

        if features is None:
            assert images_for_features is not None
            features = self.local_image_features_encoder(images_for_features)

        seq_length = features.shape[1]
        local_features = self.features_encoder(features)
        sparse_indices = self.sparse_sampler(images, local_features)

        context_features = self.context_encoder(
            context_images=images,
            sample_indices=sparse_indices,
            context_image_features=context_image_features,
        )

        if self.upsample_global_context:
            context_features = einops.rearrange(context_features, "b n c -> b c n")
            context_features = torch.nn.functional.interpolate(
                context_features, (seq_length,), mode=self.upsample_global_context
            )
            context_features = einops.rearrange(context_features, "b c n -> b n c")

        # need positional embedding for local and context features to make cross attention work,
        # if using absolute positional embedding in cross attn.
        if self.positional_embedding is not None:
            seq_length = features.shape[1]
            position_indices = (
                position_indices
                if position_indices is not None
                else torch.arange(seq_length, device=features.device)[
                    None, :
                ].repeat_interleave(features.shape[0], 0)
            )
            embeddings = self.positional_embedding(position_indices)

            if not self.upsample_global_context:
                context_features += torch.gather(
                    embeddings,
                    1,
                    sparse_indices[..., None].repeat_interleave(
                        embeddings.shape[-1], -1
                    ),
                )
            else:
                context_features += embeddings
            local_features += embeddings

        if self.disable_local_encoder: 
            tracking_decoder_inputs = dict(
                hidden_states=context_features
            )
        else: 
            tracking_decoder_inputs = dict(
                hidden_states=local_features, encoder_hidden_states=context_features
            )
        if self.upsample_global_context is None:
            # since the length of encoder hidden states is not the same as
            # hidden states, we need to explicitly pass the position ids
            # for relative attention
            position_ids = torch.arange(
                local_features.shape[1], device=local_features.device
            )[None, ...].repeat_interleave(local_features.shape[0], 0)
            encoder_position_ids = sparse_indices
            tracking_decoder_inputs = dict(
                position_ids=position_ids,
                encoder_position_ids=encoder_position_ids,
                **tracking_decoder_inputs,
            )

        if return_decoder_outputs:
            return self.tracking_decoder.module(
                **tracking_decoder_inputs,
                output_attentions=True,
            )

        tracking_decoder_outputs = self.tracking_decoder(**tracking_decoder_inputs)
        return self.head(tracking_decoder_outputs)

    def predict(self, data, device, **kwargs):
        images = data["images"].to(device)
        features = (
            data["pooled_cnn_features"].to(device)
            if "pooled_cnn_features" in data
            else None
        )
        images_for_features = (
            data["images_for_features"].to(device)
            if "images_for_features" in data
            else None
        )
        return self(
            images, features=features, images_for_features=images_for_features, **kwargs
        )

    @classmethod
    def from_config(
        cls, cfg: BuildGlobalAndLocalContextFusionConfig, num_local_features=64
    ):
        # === Context encoder ===
        if not cfg.disable_context_encoder:
            context_encoder = cls._get_context_backbone(cfg)
            if cfg.context_encoder_hidden_size != cfg.decoder_hidden_size:

                class ContextEncoderProj(nn.Sequential):
                    def forward(self, *args, **kwargs):
                        return self[1](self[0](*args, **kwargs))

                context_encoder = ContextEncoderProj(
                    context_encoder,
                    nn.Linear(cfg.context_encoder_hidden_size, cfg.decoder_hidden_size),
                )

        else:
            context_encoder = None

        # === local features image encoder ===
        local_features_image_encoder = cls._get_local_image_features_backbone(cfg)

        # === Local encoder ===
        local_features_encoder_backbone_name = cfg.local_features_encoder
        if local_features_encoder_backbone_name == "polished-snow":
            features_encoder_in_features = 64
            features_encoder_out_features = 64
            features_encoder_backbone = SimpleTemporalAttn()
            print(
                load_model_weights(
                    features_encoder_backbone,
                    "experiments/good_runs/polished-snow/checkpoint/best.pt",
                )
            )
            features_encoder_backbone.features_only = True
        elif local_features_encoder_backbone_name == "rand-init-small-transformer":
            features_encoder_in_features = 64
            features_encoder_out_features = 64
            features_encoder_backbone = SimpleTemporalAttn(features_only=True)
        elif local_features_encoder_backbone_name == "identity":
            features_encoder_in_features = num_local_features
            features_encoder_out_features = num_local_features
            logging.info(f"Setting local features encoder to identity.")
            features_encoder_backbone = nn.Identity()
        else:
            raise NotImplementedError(local_features_encoder_backbone_name)
        features_encoder = []
        if cfg.local_features_dropout_prob != 0:
            features_encoder.append(nn.Dropout(cfg.local_features_dropout_prob))
        if features_encoder_in_features != num_local_features:
            features_encoder.append(
                nn.Linear(num_local_features, features_encoder_in_features, bias=False)
            )
        features_encoder.append(features_encoder_backbone)
        if features_encoder_out_features != cfg.decoder_hidden_size:
            features_encoder.append(
                nn.Linear(
                    features_encoder_out_features, cfg.decoder_hidden_size, bias=False
                )
            )
        features_encoder = nn.Sequential(*features_encoder)

        if cfg.force_no_absolute_pos_emb_in_cross_attn:
            warnings.warn(
                f"Deprecated use of cfg.force_no_absolute_pos_emb_in_cross_attn"
            )
            # _force_use_relative_position_embedding_in_bert()

        # === Decoder and head ===
        decoder = BertWrapper(
            BertEncoder(
                BertConfig(
                    hidden_size=cfg.decoder_hidden_size,
                    intermediate_size=cfg.decoder_hidden_size * 2,
                    num_attention_heads=8,
                    position_embedding_type=cfg.pos_emb_type,
                    max_position_embeddings=1024,
                    is_decoder=not cfg.disable_context_encoder,
                    add_cross_attention=not cfg.disable_context_encoder,
                    attention_probs_dropout_prob=cfg.decoder_dropout,
                    hidden_dropout_prob=cfg.decoder_dropout,
                )
            )
        )
        head = LocalTrackingEstimationHead(cfg.decoder_hidden_size)

        model = cls(
            local_features_image_encoder,
            features_encoder,
            context_encoder,
            decoder,
            head,
            decoder_hidden_size=cfg.decoder_hidden_size,
            upsample_global_context =cfg.upsample_global_context,
            add_pos_embedding_before_decoder=cfg.add_pos_embedding_before_decoder,
            sampler=RegularGridSampler(cfg.context_encoder_sample_spacing),
        )
        if cfg.load_model_weights:
            load_model_weights(model, cfg.load_model_weights)

        return model

    @classmethod
    def _get_context_backbone(cls, cfg):
        from src.models.model_registry import simple_sparse_tracking_estimator_v2

        if cfg.context_encoder_backbone == "vital-butterfly":
            context_encoder_backbone = get_model(
                "simple_sparse_tracking_estimator_v2",
                image_size=224,
                backbone="image_resnet_avgpool",
                hidden_size=512,
                checkpoint=os.environ.get("VITAL_BUTTERFLY_CHECKPOINT", "experiments/good_runs/vital-butterfly/checkpoint/best-hacked_for_other_model.pt"),
            )
            context_encoder_backbone.features_only = True
        elif cfg.context_encoder_backbone == 'rand-init-cnn': 
            context_encoder_backbone = get_model(
                "simple_sparse_tracking_estimator_v2",
                image_size=224,
                backbone="image_resnet_avgpool",
                hidden_size=512,
            )
            context_encoder_backbone.features_only = True
        elif os.path.exists(cfg.context_encoder_backbone):
            try:
                # this is assumed to be loading a model directly produced by a script from its config and weights
                from scripts.train.predict_transform_distant_frames_v2 import (
                    get_model_from_args,
                )
                from torch.nn.modules.utils import (
                    consume_prefix_in_state_dict_if_present,
                )

                path = cfg.context_encoder_backbone
                logging.info(f"Loading context backbone from {path}")
                context_encoder_backbone = get_model_from_args(
                    OmegaConf.load(os.path.join(path, "config.yaml"))
                )
                weights = torch.load(
                    os.path.join(path, "checkpoint", "best.pt"), map_location="cpu"
                )["model"]
                consume_prefix_in_state_dict_if_present(weights, "_orig_mod.")
                context_encoder_backbone.load_state_dict(weights)
                context_encoder_backbone.features_only = True
            except:
                from src.models.sparse_context_models import ModelBuilder
                from torch.nn.modules.utils import (
                    consume_prefix_in_state_dict_if_present,
                )

                path = cfg.context_encoder_backbone
                logging.info(f"Loading context backbone from {path}")
                config = OmegaConf.load(os.path.join(path, "config.yaml"))
                model_builder = ModelBuilder(**config.model_builder)
                context_encoder_backbone = model_builder.get_model()

                weights = torch.load(
                    os.path.join(path, "checkpoint", "best.pt"), map_location="cpu"
                )["model"]
                consume_prefix_in_state_dict_if_present(weights, "_orig_mod.")
                context_encoder_backbone.load_state_dict(weights)
                context_encoder_backbone.features_only = True
        else:
            raise ValueError()

        if cfg.freeze_context_image_encoder:
            context_encoder_backbone.backbone = FrozenModuleWrapper(
                context_encoder_backbone.backbone
            )
        return context_encoder_backbone

    @classmethod
    def _get_local_image_features_backbone(cls, cfg):
        return get_model(
            cfg.local_features_image_encoder,
            backbone_checkpoint=cfg.local_features_image_encoder_checkpoint,
            **cfg.local_features_image_encoder_kwargs,
        )


def _force_use_relative_position_embedding_in_bert():
    warnings.warn("Deprecated function")
    return

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            print(
                f"Monkey patched bert init - using pos embedding type {config.position_embedding_type}"
            )
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    BertLayer.__init__ = __init__


def _get_context_backbone(
    context_encoder_backbone: str = "vital-butterfly",
    freeze_context_image_encoder: bool = False,
):
    from src.models.model_registry import simple_sparse_tracking_estimator_v2

    if context_encoder_backbone == "vital-butterfly":
        context_encoder_backbone_mod = get_model(
            "simple_sparse_tracking_estimator_v2",
            image_size=224,
            backbone="image_resnet_avgpool",
            hidden_size=512,
            checkpoint=os.environ.get("VITAL_BUTTERFLY_CHECKPOINT", "experiments/good_runs/vital-butterfly/checkpoint/best-hacked_for_other_model.pt"),
        )
        context_encoder_backbone_mod.features_only = True
    elif context_encoder_backbone.endswith(".yaml"):
        name, args, kwargs = OmegaConf.load(context_encoder_backbone)
        context_encoder_backbone_mod = get_model(name, *args, **kwargs)
        context_encoder_backbone_mod.features_only = True
    elif os.path.exists(context_encoder_backbone):
        try:
            # this is assumed to be loading a model directly produced by a script from its config and weights
            from scripts.train.predict_transform_distant_frames_v2 import (
                get_model_from_args,
            )
            from torch.nn.modules.utils import (
                consume_prefix_in_state_dict_if_present,
            )

            path = context_encoder_backbone
            logging.info(f"Loading context backbone from {path}")
            context_encoder_backbone_mod = get_model_from_args(
                OmegaConf.load(os.path.join(path, "config.yaml"))
            )
            weights = torch.load(
                os.path.join(path, "checkpoint", "best.pt"), map_location="cpu"
            )["model"]
            consume_prefix_in_state_dict_if_present(weights, "_orig_mod.")
            context_encoder_backbone_mod.load_state_dict(weights)
            context_encoder_backbone_mod.features_only = True
        except:
            from src.models.sparse_context_models import ModelBuilder
            from torch.nn.modules.utils import (
                consume_prefix_in_state_dict_if_present,
            )

            path = context_encoder_backbone
            logging.info(f"Loading context backbone from {path}")
            config = OmegaConf.load(os.path.join(path, "config.yaml"))
            model_builder = ModelBuilder(**config.model_builder)
            context_encoder_backbone_mod = model_builder.get_model()

            weights = torch.load(
                os.path.join(path, "checkpoint", "best.pt"), map_location="cpu"
            )["model"]
            consume_prefix_in_state_dict_if_present(weights, "_orig_mod.")
            context_encoder_backbone_mod.load_state_dict(weights)
            context_encoder_backbone_mod.features_only = True
    else:
        raise ValueError()

    if freeze_context_image_encoder:
        context_encoder_backbone_mod.backbone = FrozenModuleWrapper(
            context_encoder_backbone_mod.backbone
        )
    return context_encoder_backbone_mod


def _get_local_image_features_backbone(
    local_features_image_encoder,
    local_features_image_encoder_checkpoint,
    **local_features_image_encoder_kwargs,
):
    return get_model(
        local_features_image_encoder,
        backbone_checkpoint=local_features_image_encoder_checkpoint,
        **local_features_image_encoder_kwargs,
    )


@register_model
def global_and_local_context_fusion(
    pos_emb_type: str = "relative_key",
    force_no_absolute_pos_emb_in_cross_attn: bool = False,
    freeze_context_image_encoder: bool = False,
    decoder_dropout: float = 0.1,
    local_features_image_encoder: str = "lyric_dragon_feature_extractor_no_temporal",
    local_features_image_encoder_checkpoint: Optional[str] = None,
    local_features_image_encoder_kwargs: dict[str, Any] = dict(),
    local_features_encoder: Literal[
        "polished-snow", "rand-init-small-transformer", "identity"
    ] = "polished-snow",
    local_features_dropout_prob: float = 0.0,
    context_encoder_backbone: str = "vital-butterfly",
    context_encoder_hidden_size: int = 512,
    disable_context_encoder: bool = False,
    # load_model_weights: Optional[str] = None,
    add_pos_embedding_before_decoder: bool = True,
    decoder_hidden_size: int = 512,
    upsample_global_context: str | None = None,
    num_local_features=64,
    init_tracking_head=False,
    sparse_sampler='grid', 
    average_sample_spacing=16,
):
    # === Context encoder ===
    if not disable_context_encoder:
        context_encoder = _get_context_backbone(
            context_encoder_backbone=context_encoder_backbone,
            freeze_context_image_encoder=freeze_context_image_encoder,
        )
        if context_encoder_hidden_size != decoder_hidden_size:

            class ContextEncoderProj(nn.Sequential):
                def forward(self, *args, **kwargs):
                    return self[1](self[0](*args, **kwargs))

            context_encoder = ContextEncoderProj(
                context_encoder,
                nn.Linear(context_encoder_hidden_size, decoder_hidden_size),
            )

    else:
        context_encoder = None

    # === local features image encoder ===
    local_features_image_encoder_mod = _get_local_image_features_backbone(
        local_features_image_encoder,
        local_features_image_encoder_checkpoint,
        **local_features_image_encoder_kwargs,
    )

    # === Local encoder ===
    local_features_encoder_backbone_name = local_features_encoder
    if local_features_encoder_backbone_name == "polished-snow":
        features_encoder_in_features = 64
        features_encoder_out_features = 64
        features_encoder_backbone = SimpleTemporalAttn()
        print(
                            load_model_weights(
                    features_encoder_backbone,
                    "experiments/good_runs/polished-snow/checkpoint/best.pt",
                )
        )
        features_encoder_backbone.features_only = True
    elif local_features_encoder_backbone_name == "rand-init-small-transformer":
        features_encoder_in_features = 64
        features_encoder_out_features = 64
        features_encoder_backbone = SimpleTemporalAttn(features_only=True)
    elif local_features_encoder_backbone_name == "identity":
        features_encoder_in_features = num_local_features
        features_encoder_out_features = num_local_features
        logging.info(f"Setting local features encoder to identity.")
        features_encoder_backbone = nn.Identity()
    else:
        raise NotImplementedError(local_features_encoder_backbone_name)
    features_encoder = []
    if local_features_dropout_prob != 0:
        features_encoder.append(nn.Dropout(local_features_dropout_prob))
    if features_encoder_in_features != num_local_features:
        features_encoder.append(
            nn.Linear(num_local_features, features_encoder_in_features, bias=False)
        )
    features_encoder.append(features_encoder_backbone)
    if features_encoder_out_features != decoder_hidden_size:
        features_encoder.append(
            nn.Linear(features_encoder_out_features, decoder_hidden_size, bias=False)
        )
    features_encoder = nn.Sequential(*features_encoder)

    if force_no_absolute_pos_emb_in_cross_attn:
        _force_use_relative_position_embedding_in_bert()

    # === Decoder and head ===
    decoder = BertWrapper(
        BertEncoder(
            BertConfig(
                hidden_size=decoder_hidden_size,
                intermediate_size=decoder_hidden_size * 2,
                num_attention_heads=8,
                position_embedding_type=pos_emb_type,
                max_position_embeddings=1024,
                is_decoder=not disable_context_encoder,
                add_cross_attention=not disable_context_encoder,
                attention_probs_dropout_prob=decoder_dropout,
                hidden_dropout_prob=decoder_dropout,
            )
        )
    )
    head = LocalTrackingEstimationHead(
        decoder_hidden_size, apply_init_tracking_head=init_tracking_head
    )

    if sparse_sampler == 'grid': 
        sampler = RegularGridSampler(average_sample_spacing)
    else: 
        sampler = RandomSparseSampleTemporal(n_samples=512 / average_sample_spacing)


    model = GlobalAndLocalContextFusion(
        local_features_image_encoder_mod,
        features_encoder,
        context_encoder,
        decoder,
        head,
        upsample_global_context=upsample_global_context,
        add_pos_embedding_before_decoder=add_pos_embedding_before_decoder,
        sampler=sampler,
    )

    # if load_model_weights:
    #     load_model_weights(model, load_model_weights)

    return model
