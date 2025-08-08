from src.models import model_registry
from src.models.fusion_model.sampler import RegularGridSampler, SparseSampler
from src.models.utils import BertWrapper, FrozenModuleWrapper
from src.models.spatio_temporal_attn import SimpleTemporalAttn
from src.utils.utils import load_model_weights


from torch import nn
import torch 
from transformers.models.bert import BertConfig
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder


import logging


class ModelWithPretrainedContextEncoderV0Control(nn.Module):
    def __init__(
        self,
        tracking_decoder,
        features_encoder,
        head,
        sampler=SparseSampler(),
    ):
        super().__init__()
        self.features_encoder = features_encoder
        self.sparse_sampler = sampler
        self.tracking_decoder = tracking_decoder
        self.head = head

    def forward(self, features):
        local_features = self.features_encoder(features)
        tracking_decoder_outputs = self.tracking_decoder(
            local_features
        )
        return self.head(tracking_decoder_outputs)

    def predict(self, data, device):
        return self(data["pooled_cnn_features"].to(device))


class ModelWithPretrainedContextEncoderV0(nn.Module):
    def __init__(
        self,
        sparse_context_encoder,
        tracking_decoder,
        features_encoder,
        head,
        sampler=SparseSampler(),
    ):
        super().__init__()
        self.context_encoder = sparse_context_encoder
        self.features_encoder = features_encoder
        self.sparse_sampler = sampler
        self.tracking_decoder = tracking_decoder
        self.head = head

    def forward(self, images, features):
        local_features = self.features_encoder(features)
        sparse_indices = self.sparse_sampler(images, local_features)
        context_features = self.context_encoder(images, sparse_indices)
        tracking_decoder_outputs = self.tracking_decoder(
            local_features, encoder_hidden_states=context_features
        )
        return self.head(tracking_decoder_outputs)

    def predict(self, data, device):
        return self(data["images"].to(device), data["pooled_cnn_features"].to(device))


class LocalTrackingEstimationHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, 6)

    def forward(self, inputs):
        return self.fc(inputs)[:, 1:, :]


@model_registry.register_model
def pretrained_global_local_context_v1(cfg):

    # === Global encoder ===
    if not cfg.disable_global_encoder:
        context_encoder_backbone = model_registry.get_model(
            "simple_sparse_tracking_estimator_v2",
            image_size=224,
            backbone="image_resnet_avgpool",
            hidden_size=512,
            checkpoint="experiments/good_runs/vital-butterfly/checkpoint/best-hacked_for_other_model.pt",
        )
        context_encoder_backbone.features_only = True
        if cfg.freeze_global_image_encoder:
            context_encoder_backbone.backbone = FrozenModuleWrapper(context_encoder_backbone.backbone)
        context_encoder = context_encoder_backbone
    else:
        context_encoder = None

    # === Local encoder === 
    local_features_encoder_backbone_name = cfg.local_features_encoder
    if local_features_encoder_backbone_name == 'polished-snow':
        features_encoder_backbone = SimpleTemporalAttn()
        if cfg.model_kwargs.get('load_features_encoder_weights', True):
            print(
                load_model_weights(
                    features_encoder_backbone,
                    "experiments/good_runs/polished-snow/checkpoint/best.pt",
                )
            )
        features_encoder_backbone.features_only = True
    elif local_features_encoder_backbone_name == "rand-init-small-transformer":
        features_encoder_backbone = SimpleTemporalAttn()
    elif local_features_encoder_backbone_name == 'identity':
        logging.info(f"Setting local features encoder to identity.")
        features_encoder_backbone = nn.Identity()
    else:
        raise NotImplementedError(local_features_encoder_backbone_name)
    features_encoder = nn.Sequential(
        features_encoder_backbone, nn.Linear(64, cfg.decoder_hidden_size, bias=False)
    )

    # === Decoder and head === 
    head = LocalTrackingEstimationHead(512)
    decoder = BertWrapper(
        BertEncoder(
            BertConfig(
                hidden_size=cfg.decoder_hidden_size,
                intermediate_size=cfg.decoder_hidden_size * 2,
                num_attention_heads=8,
                position_embedding_type="relative_key",
                max_position_embeddings=1024,
                is_decoder=not cfg.disable_global_encoder,
                add_cross_attention=not cfg.disable_global_encoder,
            )
        )
    )

    if cfg.disable_global_encoder:
        return ModelWithPretrainedContextEncoderV0Control(
            decoder, features_encoder, head, RegularGridSampler(cfg.model_kwargs.get("grid_spacing", 16))
        )

    return ModelWithPretrainedContextEncoderV0(
        context_encoder,
        decoder,
        features_encoder,
        head,
        sampler=RegularGridSampler(cfg.model_kwargs.get("grid_spacing", 16)),
    )


class ModelWithPretrainedContextEncoderV0(nn.Module):
    def __init__(
        self,
        sparse_context_encoder,
        tracking_decoder,
        features_encoder,
        head,
        sampler=SparseSampler(),
    ):
        super().__init__()
        self.context_encoder = sparse_context_encoder
        self.features_encoder = features_encoder
        self.sparse_sampler = sampler
        self.tracking_decoder = tracking_decoder
        self.head = head

    def forward(self, images, features):
        local_features = self.features_encoder(features)
        sparse_indices = self.sparse_sampler(images, local_features)
        context_features = self.context_encoder(images, sparse_indices)
        tracking_decoder_outputs = self.tracking_decoder(
            local_features, encoder_hidden_states=context_features
        )
        return self.head(tracking_decoder_outputs)

    def predict(self, data, device):
        return self(data["images"].to(device), data["pooled_cnn_features"].to(device))

