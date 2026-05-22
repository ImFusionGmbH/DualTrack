import warnings
import torch
from torch import nn
import einops

from transformers.models.vit import ViTConfig, ViTModel
from transformers.models.bert.modeling_bert import BertConfig, BertAttention, BertLayer
from transformers.models.bert import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.vit import ViTConfig, ViTModel
from transformers.models.vit.modeling_vit import (
    ViTEmbeddings,
    ViTAttention,
)

from transformers.models.roformer.modeling_roformer import (
    RoFormerConfig,
    RoFormerEncoder,
)

import torch
from torch import nn
import einops

from src.models.model_registry import register_model



_transformer_dict = {
    'bert': (BertConfig, BertEncoder), 
    'roformer': (RoFormerConfig, RoFormerEncoder),
}


def get_transformer(version, **kwargs):
    config_class, encoder_class = _transformer_dict[version]
    config = config_class(**kwargs)
    return encoder_class(config)


class SplitSpatioTemporalAttentionEncoder(nn.Module):
    def __init__(
        self,
        speckle_feat_dim=512,
        context_feat_dim=512,
        speckle_fmap_size=16,
        context_fmap_size=16,
        patch_size=2,
        hidden_size=64,
        intermediate_size=128,
        num_attn_heads=8,
        num_layers=8,
        max_position_embeddings=512,
        disable_temporal_attn=False,
        disable_context=False,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.disable_context = disable_context

        _cfg = ViTConfig(
            hidden_size=hidden_size,
            num_channels=speckle_feat_dim,
            image_size=speckle_fmap_size,
            patch_size=patch_size,
            layer_norm_eps=1e-5,
        )
        self.speckle_patch_emb = ViTEmbeddings(_cfg)

        if not disable_context:
            _cfg = ViTConfig(
                hidden_size=hidden_size,
                num_channels=context_feat_dim,
                image_size=context_fmap_size,
                patch_size=patch_size,
                layer_norm_eps=1e-5,
            )
            self.context_patch_emb = ViTEmbeddings(_cfg)
        else:
            self.context_patch_emb = None

        # spatial attention layers for speckle and context
        _cfg = ViTConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attn_heads,
            layer_norm_eps=1e-5,
        )
        self.speckle_attn_layers = nn.ModuleList(
            [ViTAttention(_cfg) for _ in range(num_layers)]
        )

        if not disable_context:
            self.context_attn_layers = nn.ModuleList(
                [ViTAttention(_cfg) for _ in range(num_layers)]
            )
        else:
            self.context_patch_emb = None

        # temporal attention and cross attention layers
        _cfg = BertConfig(
            attn_implementation="eager",
            hidden_size=hidden_size,
            num_attention_heads=num_attn_heads,
            intermediate_size=intermediate_size,
            position_embedding_type="relative-key",
            add_cross_attention=True,
            is_decoder=True,
            max_position_embeddings=max_position_embeddings,
            layer_norm_eps=1e-5,
        )
        if not disable_temporal_attn:
            self.temporal_attn_layers = nn.ModuleList(
                [BertLayer(_cfg) for _ in range(num_layers)]
            )
        else:
            self.temporal_attn_layers = None

        # classifier layer
        self.fc = nn.Linear(hidden_size, 6)

    def forward(self, speckle_feats, context_feats=None):

        B, N, *_ = speckle_feats.shape

        speckle_feats = self._patch_embedding(
            speckle_feats, self.speckle_patch_emb
        )  # B, N, (H W), D

        if not self.disable_context:
            context_feats = self._patch_embedding(context_feats, self.context_patch_emb)

        for i in range(self.num_layers):
            speckle_feats = self.apply_spatial_attn(
                speckle_feats, self.speckle_attn_layers[i]
            )
            if not self.disable_context:
                context_feats = self.apply_spatial_attn(
                    context_feats, self.context_attn_layers[i]
                )
            if self.temporal_attn_layers is not None:
                speckle_feats = self.apply_temporal_attn(
                    speckle_feats, context_feats, self.temporal_attn_layers[i]
                )

        return speckle_feats, context_feats

    def _patch_embedding(self, feats, embedder):
        B, N, *_ = feats.shape
        feats = einops.rearrange(feats, "b n ... -> (b n) ...")
        feats = embedder(feats)
        return einops.rearrange(feats, "(b n) ... -> b n ... ", b=B, n=N)

    def apply_spatial_attn(self, feats, layer):
        B, N, L, D = feats.shape

        # apply the attention independently to each element of the temporal sequence
        # by folding the sequence dimention into the batch dimension
        feats = einops.rearrange(feats, "b n l d -> (b n) l d")
        outputs = layer(feats)[0]
        return einops.rearrange(outputs, "(b n) ... -> b n ...", b=B, n=N)

    def apply_temporal_attn(self, feats, cross_attn_source_feats, layer):
        B, N, L, D = feats.shape
        cls_tokens = feats[:, :, 0, :]  # B, N, D
        other_tokens = feats[:, :, 1:, :]
        cross_attn_source = cross_attn_source_feats[:, :, 0, :]

        # apply the attention only to the cls tokens
        cls_tokens = layer(cls_tokens, encoder_hidden_states=cross_attn_source)[
            0
        ].unsqueeze(2)
        return torch.cat([cls_tokens, other_tokens], dim=2)


class SplitSpatioTemporalAttentionModelForTrackingEstimation(nn.Module):
    def __init__(
        self,
        speckle_encoder,
        context_encoder,
        speckle_feat_dim=512,
        context_feat_dim=512,
        speckle_fmap_size=16,
        context_fmap_size=16,
        patch_size=2,
        hidden_size=64,
        intermediate_size=128,
        num_attn_heads=8,
        num_layers=8,
        max_position_embeddings=512,
        disable_temporal_attn=False,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.speckle_encoder = speckle_encoder
        self.context_encoder = context_encoder

        _cfg = ViTConfig(
            hidden_size=hidden_size,
            num_channels=speckle_feat_dim,
            image_size=speckle_fmap_size,
            patch_size=patch_size,
            layer_norm_eps=1e-5,
        )
        self.speckle_patch_emb = ViTEmbeddings(_cfg)
        _cfg = ViTConfig(
            hidden_size=hidden_size,
            num_channels=context_feat_dim,
            image_size=context_fmap_size,
            patch_size=patch_size,
            layer_norm_eps=1e-5,
        )
        self.context_patch_emb = ViTEmbeddings(_cfg)

        # spatial attention layers for speckle and context
        _cfg = ViTConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attn_heads,
            layer_norm_eps=1e-5,
        )
        self.speckle_attn_layers = nn.ModuleList(
            [ViTAttention(_cfg) for _ in range(num_layers)]
        )
        self.context_attn_layers = nn.ModuleList(
            [ViTAttention(_cfg) for _ in range(num_layers)]
        )

        # temporal attention and cross attention layers
        _cfg = BertConfig(
            attn_implementation="eager",
            hidden_size=hidden_size,
            num_attention_heads=num_attn_heads,
            intermediate_size=intermediate_size,
            position_embedding_type="relative-key",
            add_cross_attention=True,
            is_decoder=True,
            max_position_embeddings=max_position_embeddings,
            layer_norm_eps=1e-5,
        )
        if not disable_temporal_attn:
            self.temporal_attn_layers = nn.ModuleList(
                [BertLayer(_cfg) for _ in range(num_layers)]
            )
        else:
            self.temporal_attn_layers = None

        # classifier layer
        self.fc = nn.Linear(hidden_size, 6)

    def forward(self, x):
        speckle_feats = self.speckle_encoder(x)
        context_feats = self.context_encoder(x)

        B, N, *_ = speckle_feats.shape

        speckle_feats = self._patch_embedding(
            speckle_feats, self.speckle_patch_emb
        )  # B, N, (H W), D
        context_feats = self._patch_embedding(context_feats, self.context_patch_emb)

        for i in range(self.num_layers):
            speckle_feats = self.apply_spatial_attn(
                speckle_feats, self.speckle_attn_layers[i]
            )
            context_feats = self.apply_spatial_attn(
                context_feats, self.context_attn_layers[i]
            )
            if self.temporal_attn_layers is not None:
                speckle_feats = self.apply_temporal_attn(
                    speckle_feats, context_feats, self.temporal_attn_layers[i]
                )

        pre_classifier_features = speckle_feats[:, 1:, 0, :]
        estimated_tracking = self.fc(pre_classifier_features)

        return estimated_tracking

    def _patch_embedding(self, feats, embedder):
        B, N, *_ = feats.shape
        feats = einops.rearrange(feats, "b n ... -> (b n) ...")
        feats = embedder(feats)
        return einops.rearrange(feats, "(b n) ... -> b n ... ", b=B, n=N)

    def apply_spatial_attn(self, feats, layer):
        B, N, L, D = feats.shape

        # apply the attention independently to each element of the temporal sequence
        # by folding the sequence dimention into the batch dimension
        feats = einops.rearrange(feats, "b n l d -> (b n) l d")
        outputs = layer(feats)[0]
        return einops.rearrange(outputs, "(b n) ... -> b n ...", b=B, n=N)

    def apply_temporal_attn(self, feats, cross_attn_source_feats, layer):
        B, N, L, D = feats.shape
        cls_tokens = feats[:, :, 0, :]  # B, N, D
        other_tokens = feats[:, :, 1:, :]
        cross_attn_source = cross_attn_source_feats[:, :, 0, :]

        # apply the attention only to the cls tokens
        cls_tokens = layer(cls_tokens, encoder_hidden_states=cross_attn_source)[
            0
        ].unsqueeze(2)
        return torch.cat([cls_tokens, other_tokens], dim=2)


def video_rn18_backbone(weights_path=None, freeze_backbone=True):
    from .model_registry import video_rn18_plus_attention

    model = video_rn18_plus_attention(
        weights_path=weights_path, freeze_backbone=freeze_backbone
    )


class FeatureExtractorWithSpatialSelfAttention(nn.Module):
    def __init__(self, backbone, feature_map_size=16, num_features=512):
        super().__init__()
        self.backbone = backbone
        self.feature_map_size = feature_map_size
        self.features_proj = nn.Conv2d(num_features, 64, (3, 3), (2, 2), (1, 1)).cuda()

        # use components from huggingface to create the self-attention layers
        from transformers.models.vit import ViTConfig, ViTModel
        from transformers.models.vit.modeling_vit import (
            ViTEmbeddings,
            ViTPatchEmbeddings,
        )

        image_size = 256  # arbitrarily set
        patch_size = int(
            image_size / (feature_map_size / 2)
        )  # needed to make the dummy patch embedding the right shape

        vit_config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=64,
            num_hidden_layers=4,
            intermediate_size=32,
            num_attention_heads=4,
        )

        self.vit = ViTModel(
            vit_config,
            add_pooling_layer=False,
        )

        class _ReplacePatchEmbedding(ViTPatchEmbeddings):
            def forward(self, x, **_):
                return x.flatten(2).transpose(1, 2)

        self.vit.embeddings.patch_embeddings = _ReplacePatchEmbedding(vit_config)
        self.fc = torch.nn.Linear(64, 6)

    def forward(self, x):
        B, N, C, H, W = x.shape
        features = self.backbone(x)
        B, N, C, H, W = features.shape
        features = einops.rearrange(
            features, "b n c h w -> (b n) c h w"
        )  # fold sequence dim into batch dim
        features = self.features_proj(features)
        vit_output = self.vit(features).last_hidden_state
        cls_tokens = vit_output[:, 0, :]

        cls_tokens = einops.rearrange(cls_tokens, "(b n) c -> b n c", b=B, n=N)
        outputs = self.fc(cls_tokens)
        return outputs[:, 1:, :]


class FeatureExtractorWithSpatialSelfAttentionV0(nn.Module):
    def __init__(self, backbone, feature_map_size=16, num_features=512):
        super().__init__()
        self.backbone = backbone
        self.feature_map_size = feature_map_size
        self.features_proj = nn.Conv2d(num_features, 64, (3, 3), (2, 2), (1, 1)).cuda()

        # use components from huggingface to create the self-attention layers
        from transformers.models.vit import ViTConfig, ViTModel
        from transformers.models.vit.modeling_vit import (
            ViTEmbeddings,
            ViTPatchEmbeddings,
        )

        image_size = 256  # arbitrarily set
        patch_size = int(
            image_size / (feature_map_size / 2)
        )  # needed to make the dummy patch embedding the right shape

        vit_config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=64,
            num_hidden_layers=4,
            intermediate_size=32,
            num_attention_heads=4,
        )

        self.vit = ViTModel(
            vit_config,
            add_pooling_layer=False,
        )

        class _ReplacePatchEmbedding(ViTPatchEmbeddings):
            def forward(self, x, **_):
                return x.flatten(2).transpose(1, 2)

        self.vit.embeddings.patch_embeddings = _ReplacePatchEmbedding(vit_config)
        self.fc = torch.nn.Linear(64, 6)

    def forward(self, x):
        B, N, C, H, W = x.shape
        features = self.backbone(x)
        B, N, C, H, W = features.shape
        features = einops.rearrange(
            features, "b n c h w -> (b n) c h w"
        )  # fold sequence dim into batch dim
        features = self.features_proj(features)
        vit_output = self.vit(features).last_hidden_state
        cls_tokens = vit_output[:, 0, :]

        cls_tokens = einops.rearrange(cls_tokens, "(b n) c -> b n c", b=B, n=N)
        outputs = self.fc(cls_tokens)
        return outputs[:, 1:, :]


class FeatureExtractorWithSPTAttention(nn.Module):
    def __init__(
        self,
        backbone,
        feature_map_size=16,
        num_features=512,
        patch_size=2,
        num_classes=6,
        position_embedding_type="relative_key",
        temporal_attn_type="bert",
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_map_size = feature_map_size
        self.patch_size = patch_size

        # use components from huggingface to create the self-attention layers
        vit_config = ViTConfig(
            image_size=feature_map_size,
            patch_size=patch_size,
            num_channels=num_features,
            hidden_size=64,
            num_hidden_layers=4,
            intermediate_size=32,
            num_attention_heads=4,
        )
        self.vit = ViTModel(
            vit_config,
            add_pooling_layer=False,
        )

        if temporal_attn_type == "bert":
            config_cls = BertConfig
            model_cls = BertEncoder
        elif temporal_attn_type == "roformer":
            config_cls = RoFormerConfig
            model_cls = RoFormerEncoder
        else:
            raise NotImplementedError(temporal_attn_type)

        config = config_cls(
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=32,
            hidden_size=64,
            position_embedding_type=position_embedding_type,
            max_position_embeddings=1024,
        )
        self.bert = model_cls(config)
        self.fc = torch.nn.Linear(64, num_classes)

    def forward(
        self,
        images,
        *,
        vit_output=None,
        encoder_hidden_states=None,
        return_hiddens=False,
    ):
        B, N, C, H, W = images.shape

        if vit_output is None:
            features = self.backbone(images)
            B, N, C, H, W = features.shape
            features = einops.rearrange(
                features, "b n c h w -> (b n) c h w"
            )  # fold sequence dim into batch dim
            vit_output = self.vit(features).last_hidden_state
            cls_tokens = vit_output[:, 0, :]
        else:
            cls_tokens = vit_output[0]

        cls_tokens = einops.rearrange(cls_tokens, "(b n) c -> b n c", b=B, n=N)
        cls_tokens = self.bert(
            cls_tokens, encoder_hidden_states=encoder_hidden_states
        ).last_hidden_state
        outputs = self.fc(cls_tokens)
        return outputs[:, 1:, :]

    def get_intermediate_state(self, x):
        if isinstance(x, dict):
            x = x["images"]

        B, N, C, H, W = x.shape
        features = self.backbone(x)
        B, N, C, H, W = features.shape
        features = einops.rearrange(
            features, "b n c h w -> (b n) c h w"
        )  # fold sequence dim into batch dim
        vit_output = self.vit(features).last_hidden_state
        cls_tokens = vit_output[:, 0, :]

        return {"vit_output": cls_tokens}


class FeatureExtractorWithSPTAttentionAndSparseFM(nn.Module):
    def __init__(
        self,
        backbone,
        feature_map_size=16,
        num_features=512,
        patch_size=2,
        num_classes=6,
        position_embedding_type="relative_key",
        temporal_attn_type="bert",
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_map_size = feature_map_size
        self.patch_size = patch_size

        # use components from huggingface to create the self-attention layers
        vit_config = ViTConfig(
            image_size=feature_map_size,
            patch_size=patch_size,
            num_channels=num_features,
            hidden_size=64,
            num_hidden_layers=4,
            intermediate_size=32,
            num_attention_heads=4,
        )
        self.vit = ViTModel(
            vit_config,
            add_pooling_layer=False,
        )

        if temporal_attn_type == "bert":
            config_cls = BertConfig
            model_cls = BertEncoder
        elif temporal_attn_type == "roformer":
            config_cls = RoFormerConfig
            model_cls = RoFormerEncoder
        else:
            raise NotImplementedError(temporal_attn_type)

        config = config_cls(
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=32,
            hidden_size=64,
            position_embedding_type=position_embedding_type,
            max_position_embeddings=1024,
        )
        self.bert = model_cls(config)
        self.fc = torch.nn.Linear(64, num_classes)

    def forward(self, images, *, vit_output=None, encoder_hidden_states=None):
        B, N, C, H, W = images.shape

        if vit_output is None:
            features = self.backbone(images)
            B, N, C, H, W = features.shape
            features = einops.rearrange(
                features, "b n c h w -> (b n) c h w"
            )  # fold sequence dim into batch dim
            vit_output = self.vit(features).last_hidden_state
            cls_tokens = vit_output[:, 0, :]
        else:
            cls_tokens = vit_output[0]

        cls_tokens = einops.rearrange(cls_tokens, "(b n) c -> b n c", b=B, n=N)
        cls_tokens = self.bert(
            cls_tokens, encoder_hidden_states=encoder_hidden_states
        ).last_hidden_state
        outputs = self.fc(cls_tokens)
        return outputs[:, 1:, :]

    def get_intermediate_state(self, x):
        if isinstance(x, dict):
            x = x["images"]

        B, N, C, H, W = x.shape
        features = self.backbone(x)
        B, N, C, H, W = features.shape
        features = einops.rearrange(
            features, "b n c h w -> (b n) c h w"
        )  # fold sequence dim into batch dim
        vit_output = self.vit(features).last_hidden_state
        cls_tokens = vit_output[:, 0, :]

        return {"vit_output": cls_tokens}


class FeatureExtractorWithSPTAttnAndFMFeatures(nn.Module):
    def __init__(
        self,
        backbone,
        fm_backbone=None,
        feature_map_size=16,
        num_features=512,
        patch_size=2,
        num_fm_features=16,
        fm_feature_map_size=16,
        fm_patch_size=2,
        temporal_attn_type="bert",
        position_embedding_type="relative_key",
        use_temporal_encoder_for_fm: bool = False,
        fm_temporal_stride: int | None = None,
        hidden_size=64,
        num_hidden_layers=4,
        intermediate_size=32,
        num_attention_heads=4,
        bert_kw=dict(),
    ):
        super().__init__()
        self.backbone = backbone
        self.fm_backbone = fm_backbone
        self.feature_map_size = feature_map_size
        self.patch_size = patch_size
        self.fm_temporal_stride = fm_temporal_stride

        # use components from huggingface to create the self-attention layers
        from transformers.models.bert import BertConfig
        from transformers.models.bert.modeling_bert import BertEncoder
        from transformers.models.vit import ViTConfig, ViTModel

        vit_config = ViTConfig(
            image_size=feature_map_size,
            patch_size=patch_size,
            num_channels=num_features,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
        )
        self.vit = ViTModel(
            vit_config,
            add_pooling_layer=False,
        )

        fm_vit_config = ViTConfig(
            image_size=fm_feature_map_size,
            patch_size=fm_patch_size,
            num_channels=num_fm_features,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
        )
        self.fm_vit = ViTModel(fm_vit_config, add_pooling_layer=False)

        if temporal_attn_type == "bert":
            config_cls = BertConfig
            model_cls = BertEncoder
        elif temporal_attn_type == "roformer":
            config_cls = RoFormerConfig
            model_cls = RoFormerEncoder
        else:
            raise NotImplementedError(temporal_attn_type)

        _bert_kw = dict(
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
        )
        _bert_kw.update(bert_kw)

        bert_hidden_size = _bert_kw["hidden_size"]
        if bert_hidden_size != hidden_size:
            self.vit_bert_adapter = nn.Linear(hidden_size, bert_hidden_size, bias=False)
            self.vit_bert_adapter_fm = nn.Linear(
                hidden_size, bert_hidden_size, bias=False
            )
        else:
            self.vit_bert_adapter = None
            self.vit_bert_adapter_fm = None

        bert_config = config_cls(
            position_embedding_type=position_embedding_type,
            max_position_embeddings=1024,
            add_cross_attention=True,
            is_decoder=True,
            **_bert_kw,
        )
        self.bert = model_cls(bert_config)

        if use_temporal_encoder_for_fm:
            self.vit_bert = model_cls(bert_config)
        else:
            self.vit_bert = None

        self.fc = torch.nn.Linear(bert_hidden_size, 6)

    def forward(self, images, fm_features=None, images_fm=None):
        B, N, C, H, W = images.shape
        cnn_features = self.backbone(images)
        B, N, C, H, W = cnn_features.shape
        cnn_features = einops.rearrange(
            cnn_features, "b n c h w -> (b n) c h w"
        )  # fold sequence dim into batch dim
        if self.fm_backbone is not None and images_fm is not None:
            # we have to compute the foundation model features ourselves
            if self.fm_temporal_stride is not None:
                images_fm = images_fm[:, :: self.fm_temporal_stride, ...]
            N_fm = images_fm.shape[1]
            fm_features = self.fm_backbone(images_fm)
        else:
            N_fm = N

        assert fm_features is not None

        fm_features = einops.rearrange(fm_features, "b n c h w -> (b n) c h w")
        vit_output = self.vit(cnn_features).last_hidden_state
        fm_vit_output = self.fm_vit(fm_features).last_hidden_state

        cls_tokens = vit_output[:, 0, :]
        if self.vit_bert_adapter is not None:
            cls_tokens = self.vit_bert_adapter(cls_tokens)
        fm_cls_tokens = fm_vit_output[:, 0, :]
        if self.vit_bert_adapter_fm is not None:
            fm_cls_tokens = self.vit_bert_adapter_fm(fm_cls_tokens)

        cls_tokens = einops.rearrange(cls_tokens, "(b n) c -> b n c", b=B, n=N)
        fm_cls_tokens = einops.rearrange(fm_cls_tokens, "(b n) c -> b n c", b=B, n=N_fm)

        if self.vit_bert is not None:
            fm_cls_tokens = self.vit_bert(fm_cls_tokens).last_hidden_state

        cls_tokens = self.bert(
            cls_tokens, encoder_hidden_states=fm_cls_tokens
        ).last_hidden_state
        outputs = self.fc(cls_tokens)
        return outputs[:, 1:, :]

    def forward_data(self, data):
        outputs = self(data["images"], data["features"])
        data["prediction"] = outputs
        if "targets" in data:
            data["loss"] = torch.nn.functional.mse_loss(outputs, data["targets"])
        return data


class FeatureExtractorWithSpatialAttentionAndFMFeatures(nn.Module):
    def __init__(
        self,
        backbone,
        feature_map_size=16,
        num_features=512,
        patch_size=2,
        num_fm_features=16,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_map_size = feature_map_size
        self.patch_size = patch_size

        # use components from huggingface to create the self-attention layers
        from transformers.models.vit import ViTConfig, ViTModel

        vit_config = ViTConfig(
            image_size=feature_map_size,
            patch_size=patch_size,
            num_channels=num_features + num_fm_features,
            hidden_size=64,
            num_hidden_layers=4,
            intermediate_size=32,
            num_attention_heads=4,
        )
        self.vit = ViTModel(
            vit_config,
            add_pooling_layer=False,
        )
        self.fc = torch.nn.Linear(64, 6)

    def forward(self, x, fm_features):
        B, N, C, H, W = x.shape
        features = self.backbone(x)

        B, N, C, H, W = features.shape

        # fold sequence dim into batch dim
        features = einops.rearrange(features, "b n c h w -> (b n) c h w")
        fm_features = einops.rearrange(fm_features, "b n c h w -> (b n) c h w")

        if fm_features.shape[-2:] != features.shape[-2:]:
            fm_features = nn.functional.interpolate(fm_features, features.shape[-2:])

        features_cat = torch.cat([fm_features, features], dim=1)

        vit_output = self.vit(features_cat).last_hidden_state
        cls_tokens = vit_output[:, 0, :]

        cls_tokens = einops.rearrange(cls_tokens, "(b n) c -> b n c", b=B, n=N)
        outputs = self.fc(cls_tokens)
        return outputs[:, 1:, :]

    def forward_data(self, data):
        outputs = self(data["images"], data["features"])
        data["prediction"] = outputs
        if "targets" in data:
            data["loss"] = torch.nn.functional.mse_loss(outputs, data["targets"])
        return data


class FeatureExtractorWithSPTAttnAndFMFeaturesEarlyConcat(nn.Module):
    def __init__(
        self,
        backbone,
        fm_backbone=None,
        feature_map_size=16,
        num_features=512,
        patch_size=2,
        num_fm_features=16,
    ):
        super().__init__()
        self.backbone = backbone
        self.fm_backbone = fm_backbone
        self.feature_map_size = feature_map_size
        self.patch_size = patch_size

        # use components from huggingface to create the self-attention layers
        from transformers.models.bert import BertConfig
        from transformers.models.bert.modeling_bert import BertEncoder
        from transformers.models.vit import ViTConfig, ViTModel

        vit_config = ViTConfig(
            image_size=feature_map_size,
            patch_size=patch_size,
            num_channels=((num_features + num_fm_features)),
            hidden_size=64,
            num_hidden_layers=4,
            intermediate_size=32,
            num_attention_heads=4,
        )
        self.vit = ViTModel(
            vit_config,
            add_pooling_layer=False,
        )

        bert_config = BertConfig(
            attn_implementation="eager",
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=32,
            hidden_size=64,
            position_embedding_type="relative_key",
            max_position_embeddings=1024,
        )
        self.bert = BertEncoder(bert_config)

        # self.bert.forward()

        self.fc = torch.nn.Linear(64, 6)

    def forward(self, x, fm_features=None, fm_image=None):
        B, N, C, H, W = x.shape
        features = self.backbone(x)
        B, N, C, H, W = features.shape
        features = einops.rearrange(
            features, "b n c h w -> (b n) c h w"
        )  # fold sequence dim into batch dim

        if fm_features is None:
            assert self.fm_backbone is not None
            fm_features = self.fm_backbone(fm_image)
        fm_features = einops.rearrange(fm_features, "b n c h w -> (b n) c h w")

        # we simply concatenate the fm features with the other features
        if fm_features.shape[-2:] != features.shape[-2:]:
            warnings.warn(
                f"Foundation model feature map shape is different from resnet feature map shape. Resizing..."
            )
            fm_features = nn.functional.interpolate(fm_features, features.shape[-2:])

        fused_features = torch.cat([features, fm_features], dim=1)  # B (C1+C2) H W

        vit_output = self.vit(fused_features).last_hidden_state
        cls_tokens = vit_output[:, 0, :]
        cls_tokens = einops.rearrange(cls_tokens, "(b n) c -> b n c", b=B, n=N)
        cls_tokens = self.bert(cls_tokens).last_hidden_state

        outputs = self.fc(cls_tokens)
        return outputs[:, 1:, :]

    def forward_data(self, data):
        outputs = self(data["images"], data["features"])
        data["prediction"] = outputs
        if "targets" in data:
            data["loss"] = torch.nn.functional.mse_loss(outputs, data["targets"])
        return data


class FeatureExtractorWithSPTAttnAndFMFeaturesLateConcat(nn.Module):
    def __init__(
        self,
        backbone,
        fm_backbone=None,
        feature_map_size=16,
        num_features=512,
        patch_size=2,
        num_fm_features=16,
        fm_feature_map_size=16,
        fm_patch_size=2,
        position_embedding_type="relative_key",
    ):
        super().__init__()
        self.backbone = backbone
        self.fm_backbone = fm_backbone
        self.feature_map_size = feature_map_size
        self.patch_size = patch_size

        # use components from huggingface to create the self-attention layers
        from transformers.models.bert import BertConfig
        from transformers.models.bert.modeling_bert import BertEncoder
        from transformers.models.vit import ViTConfig, ViTModel

        vit_config = ViTConfig(
            image_size=feature_map_size,
            patch_size=patch_size,
            num_channels=num_features,
            hidden_size=64,
            num_hidden_layers=4,
            intermediate_size=32,
            num_attention_heads=4,
        )
        self.vit = ViTModel(
            vit_config,
            add_pooling_layer=False,
        )
        fm_vit_config = ViTConfig(
            image_size=fm_feature_map_size,
            patch_size=fm_patch_size,
            num_channels=num_fm_features,
            hidden_size=64,
            num_hidden_layers=4,
            intermediate_size=32,
            num_attention_heads=4,
        )
        self.fm_vit = ViTModel(fm_vit_config, add_pooling_layer=False)

        bert_config = BertConfig(
            attn_implementation="eager",
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=32,
            hidden_size=128,
            position_embedding_type=position_embedding_type,
            max_position_embeddings=1024,
        )
        self.bert = BertEncoder(bert_config)

        # self.bert.forward()

        self.fc = torch.nn.Linear(128, 6)

    def forward(self, x, fm_features=None, fm_image=None):
        B, N, C, H, W = x.shape
        features = self.backbone(x)
        B, N, C, H, W = features.shape
        features = einops.rearrange(
            features, "b n c h w -> (b n) c h w"
        )  # fold sequence dim into batch dim

        if fm_features is None:
            assert self.fm_backbone is not None
            fm_features = self.fm_backbone(fm_image)
        fm_features = einops.rearrange(fm_features, "b n c h w -> (b n) c h w")

        vit_output = self.vit(features).last_hidden_state
        fm_vit_output = self.fm_vit(fm_features).last_hidden_state

        cls_tokens = vit_output[:, 0, :]
        fm_cls_tokens = fm_vit_output[:, 0, :]

        cls_tokens = torch.cat((cls_tokens, fm_cls_tokens), dim=-1)

        cls_tokens = einops.rearrange(cls_tokens, "(b n) c -> b n c", b=B, n=N)
        cls_tokens = self.bert(
            cls_tokens,
        ).last_hidden_state
        outputs = self.fc(cls_tokens)
        return outputs[:, 1:, :]


class ViTForSpatialAttention(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        num_hidden_layers=12,
        num_attention_heads=8,
        intermediate_size=512,
        image_size=224,
        patch_size=16,
        num_channels=3,
        **kwargs,
    ):
        super().__init__()

        conf = ViTConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            image_size=image_size,
            num_attention_heads=num_attention_heads,
            patch_size=patch_size,
            num_channels=num_channels,
            **kwargs,
        )
        self.vit = ViTModel(conf, add_pooling_layer=False)

    def forward(self, image_sequence):
        B, N, C, H, W = image_sequence.shape

        x = einops.rearrange(image_sequence, "b n c h w -> (b n) c h w")
        x = self.vit(x).last_hidden_state

        cls_tokens = x[:, 0, :]
        cls_tokens = einops.rearrange(cls_tokens, "(b n) c -> b n c", b=B, n=N)
        return cls_tokens


class SimpleTemporalAttn(nn.Module):

    def __init__(
        self,
        hidden_size=64,
        num_hidden_layers=4,
        intermediate_size=32,
        num_attention_heads=4,
        features_only=False,
        max_position_embeddings=1024,
        transformer_type='bert',
        input_size=None,
        attn_implementation='eager',
        **kwargs,
    ):
        super().__init__()
        if transformer_type == 'bert':
            self.encoder = BertEncoder(
                BertConfig(
                    attn_implementation=attn_implementation,
                    hidden_size=hidden_size,
                    num_hidden_layers=num_hidden_layers,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    position_embedding_type="relative_key",
                    max_position_embeddings=max_position_embeddings,
                    **kwargs,
                )
            )
        elif transformer_type == 'roformer':
            self.encoder = RoFormerEncoder(
                RoFormerConfig(
                    attn_implementation=attn_implementation,
                    hidden_size=hidden_size,
                    num_hidden_layers=num_hidden_layers,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    max_position_embeddings=max_position_embeddings,
                    **kwargs,
                )
            )

        self.fc = nn.Linear(hidden_size, 6)
        self.features_only = features_only

        if input_size is not None and input_size != hidden_size:
            self.proj = nn.Linear(input_size, hidden_size)
        else: 
            self.proj = None

    def forward(self, features):
        if self.proj is not None: 
            features = self.proj(features)

        hidden_state = self.encoder(features).last_hidden_state
        if self.features_only:
            return hidden_state

        outputs = self.fc(hidden_state)[:, 1:, :]
        return outputs

    def predict(self, data, device):
        return self(data["pooled_cnn_features"].to(device))



# Global and local predictors
# class
