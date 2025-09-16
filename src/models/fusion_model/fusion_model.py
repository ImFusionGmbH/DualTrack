import logging
import os
import typing

import torch
from src.engine.tracking_estimator import BaseTrackingEstimator
from src.models.model_registry import register_model, get_model
from torch import nn

from src.models.utils import BertWrapper, FrozenModuleWrapper
from .sampler import SparseSampler, RegularGridSampler
from src.models import local_encoder
from src.models import global_encoder


class DualTrack(BaseTrackingEstimator, nn.Module):
    def __init__(
        self,
        global_encoder,
        local_encoder,
        fusion_module,
        head,
        sampler=RegularGridSampler(16),
        disable_global_encoder=False,
    ):
        super().__init__()
        self.global_encoder = global_encoder
        self.local_encoder = local_encoder
        self.fusion_module = fusion_module
        self.head = head
        self.sampler = sampler
        self.disable_global_encoder = disable_global_encoder

    def _forward(self, global_encoder_inputs, local_encoder_inputs): 
        local_features = self.local_encoder(local_encoder_inputs)
        sparse_indices = self.sampler(global_encoder_inputs, local_features)

        # select the global encoder inputs corresponding to the indices
        B = global_encoder_inputs.shape[0]

        subsampled_global_encoder_inputs = []
        for i in range(B): 
            subsampled_global_encoder_inputs.append(
                global_encoder_inputs[i][sparse_indices[i]].contiguous()
            )        
        global_encoder_inputs = torch.stack(
            subsampled_global_encoder_inputs, dim=0
        )

        if self.disable_global_encoder:
            global_features = None
        else:
            global_features = self.global_encoder(global_encoder_inputs, sparse_indices)

        tracking_decoder_outputs = self.fusion_module(
            local_features, encoder_hidden_states=global_features
        )
        return self.head(tracking_decoder_outputs)

    @typing.overload
    def forward(self, global_encoder_inputs, local_encoder_inputs):
        ...
    
    @typing.overload 
    def forward(self, inputs: dict): 
        ...

    def forward(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], dict): 
            return self._forward(**args[0])
        else:
            return self._forward(*args, **kwargs)

    def forward_dict(self, data):
        device = self.device

        sweep_ids = data['sweep_id']
        self.set_current_data_ids(sweep_ids)

        if "local_encoder_images" in data:
            local_encoder_inputs = data["local_encoder_images"].to(device)
            self.local_encoder[0].input_mode = "images"
        elif "local_encoder_intermediates" in data:
            local_encoder_inputs = data["local_encoder_intermediates"].to(device)
            self.local_encoder[0].input_mode = "features"
        else:
            raise ValueError(
                "One of `local_encoder_images` or `local_encoder_intermediates` should be provided."
            )

        global_encoder_inputs = data["global_encoder_images"].to(device)

        data["prediction"] = self(global_encoder_inputs, local_encoder_inputs)

        return data

    def validate(self, data):
        self.eval()
        assert "local_encoder_images" in data
        assert "local_encoder_intermediates" in data

        self.local_encoder.input_mode = "images"
        intermediates = self.local_encoder.forward_intermediates(
            data["local_encoder_images"].to(self.device)
        )

        assert torch.allclose(
            intermediates, data["local_encoder_intermediates"].to(self.device)
        )

    def set_current_data_ids(self, data_ids): 
        def _apply(mod): 
            mod._current_data_ids = data_ids
        self.apply(_apply)

    def predict(self, batch):
        batch = self.forward_dict(batch)
        pred = batch["prediction"]

        # pred = model.predict(batch, device)
        return pred

    def get_loss(self, batch, pred=None):
        pred = pred if pred is not None else self.predict(batch)

        targets = batch["targets"].to(self.device)
        padding_lengths = batch["padding_size"]

        mse_loss = nn.MSELoss(reduction="none")

        def _get_loss(pred, targets):
            B, N, D = (
                pred.shape
                if isinstance(pred, torch.Tensor)
                else list(pred.values())[0].shape
            )

            mask = torch.ones(B, N, D, dtype=torch.bool, device=self.device)
            for i, padding_length in enumerate(padding_lengths):
                if padding_length > 0:
                    mask[i, -padding_length:, :] = 0

            loss = mse_loss(pred, targets)
            masked_loss = torch.where(mask, loss, torch.nan)
            mse_loss_val = masked_loss.nanmean()
            return mse_loss_val

        if isinstance(pred, dict):
            loss = torch.tensor(0.0, device=self.device)
            if "local" in pred:
                loss += _get_loss(pred["local"], targets)
            if "global" in pred:
                loss += _get_loss(pred["global"], batch["targets_global"].to(self.device))
            if "absolute" in pred:
                loss += _get_loss(pred["absolute"], batch["targets_absolute"].to(self.device))
            return loss
        else:
            return _get_loss(pred, targets)

    @property
    def device(self): 
        return next(self.parameters()).device


class LocalTrackingEstimationHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, 6)

    def forward(self, inputs):
        return self.fc(inputs)[:, 1:, :]


def _get_bert_encoder(implementation="orig", **kwargs):
    # Ensure default attention implementation is eager for HF and local BERT
    kwargs.setdefault("attn_implementation", "eager")
    kwargs.setdefault("_attn_implementation", "eager")
    if implementation == "orig":
        from transformers.models.bert.modeling_bert import BertConfig, BertEncoder

        _cfg = BertConfig(**kwargs)
        # Robustly set private attribute expected by HF modeling code
        if not hasattr(_cfg, "_attn_implementation") or _cfg._attn_implementation is None:
            _cfg._attn_implementation = "eager"
        return BertEncoder(_cfg)
    else:
        from src.models.bert import BertConfig, BertEncoder

        return BertEncoder(BertConfig(**kwargs))


@register_model
def dualtrack_fusion_model(
    *,
    global_encoder_cfg=dict(name="global_encoder_cnn"),
    local_encoder_cfg=dict(name="dualtrack_loc_enc_stg3"),
    disable_global_encoder=False,
    freeze_global_image_encoder=False,
    decoder_hidden_size=512,
    grid_spacing=16,
    max_position_embeddings=1024,
):

    # === Global encoder ===
    if not disable_global_encoder:
        global_encoder_cfg["features_only"] = True
        global_encoder_backbone = get_model(**global_encoder_cfg)

        if freeze_global_image_encoder:
            global_encoder_backbone.backbone = FrozenModuleWrapper(
                global_encoder_backbone.backbone
            )
        global_encoder = global_encoder_backbone
    else:
        global_encoder = None

    # === Local encoder ===
    local_encoder_cfg["features_only"] = True
    local_encoder_backbone = get_model(**local_encoder_cfg)
    local_encoder = nn.Sequential(
        local_encoder_backbone, nn.Linear(64, decoder_hidden_size, bias=False)
    )

    # === Decoder and head ===
    head = LocalTrackingEstimationHead(decoder_hidden_size)
    decoder = BertWrapper(
        _get_bert_encoder(
            implementation="orig",
            hidden_size=decoder_hidden_size,
            intermediate_size=decoder_hidden_size * 2,
            num_attention_heads=8,
            position_embedding_type="relative_key",
            max_position_embeddings=max_position_embeddings,
            is_decoder=not disable_global_encoder,
            add_cross_attention=not disable_global_encoder,
        )
    )

    return DualTrack(
        global_encoder,
        local_encoder,
        decoder,
        head,
        sampler=RegularGridSampler(grid_spacing),
        disable_global_encoder=disable_global_encoder,
    )


@register_model
def dualtrack_tus_rec_2024(pretrained=True, **kwargs):
    model = dualtrack_fusion_model(
        local_encoder_cfg=dict(name="dualtrack_loc_enc_stg3_legacy"), **kwargs
    )
    if pretrained: 
        path = os.getenv(
            "DUALTRACK_FINAL_CHECKPOINT_PATH"
        ) or 'data/checkpoints/dualtrack_final.pt'

        print(model.load_state_dict(
            torch.load(path, map_location='cpu')
        ))

    return model 
