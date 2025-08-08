import einops
import h5py
from torch import chunk, nn
import torch
import math


class FrozenModuleWrapper(nn.Module):
    def __init__(self, module, frozen=True, allow_grad_passthrough=False, always_eval_mode=None):
        super().__init__()
        self.module = module
        self.frozen = frozen
        self.allow_grad_passthrough = allow_grad_passthrough
        self.always_eval_mode = always_eval_mode

        if self.frozen:
            for p in self.module.parameters():
                p.requires_grad = False
            self.module.eval()

    def forward(self, *args, **kwargs):
        grad_enabled = False
        if self.allow_grad_passthrough: 
            grad_enabled = True 
        if not self.frozen: 
            grad_enabled = True
        with torch.set_grad_enabled(grad_enabled):
            return self.module(*args, **kwargs)

    def train(self, mode: bool = True):
        out = super().train(mode)
        if self.frozen or self.always_eval_mode:
            self.module.eval()
        return out


def apply_image_module_to_video(model, video):
    B, N, C, H, W = video.shape
    images = einops.rearrange(video, "b n c h w -> (b n) c h w")
    outputs = model(images)
    return einops.rearrange(outputs, "(b n) c h w -> b n c h w", b=B, n=N)


class ApplyImageModuleToVideo(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, video):
        B, N, C, H, W = video.shape
        images = einops.rearrange(video, "b n c h w -> (b n) c h w")
        outputs = self.module(images)
        return einops.rearrange(outputs, "(b n) c h w -> b n c h w", b=B, n=N)


def apply_model_chunked_cpu(model, inputs, chunk_size, chunk_dim):
    N = inputs.shape[chunk_dim]
    n_chunks = N // chunk_size + 1

    outputs = []
    for sub_input in torch.chunk(inputs, n_chunks, dim=chunk_dim):
        outputs.append(model(sub_input).detach().cpu())

    return torch.cat(outputs, dim=chunk_dim)


def apply_model_chunked(model, inputs, chunk_size, chunk_dim):
    N = inputs.shape[chunk_dim]
    n_chunks = N // chunk_size + 1

    outputs = []
    for sub_input in torch.chunk(inputs, n_chunks, dim=chunk_dim):
        outputs.append(model(sub_input))

    return torch.cat(outputs, dim=chunk_dim)


def apply_lora_conversion(model, rank):
    import loralib as lora

    def _apply_lora_conversion(model, rank):
        for name, child in model.named_children():
            if isinstance(child, torch.nn.Linear):
                new_linear = lora.Linear(child.in_features, child.out_features, r=rank)
                setattr(model, name, new_linear)
            elif isinstance(child, torch.nn.Embedding):
                new_emb = lora.Embedding(
                    child.num_embeddings, child.embedding_dim, rank
                )
                setattr(model, name, new_emb)

            _apply_lora_conversion(child, rank)

    _apply_lora_conversion(model, rank)
    lora.mark_only_lora_as_trainable(model)


class ViTWrapper(nn.Module): 
    def __init__(self, vit): 
        super().__init__()
        self.vit = vit 

    def forward(self, x): 
        B, N, C, H, W = x.shape 

        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.vit(x)
        x = x[:, 0, :] # pull cls token 
        x = einops.rearrange(x, '(b n) ... -> b n ... ', b=B)
        return x


class BertWrapper(nn.Module):
    """Wrapper for huggingface models to return last hidden state directly"""
    
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs).last_hidden_state


def init_tracking_head(layer):
    """Initializes a linear layer to have the right mean and std output.""" 
    torch.nn.init.xavier_normal_(layer.weight)
    torch.nn.init.zeros_(layer.bias)

    stds = torch.tensor([
        0.2799165158083181,
        0.053442486743915385,
        0.2535340056702658,
        0.057689968886665575,
        0.04319821396427995,
        0.0593536838407785,
    ])
    means = torch.tensor(
        [
        -0.005158981479095082,
        -0.001158850745153325,
        0.0013826683913399722,
        0.0023931699528833,
        -0.002536075708789752,
        0.002052411696092074,
        ]
    )
    layer.weight.data *= (stds.unsqueeze(-1) / math.sqrt(2))
    layer.bias.data += means


class TwoDModuleFor3DSequenceWrapper(nn.Module):
    def __init__(self, module, window_size=2, max_batch_size_for_computation=512):
        super().__init__()
        self.module = module
        self.window_size = window_size
        self.max_batch_size_for_computation = max_batch_size_for_computation

    def forward(self, x: torch.Tensor):
        B, N, C, H, W = x.shape  # x must have sequence dim at position 1
        x = x.unfold(1, self.window_size, 1)  # B N - 1 C H W window_size

        B, N_new, C, H, W, window_size = x.shape
        x = einops.rearrange(x, "b n c h w window_size -> (b n) (c window_size) h w")

        outputs = []
        for x_as_minibatch in torch.utils.data.DataLoader(
            x, batch_size=self.max_batch_size_for_computation, shuffle=False
        ):
            outputs.append(self.module(x_as_minibatch))
        x = torch.concatenate(outputs, dim=0)
        x = einops.rearrange(x, "(b n) ... -> b n ...", n=N_new)

        return x


