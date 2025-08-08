import einops
from src.models.segment_anything.modeling.image_encoder import Block, ImageEncoderViT
import torch
from torch import nn


class ImageEncoderWrapper(nn.Module):
    def __init__(self, image_encoder, norm='batch'):
        super().__init__()
        self.image_encoder = image_encoder

        norms = dict(
            batch = nn.BatchNorm3d, 
            instance=nn.InstanceNorm3d, 
            none=lambda d: nn.Identity(d)
        )

        self.instance_norms = nn.ModuleList(
            [
                norms[norm](768) for i in range(len(self.image_encoder.blocks))
            ]
        )

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = einops.rearrange(x, "b n c h w -> (b n) c h w")

        x = self.image_encoder.patch_embed(x)
        if self.image_encoder.pos_embed is not None:
            x = x + interpolate_pos_encoding(x, self.image_encoder.pos_embed)

        for i, blk in enumerate(self.image_encoder.blocks):
            x = blk(x)  # B, C, H, W
            x = self.apply_norm(x, self.instance_norms[i], B)

        x = self.image_encoder.neck(x.permute(0, 3, 1, 2))
        x = einops.rearrange(x, '(b n) c h w -> b n c h w', b=B)
        return x

    def apply_norm(self, x, norm, batch_size): 
        x = einops.rearrange(x, "(b n) h w c -> b c n h w", b=batch_size)
        x = norm(x)
        x = einops.rearrange(x, 'b c n h w -> (b n) h w c')
        return x


def get_wrapped_medsam_encoder(norm='batch'):
    from prostNfound.sam_wrappers import build_medsam

    return ImageEncoderWrapper(build_medsam().image_encoder, norm=norm)


def interpolate_pos_encoding(x, pos_embed):
    npatch_in_h = x.shape[1]
    npatch_in_w = x.shape[2]

    patch_pos_embed = pos_embed

    npatch_native_h = patch_pos_embed.shape[1]
    npatch_native_w = patch_pos_embed.shape[2]

    if npatch_native_h == npatch_in_h and npatch_native_w == npatch_in_w:
        return pos_embed

    w0 = npatch_in_w
    h0 = npatch_in_h
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.permute(0, 3, 1, 2),
        scale_factor=(h0 / npatch_native_h, w0 / npatch_native_w),
        mode="bicubic",
    )
    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
    return patch_pos_embed


def forward_return_features(image_encoder: ImageEncoderViT, x, return_hiddens=False):
    # "Return hiddens" feature added

    x = image_encoder.patch_embed(x)
    if image_encoder.pos_embed is not None:
        x = x + interpolate_pos_encoding(x, image_encoder.pos_embed)

    hiddens = []
    for blk in image_encoder.blocks:
        x = blk(x)
        if return_hiddens:
            hiddens.append(x)

    x = image_encoder.neck(x.permute(0, 3, 1, 2))

    return (x, hiddens) if return_hiddens else x


if __name__ == "__main__":
    model = get_wrapped_medsam_encoder()
    inp = torch.randn(1, 8, 3, 224, 224)

    print(model(inp).shape)
