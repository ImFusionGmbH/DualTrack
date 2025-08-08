from functools import partial
import einops
from torch import nn

from torchvision.models.video.resnet import (
    BasicBlock,
    Conv3DSimple,
    VideoResNet,
    Conv3DNoTemporal,
    Conv2Plus1D,
    Bottleneck,
)
import torch
import timm
from typing import Optional

from src.models.causal_conv import CausalConv3d


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem"""

    def __init__(self, chans=3) -> None:
        super().__init__(
            nn.Conv3d(
                chans,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


class Conv3DNoDownsampleTemporal(nn.Conv3d):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        midplanes: Optional[int] = None,
        stride: int = 1,
        padding: int = 1,
    ) -> None:

        super().__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=(1, stride, stride),
            padding=(padding, padding, padding),
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride: int) -> tuple[int, int, int]:
        return 1, stride, stride


class CausalConv3DNoDownsampleTemporal(CausalConv3d):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        midplanes: Optional[int] = None,
        stride: int = 1,
        padding: int = 1,
    ) -> None:

        super().__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=(1, stride, stride),
            padding=(padding, padding, padding),
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride: int) -> tuple[int, int, int]:
        return 1, stride, stride


class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution"""

    def __init__(self) -> None:
        super().__init__(
            nn.Conv3d(
                1,
                45,
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                45,
                64,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        

class CausalR2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution"""

    def __init__(self) -> None:
        super().__init__(
            nn.Conv3d(
                1,
                45,
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            CausalConv3d(
                45,
                64,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


class RStemNoTemporal(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Conv3d(
                1,
                45,
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                45,
                64,
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


class VideoResnetFeaturesOnly(VideoResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = None
        self.avgpool = None

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # Flatten the layer to fc
        # x = x.flatten(1)
        # x = self.fc(x)

        return x


class SmallVideoResNetFeaturesOnly(VideoResnetFeaturesOnly):
    def __init__(self, *args, width_decrease_factor=2, **kwargs):
        self.width_decrease_factor = width_decrease_factor
        super().__init__(*args, **kwargs)

    def _make_layer(
        self,
        block,
        conv_builder,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        return super()._make_layer(
            block, conv_builder, planes // self.width_decrease_factor, blocks, stride
        )


def video_rn18_base():
    model = VideoResnetFeaturesOnly(
        BasicBlock,
        [Conv3DNoDownsampleTemporal] * 4,
        [2, 2, 2, 2],
        R2Plus1dStem,
    )
    return model


def causal_video_rn18_base(): 
    model = VideoResnetFeaturesOnly(
        BasicBlock,
        [CausalConv3DNoDownsampleTemporal] * 4,
        [2, 2, 2, 2],
        CausalR2Plus1dStem,
    )
    return model 


def video_rn18_small_temporal_context():
    model = VideoResnetFeaturesOnly(
        BasicBlock, 
        [Conv3DNoDownsampleTemporal, Conv3DNoTemporal, Conv3DNoTemporal, Conv3DNoTemporal], 
        [2, 2, 2, 2], 
        R2Plus1dStem,
    )
    return model 


def causal_video_rn18_small_temporal_context():
    model = VideoResnetFeaturesOnly(
        BasicBlock, 
        [CausalConv3DNoDownsampleTemporal, Conv3DNoTemporal, Conv3DNoTemporal, Conv3DNoTemporal], 
        [2, 2, 2, 2], 
        CausalR2Plus1dStem,
    )
    return model 


def causal_video_rn18_2_frames():
    model = VideoResnetFeaturesOnly(
        BasicBlock, 
        [Conv3DNoTemporal] * 4, 
        [2, 2, 2, 2], 
        CausalR2Plus1dStem
    )
    return model


def tiny_resnet_base():
    model = SmallVideoResNetFeaturesOnly(
        BasicBlock,
        [Conv3DNoDownsampleTemporal] * 4,
        [1, 1, 1, 1],
        R2Plus1dStem,
    )
    return model


def video_rn18_no_temporal():
    return VideoResnetFeaturesOnly(
        BasicBlock, 
        [Conv3DNoTemporal] * 4, 
        [2, 2, 2, 2], 
        RStemNoTemporal,
    )


def tiny_rn_no_temporal(): 
    return SmallVideoResNetFeaturesOnly(
        BasicBlock, 
        [Conv3DNoTemporal] * 4,
        [1, 1, 1, 1],
        RStemNoTemporal,
    )


class VideoResnetWrapperWithPooling(nn.Module):
    """Wraps a video resnet model to output a 2D feature map and accept the 3D input of shape
    (b, n, c, h, w), where:
        b: batch size
        n: number of frames
        c: number of channels
        h: height
        w: width
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(512, 6)

    def forward(self, x):
        x = einops.rearrange(x, "b n c h w -> b c n h w")
        features = self.backbone(x)  # b c n h w
        features_pooled = features.mean(
            (-2, -1)
        )  # TODO: Implement adaptive pooling based on input size
        features_pooled = einops.rearrange(features_pooled, "b c n -> b n c")

        return self.fc(features_pooled)[:, 1:, :]


class VideoResnetWrapperForFeatureMaps(nn.Module):
    """Wraps a video resnet model to output a 2D feature map and accept the 3D input of shape
    (b, n, c, h, w), where:
        b: batch size
        n: number of frames
        c: number of channels
        h: height
        w: width
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        x = einops.rearrange(x, "b n c h w -> b c n h w")
        features = self.backbone(x)  # b c n h w
        features = einops.rearrange(features, "b c n h w -> b n c h w")
        return features


class VideoResnetWrapperForSequenceRegression(nn.Module):
    def __init__(self, backbone, num_features=512):
        super().__init__()
        self.backbone = backbone
        self.num_features = num_features
        self.fc = nn.Linear(num_features, 6)

    def forward(self, x):
        x = einops.rearrange(x, "b n c h w -> b c n h w")
        features = self.backbone(x)  # b c n h w
        features_pooled = features.mean(
            (-2, -1)
        )  # TODO: Implement adaptive pooling based on input size
        features_pooled = einops.rearrange(features_pooled, "b c n -> b n c")

        return self.fc(features_pooled)[:, 1:, :]


class BaselineResnet(nn.Module):
    def __init__(self, model_name="resnet10t"):
        self.resnet = timm.create_model(model_name, in_chans=2, num_classes=6)

    def forward(self, x):
        B, N, C, H, W = x.shape

        assert C == 1, "Shape mismatch"
        x = x[:, :, 0, ::]  # B, N, H, W
        x = x.unfold(1, 2, 1)  # B, N-1, H, W, 2
        x = einops.rearrange(x, "b n h w c -> (b n) c h w")

        x = self.resnet(x)  # (b n) 6
        x = einops.rearrange(x, "(b n) c -> b n c", b=B, n=(N - 1))

        # for compatibility with other models that expect output length to be the same size
        # as input length, we need to pad with zeros.
        # TODO: Fix this implementation - consider using proper attention mechanism
        x = torch.concat(
            [torch.zeros((B, 1, 6), device=x.device, dtype=x.dtype), x], dim=1
        )

        return x
