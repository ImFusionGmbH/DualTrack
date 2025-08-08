"""
Implements basic causal convolutions and a ResNet model with causal convolutions.
"""


import torch
from torch import nn
import einops


class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        norm=nn.BatchNorm3d,
        conv_layer=nn.Conv3d,
    ):

        super().__init__()

        self.conv1 = conv_layer(
            in_channels,
            out_channels,
            kernel_size,
            stride=(1, stride, stride),
            padding=padding,
            bias=False,
        )
        self.norm1 = norm(out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = conv_layer(
            out_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.norm2 = norm(out_channels)
        self.act2 = nn.ReLU()

        use_downsample = (stride > 1) or in_channels != out_channels
        if use_downsample:
            self.downsample = torch.nn.Sequential(
                torch.nn.AvgPool3d(
                    kernel_size=(1, stride, stride), stride=(1, stride, stride)
                ),
                nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=False),
                norm(out_channels),
            )
        else:
            self.downsample = torch.nn.Identity()

    def forward(self, x):
        shortcut = self.downsample(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)

        return self.act2(x + shortcut)


class CausalConv3d(torch.nn.Conv3d):

    def forward(self, input) -> torch.Tensor:
        _, _, k, _, _ = self.weight.data.shape
        self.weight.data[:, :, k // 2 + 1 :, :, :].fill_(0)

        return super().forward(input)


class ResnetForTrackingEstimation(torch.nn.Module):
    def __init__(
        self,
        norm="batch",
        conv_layer=CausalConv3d,
    ):
        super().__init__()
        self.conv1 = conv_layer(1, 64, (1, 7, 7), (1, 2, 2), (0, 3, 3))

        if norm == "batch":
            norm_fn = nn.BatchNorm3d
        elif norm == "group":

            def norm_fn(d):
                return nn.GroupNorm(8, d)

        else:
            raise ValueError()

        self.blocks = torch.nn.ModuleList()
        self.blocks.append(
            ResidualBlock(64, 128, 3, 2, 1, conv_layer=conv_layer, norm=norm_fn)
        )
        self.blocks.append(
            ResidualBlock(128, 256, 3, 2, 1, conv_layer=conv_layer, norm=norm_fn)
        )
        self.blocks.append(
            ResidualBlock(256, 512, 3, 2, 1, conv_layer=conv_layer, norm=norm_fn)
        )

        self.context_length = 4

        self.fc = torch.nn.Linear(512, 6)

    def forward(self, x):
        x = einops.rearrange(x, "b n c h w -> b c n h w")
        x = self.conv1(x)

        for block in self.blocks:
            x = block(x)

        x = x.mean((-1, -2))
        x = einops.rearrange(x, "b c n -> b n c")
        return self.fc(x)[:, 1:, :]


class ConvThenS4Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            CausalConv3d(1, 24, (3, 3, 3), (1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(24),
            nn.ReLU(),
            CausalConv3d(24, 32, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            CausalConv3d(32, 64, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
        )

        self.layer1 = ResidualBlock(
            64, 64, (1, 3, 3), 1, (0, 1, 1), conv_layer=CausalConv3d
        )
        self.layer2 = ResidualBlock(
            64, 128, (1, 3, 3), 2, (0, 1, 1), conv_layer=CausalConv3d
        )
        self.layer3 = ResidualBlock(
            128, 256, (1, 3, 3), 2, (0, 1, 1), conv_layer=CausalConv3d
        )
        self.layer4 = ResidualBlock(
            256, 512, (1, 3, 3), 2, (0, 1, 1), conv_layer=CausalConv3d
        )

        self.seq_model = S4Model(512, 6, 64, lr=1e-4)

        # self.fc = nn.Linear(512, 6)

    def forward(self, x):

        x = einops.rearrange(x, "b n c h w -> b c n h w")

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.mean((-1, -2))
        x = einops.rearrange(x, "b c n -> b n c")

        x = self.seq_model(x)

        return x
