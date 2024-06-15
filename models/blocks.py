from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return self.conv(x) + x


class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, kernel_size=3, act='relu'):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
            padding_mode='reflect'
        )
        self.instanceNorm = nn.InstanceNorm2d(out_channels)
        self.act = nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2)
        self.use_bn = use_bn

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.instanceNorm(x)
        x = self.act(x)
        return x


class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.instanceNorm = nn.InstanceNorm2d(out_channels)
        self.act = nn.ReLU()
        self.use_bn = use_bn

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.instanceNorm(x)
        return self.act(x)


class FeatureMapBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=7,
            padding=3,
            padding_mode='reflect'
        )

    def forward(self, x):
        return self.conv(x)
