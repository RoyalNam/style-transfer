import torch
from torch import nn
from blocks import ResidualBlock, FeatureMapBlock, ContractingBlock, ExpandingBlock


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hidden_channels=64, n_res=9):
        super().__init__()
        self.upfeature = FeatureMapBlock(in_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, hidden_channels*2)
        self.contract2 = ContractingBlock(hidden_channels*2, hidden_channels*8)
        self.contract3 = ContractingBlock(hidden_channels*8, hidden_channels*4)

        self.res = nn.Sequential(
            *[ResidualBlock(hidden_channels*4) for _ in range(n_res)]
        )

        self.expand1 = ExpandingBlock(hidden_channels*4, hidden_channels*8)
        self.expand2 = ExpandingBlock(hidden_channels*8, hidden_channels*2)
        self.expand3 = ExpandingBlock(hidden_channels*2, hidden_channels)
        self.downfeature = FeatureMapBlock(hidden_channels, out_channels)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.upfeature(x)
        x = self.contract1(x)
        x = self.contract2(x)
        x = self.contract3(x)
        x = self.res(x)
        x = self.expand1(x)
        x = self.expand2(x)
        x = self.expand3(x)
        x = self.downfeature(x)
        return self.tanh(x)


if __name__ == '__main__':
    x = torch.randn((1, 3, 256, 256))
    gen = Generator(3, 3, 64, 6)
    out = gen(x)
    print(out.shape)
