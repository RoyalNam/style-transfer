import torch
from torch import nn
from blocks import FeatureMapBlock, ContractingBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, hidden_channel=64):
        super().__init__()
        self.upfeature = FeatureMapBlock(in_channels, hidden_channel)
        self.contract1 = ContractingBlock(hidden_channel, hidden_channel*2, use_bn=False, kernel_size=4, act='lrelu')
        self.contract2 = ContractingBlock(hidden_channel*2, hidden_channel*4, kernel_size=4, act='lrelu')
        self.contract3 = ContractingBlock(hidden_channel*4, hidden_channel*8, kernel_size=4, act='lrelu')
        self.final = nn.Conv2d(hidden_channel*8, 1, kernel_size=1)

    def forward(self, x):
        x = self.upfeature(x)
        x = self.contract1(x)
        x = self.contract2(x)
        x = self.contract3(x)
        x = self.final(x)
        return x


if __name__ == '__main__':
    x = torch.randn((1, 3, 256, 256))
    disc = Discriminator(3, 64)
    out = disc(x)
    print(out.shape)
