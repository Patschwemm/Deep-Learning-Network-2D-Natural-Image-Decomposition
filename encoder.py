import torch.nn as nn
from typing import List


# simple conv encoder class that extracts the features of a picture


class conv_block(nn.Module):
    def __init__(self, act_ftn, in_channels, out_channels) -> None:
        super().__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding="same"
            ),
            # nn.BatchNorm2d(num_features=out_channels),
            act_ftn,
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )

    def forward(self, x):
        return self.convlayer(x)


class simpleConvEncoder2d(nn.Module):
    def __init__(
        self,
        act_ftn: nn.Module,
        nChannels: List,
    ):
        super().__init__()
        
        encoder = []
        for c_in, c_out in zip(nChannels[:-1], nChannels[1:]):
            encoder.append(
                conv_block(
                    act_ftn=act_ftn,
                    in_channels=c_in,
                    out_channels=c_out,
                )
            )
        self.encoder = nn.Sequential(*encoder)


    def forward(self, x):
        x = self.encoder(x)
        return x

