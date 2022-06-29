import torch.nn as nn


#simple conv encoder class that extracts the features of a picture
class simpleConvEncoder2d(nn.Module):

    def __init__(self, nLayers, nChannelsInit=8, nInputChannels=3) -> None:
        super().__init__()
        nOutputChannels = nChannelsInit
        encoder = []

        for i in range(nLayers):
            encoder.append(nn.Conv2d(
                in_channels = nInputChannels,
                out_channels=nOutputChannels,
                kernel_size=3,
                padding=1,
                padding_mode="reflect"))
            encoder.append(nn.BatchNorm2d(num_features=nOutputChannels))
            encoder.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            encoder.append(nn.MaxPool2d(kernel_size=2, dilation=2))
            nInputChannels = nOutputChannels
            nOutputChannels = nOutputChannels*2
        self.encoder = nn.Sequential(*encoder)
        # because the last multiplication is not considered
        # for the layers in the loop
        self.outputChannels = nOutputChannels //2
        
    def outputChannels(self):
        return self.outputChannels

    def forward(self, x):
        x = self.encoder(x)
        return x


