from pyrsistent import inc
import torch 
import torch.nn as nn
from encoder import simpleConvEncoder2d
import modules

class Main_Network(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.encoder = simpleConvEncoder2d(nLayers=3, nChannelsInit=8, nInputChannels=3)
        outChannels = self.encoder.outputChannels()

        # construct fully connected layer
        nLayers_fc = 2
        layers = []
        for i in range(nLayers_fc):
            layers.append(nn.Conv2d(
                in_channels=outChannels,
                out_channels=outChannels,
                kernel_size=1))
            layers.append(nn.BatchNorm2d(num_features=outChannels))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        self.fcLayers = nn.Sequential(*layers)

        # possibly add bias term values here
        biasTerms = 0

        # still to be defined
        self.primitivesTable = modules.Primitives(params, outChannels)
