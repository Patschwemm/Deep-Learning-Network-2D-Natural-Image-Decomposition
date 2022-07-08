import torch 
import torch.nn as nn
from encoder import simpleConvEncoder2d
import modules
from typing import Dict



class DecompositionNetwork(nn.Module):

    
    def __init__(self, prim_dict: Dict, batch_size: int, bias: int=0.1) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.bias = bias
        self.encoder = simpleConvEncoder2d(nLayers=3, nChannelsInit=8, nInputChannels=1)
        outChannels = self.encoder.outputChannels

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
        
        self.convlayer = nn.Sequential(*layers)

        self.outputLayer = nn.Sequential(
            nn.Linear(in_features= 32 * 32 * 32, out_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=256, out_features=64),
            )
        self.outputLayer.apply(self.init_weights)
        self.primitiveLayer = modules.PrimitiveModule(prim_dict, outChannels=64)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(self.bias)


    def forward(self, x):
        encoding = self.encoder(x)
        features = self.convlayer(encoding)
        # print(features.shape)
        features = features.view(self.batch_size, -1)
        fc_layer = self.outputLayer(features)
        # print("primitive predict shape:",primitive.shape)
        primitive = self.primitiveLayer(fc_layer)
        return primitive




