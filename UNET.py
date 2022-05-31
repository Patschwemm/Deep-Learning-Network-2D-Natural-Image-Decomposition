# Implement raw U-net
from unittest import skip
from urllib.parse import uses_params
from numpy import append
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            # bias gets cancelled by the batch norm afterwards
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Downscaling of the layers:
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels=feature

        #Upscaling of the layers (with added skip connection, -> feature*2):
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.midlayer = DoubleConv(features[-1], features[-1] * 2)
        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        #save the skip connections
        for down_layer in self.downs:
            x = down_layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.midlayer(x)
        #reverse the skip connections, so we go with the lowest to highest skip connection
        skip_connections = skip_connections[::-1]

        # because we want to do the upscaling layer and then a double convolution
        for idx in range(0, len(self.ups), 2):
            # upsamling
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            # concat the skip connection

            # it might be that the shapes to do match, 
            # because max pooling takes the floor of a number devided by 2 (if pixel number uneven)
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection,x ), dim=1)
            # run through the double convolution
            x = self.ups[idx+1](concat_skip)

        return self.final_layer(x)

def test():
    x = torch.randn((3, 3, 160, 160))
    model = UNET(in_channels=3, out_channels= 3)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

        
if __name__== "__main__":
    test()




        


        



