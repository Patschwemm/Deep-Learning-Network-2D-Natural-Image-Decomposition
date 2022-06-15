from turtle import forward, shape
import torch
import torch.nn as nn


class ShapePredModule(nn.Module):
    def __init__(self, params, outChannelsV, biasTerm=None) -> None:
        super().__init__()
        shapeLayer = nn.Conv2d(
            in_channels=outChannelsV,
            out_channels=params.nz,
            kernel_size=1
            )
        self.shapeLrDecay = params.shapeLrDecay
        shapeLayer.note = "shapePred"

        # could apply weights initialisation here, in 3D case Gridbound has been applied
        self.shapeLayer = shapeLayer

    def forward(self, feature):
        x = self.shapeLayer(feature)
        x = x * self.shapeLrDecay
        x = nn.Sigmoid(x)
        x = x.view(feature.size(0), -1)
        return x


class PositionPredModule(nn.Module):
    def __init__(self, params, outChannelsV, biasTerm=None) -> None:
        super().__init__()


    def forward(self, x):
        pass

