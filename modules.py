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


class RotationPredModule(nn.Module):
    def __init__(self, params, outChannelsV, biasTerm=None) -> None:
        super().__init__()
        self.rotationLayer = nn.Conv2d(
            in_channels=outChannelsV, 
            out_channels=1, 
            kernel_size=1
            )
        self.rotationLayer.note = "rotationPred"

        def forward(self, feature):
            x = self.rotationLayer(feature)
            x = x.view(x.size(0), -1)


class PositionPredModule(nn.Module):
    def __init__(self, params, outChannelsV, biasTerm=None) -> None:
        super().__init__()
        self.positionLayer = nn.Con2d(
            in_channels=outChannelsV,
            out_channels=2, 
            kernel_size=1
            )
        self.positionLayer.note = "positionPred"

    def forward(self, feature):
        x = self.positionLayer(feature)
        x = x.view(x.size(0), -1)
        return x

class ProbPredModule(nn.Module):
    def __init__(self, params, outChannelsV, biasTerms=None) -> None:
        super().__init__()
        self.probLayer = nn.Conv2d(outChannelsV, 1, kernel_size=1)
        self.probLayer.note = "probPred"
        self.prune = params.prune
        # self.probLrDecay = params.probLrDecay

    def forward(self, feature):
        x = self.probLayer(feature)
        # x = x * self.probLrDecay
        x = nn.Sigmoid(x)

        stochastic_outputs = x.view(feature.size(0), -1).bernoulli()
        selections = stochastic_outputs
        if self.prune == 0:
            selections = torch.autograd(torch.FloatTensor(x.size()).fill_(1).type_as(x.data))
        return torch.cat([x, selections], dim=1), stochastic_outputs
    

class PrimitivePredModule(nn.Module):

    def __init__(self, params, outChannelsV, biasTerm) -> None:
        super().__init__()
        self.shapePred = ShapePredModule(params, outChannelsV, biasTerm)
        self.rotationPred = RotationPredModule(params, outChannelsV, biasTerm)
        self.positionPred = PositionPredModule(params, outChannelsV, biasTerm)
        self.probPred = ProbPredModule(params, outChannelsV, biasTerm)

    def forward(self, feature):
        shape = self.shapePred(feature)
        rotation = self.rotationPred(feature)
        position = self.positionPred(feature)
        probPred, stochastic_outputs = self.probPred(feature)
        output = torch.cat([shape, rotation, position, probPred], dim = 1)
        return output, stochastic_outputs



        
