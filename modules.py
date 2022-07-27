from turtle import forward, shape
import torch
import torch.nn as nn



class RectPredModule(nn.Module):
    def __init__(self, outChannels) -> None:
        super().__init__()
        self.rectLayer = nn.Linear(
            in_features=outChannels, 
            out_features=4
            )
        self.rectLayer.note = "rectPred"

    def forward(self, feature):
        x = self.rectLayer(feature)
        x = x.view(x.size(0), -1)
        return x


class PrimitiveModule(nn.Module):

    def __init__(self, prim_dict, outChannels) -> None:
        super().__init__()
        self.rect_count = prim_dict["Rectangles"]
        self.tri_count = prim_dict["Triangles"]
        self.square_count = prim_dict["Circles"]
        self.rectPreds = nn.ModuleList([(RectPredModule(outChannels)) for i in range(self.rect_count)])
        # for future work insert triangle and square module list here

    def forward(self, feature):
        primitives_list = []
        for prim_type_layer in self.rectPreds:
            primitives_list.append(prim_type_layer(feature))
        output = torch.stack(primitives_list)
        return output



        
