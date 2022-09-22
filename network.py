# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from itertools import chain
# typing
from typing import Tuple, Optional
from itertools import compress


class ConvBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(ConvBlock, self).__init__(
            # conv-block
            nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, ceil_mode=True),
        )

    def get_output_dim(self, w:int, h:int):
        return (
            math.ceil(w / 2), 
            math.ceil(h / 2)
        )

class RectangleModule(nn.Module):

    def __init__(
        self, 
        num_recs:int, 
        hidden_dim:int
    ) -> None:
        # initialize rectangle module
        super(RectangleModule, self).__init__()
        # save number of rectangles
        self.num_recs = num_recs
        # create prediction layers
        self.pred = nn.Linear(hidden_dim, self.num_recs * (2*2+1+1))

    def forward(self, x):
        # predict and split into rectangle and probability and rotation
        rp = self.pred(x).reshape(-1, self.num_recs, 2*2+1+1)
        r, p, theta = rp[..., :4], rp[..., 4], rp[..., 5]
        # unravel rectangles
        r = r.reshape(-1, self.num_recs, 2, 2)
        p = p.reshape(-1, self.num_recs)
        theta = theta.reshape(-1, self.num_recs)
        # apply activations
        r = r.sigmoid()
        p = p.sigmoid()
        theta = theta.sigmoid() * 360
        # return
        return (r, p, theta)

class CircleModule(nn.Module):

    def __init__(
        self, 
        num_circs:int, 
        hidden_dim:int
    ) -> None:
        # initialize rectangle module
        super(CircleModule, self).__init__()
        # save number of rectangles
        self.num_circs = num_circs
        # create prediction layers
        self.pred = nn.Linear(hidden_dim, self.num_circs * (3+1))

    def forward(self, x):
        # predict and split into rectangle and probability
        cp = self.pred(x).reshape(-1, self.num_circs, 3+1)
        c, p = cp[..., :3], cp[..., 3:]
        # unravel rectangles
        c = c.reshape(-1, self.num_circs, 3)
        p = p.reshape(-1, self.num_circs)
        # apply activations
        c = c.sigmoid()
        p = p.sigmoid()
        # return
        return (c, p)

class Model(nn.Module):

    def __init__(
        self, 
        img_size:Tuple[int, int],
        conv_channels:Tuple[int, ...],
        dense_layers:Tuple[int, ...],
        num_recs:int,
        num_circs:int,
    ) -> None:
        super(Model, self).__init__()
        # read image shape
        w, h = img_size
        # encoder
        self.encoder = nn.Sequential(
            *(
                ConvBlock(cin, cout, 3, padding='same')
                for cin, cout in zip(
                    conv_channels[:-1], 
                    conv_channels[1:]
                )
            )
        )

        # compute encoder output dimension
        for block in self.encoder:
            w, h = block.get_output_dim(w, h)
        enc_feat_dim = w * h * conv_channels[-1]
        
        # dense encoder
        self.dense = nn.Sequential(
            *(
                nn.Sequential(
                    nn.Linear(fin, fout),
                    nn.ReLU()
                )
                for fin, fout in zip(
                    chain([enc_feat_dim], dense_layers[:-1]),
                    dense_layers
                )
            )
        )
        # predictors
        hidden_dim = dense_layers[-1]

        prim_list = []
        prim_list.append(RectangleModule(num_recs, hidden_dim) if num_recs > 0 else None)
        prim_list.append(CircleModule(num_circs, hidden_dim)if num_circs > 0 else None)

        self.prim_layer = nn.ModuleList([*prim_list])
        self.num_dict = {"Rectangle": num_recs, "Circle": num_circs}
        print(self.num_dict)

    def forward(self, x) -> torch.Tensor:
        # encode image input
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        x = self.dense(x)
        # pass through predictor
        output = {"Rectangle": torch.Tensor, "Circle": torch.Tensor}
        for prim_type_layer, prim_type in zip(self.prim_layer, self.num_dict):
            if self.num_dict[prim_type] > 0:
                output[prim_type] = prim_type_layer(x)

        return output
        