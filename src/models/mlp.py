import torch
from torch import nn 
import typing as tp 


class MLP(nn.Module): 
    def __init__(self, *dims: tp.Sequence[int]):
        super().__init__()
        self.layers = nn.ModuleList() 

        for i in range(len(dims) - 1): 
            last_layer = i + 1 == len(dims) - 1

            self.layers.append(
                nn.Sequential(
                    nn.Linear(
                        in_features=dims[i], 
                        out_features=dims[i+1]
                    ), 
                    nn.ReLU() if not last_layer else nn.Identity()
                )
            )

    def forward(self, X): 
        for layer in self.layers: 
            X = layer(X)

        return X 
        