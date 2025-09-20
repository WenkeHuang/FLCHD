from copy import deepcopy
from typing import Literal

import torch
from torch import nn

from ..utilities.misc import init_linear_weights


class CPTConv(nn.Module):
    def __init__(self, in_feat: int, n_cpt: int, hidden_dim: int, out_feat: int = None):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_feat, n_cpt, kernel_size=[1, 1]),
            nn.ReLU(),
        )
        if out_feat is not None:
            self.linear = nn.Linear(hidden_dim, out_feat)

    def forward(self, x: torch.Tensor):
        output: torch.Tensor = self.proj(x)
        output = output.flatten(2)
        if hasattr(self, "linear"):
            output = self.linear(output)
        return output


class CPTProjector(nn.Module):
    def __init__(
        self,
        in_feat: int,
        out_feat: int,
        n_cpt: int,
        proj_type: Literal["Linear", "FC", "MLP"],
        share_params: bool,
        hidden_dim: int = None,
    ):
        """Projection layer, feat -> cpt embeds

        Args:
            in_feat (int): input feature dimension
            out_feat (int): output cpt embedding dimension
            n_cpt (int): number of concepts
            proj_type (Literal[&quot;Linear&quot;, &quot;FC&quot;, &quot;MLP&quot;]): The type of projection layer, Linear, FC, or MLP
            share_params (bool): Whether to share the parameters among the projection layers
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_feat
        if proj_type == "Linear":
            proj = nn.Linear(in_feat, out_feat)
        elif proj_type == "FC":
            proj = nn.Sequential(
                nn.Linear(in_feat, out_feat),
                nn.ReLU(),
            )
        elif proj_type == "MLP":
            proj = nn.Sequential(
                nn.Linear(in_feat, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_feat),
            )

        proj = nn.Sequential(nn.LayerNorm(in_feat), proj)

        if not share_params:
            proj = nn.ModuleList([deepcopy(proj) for _ in range(n_cpt)])

        init_linear_weights(proj)
        self.proj = proj
        self.n_cpt = n_cpt

    def forward(self, x) -> torch.Tensor:
        if isinstance(self.proj, nn.ModuleList):
            return torch.stack([proj(x) for proj in self.proj], dim=1)
        elif isinstance(self.proj, nn.Module):
            return torch.stack([self.proj(x) for _ in range(self.n_cpt)], dim=1)
        else:
            raise ValueError("proj should be nn.Module or nn.ModuleList")


class CPTPredictor(nn.Module):
    def __init__(
        self,
        in_feat: int,
        n_cpt: int,
        activation: nn.Module = None,
        share_params: bool = False,
        dropout: float = 0.0,
    ):
        """Concept prediction layer, cpt embeds -> cls scores

        Args:
            in_feat (int): input cpt embedding dimension
            activation (nn.Module, optional): activation function. Defaults to None.
        """
        super().__init__()
        self.share_params = share_params
        layers = [nn.Linear(in_feat, 1)]
        if activation is not None:
            layers.append(activation)
        if dropout is not None and dropout > 0:
            layers.append(nn.Dropout(dropout))
        head = nn.Sequential(*layers)
        if not share_params:
            head = nn.ModuleList([deepcopy(head) for _ in range(n_cpt)])
        init_linear_weights(head)
        self.head = head
        self.n_cpt = n_cpt

    def forward(self, x) -> torch.Tensor:
        if isinstance(self.head, nn.ModuleList):
            return torch.cat(
                [self.head[i](x[:, i, :]) for i in range(self.n_cpt)], dim=1
            )
        elif isinstance(self.head, nn.Module):
            return torch.cat([self.head(x[:, i, :]) for i in range(self.n_cpt)], dim=1)
        else:
            raise ValueError("proj should be nn.Module or nn.ModuleList")
