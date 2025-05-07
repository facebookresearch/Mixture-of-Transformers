###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


class ModalityUntiedFeedForward(torch.nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        hidden_dim: int = 1024 * 4,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        mp_size: int = 1,
        n_modalities: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.n_modalities = n_modalities

        self.local_experts = torch.nn.ModuleList(
            [
                SimpleFeedForward(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    multiple_of=multiple_of,
                )
                for _ in range(self.n_modalities)
            ]
        )

        self.local_experts_ffn_norm = torch.nn.ModuleList(
            [SimpleRMSNorm(dim, eps=1e-5) for _ in range(self.n_modalities)]
        )

    def forward(
        self,
        x,
        modality_masks,
    ):

        expert_outputs = []
        for i in range(self.n_modalities):
            expert_input = x[modality_masks[i]]
            expert_output = self.local_experts[i](expert_input)
            expert_output = self.local_experts_ffn_norm[i](expert_output)
            expert_outputs.append(expert_output)

        merged_output = torch.empty(
            (x.size(0), expert_outputs[0].size(1)),
            device=x.device,
            dtype=x.dtype,
        )
        for i in range(self.n_modalities - 1, -1, -1):
            expert_output = expert_outputs[i]
            merged_output[modality_masks[i]] = expert_output

        return merged_output


class SimpleFeedForward(nn.Module):
    # taken from https://github.com/facebookresearch/lingua/blob/main/lingua/transformer.py
    def __init__(
        self,
        dim: int = 1024,
        hidden_dim: int = 1024 * 4,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        mp_size: int = 1,
    ):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B S D
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        output = self.w2(F.silu(x1) * x3)
        return output


class SimpleRMSNorm(nn.Module):
    # taken from https://github.com/facebookresearch/lingua/blob/main/lingua/transformer.py
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore
