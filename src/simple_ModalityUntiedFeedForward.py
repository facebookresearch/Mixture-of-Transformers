###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
from typing import Optional

import torch

from src.utils import merge_modalities, SimpleRMSNorm
from torch.nn import functional as F


class SimpleModalityUntiedFeedForward(torch.nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        hidden_dim: int = 1024 * 4,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        n_modalities: int = 2,
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

        merged_output = merge_modalities(expert_outputs, modality_masks)

        return merged_output


class SimpleFeedForward(torch.nn.Module):
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

        self.w1 = torch.nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w3 = torch.nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = torch.nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [BxS, in_D] -> [BxS, out_D]
        x1 = self.w1(x)
        x3 = self.w3(x)
        output = self.w2(F.silu(x1) * x3)
        return output
