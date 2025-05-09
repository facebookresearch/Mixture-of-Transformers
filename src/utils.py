###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
from typing import List

import torch


def merge_modalities(
    expert_outputs: List[torch.Tensor], modality_masks: List[torch.Tensor]
):
    """
    Merge modality-specific outputs into a unified tensor.

    Args:
        expert_outputs: List of modality-specific outputs. Each expert_output has shape [bs x seq_len, D].
        modality_masks: List of modality-specific masks. Each mask has shape [bs x seq_len].
    """
    assert len(expert_outputs) == len(modality_masks)
    assert len(expert_outputs) > 0

    if len(expert_outputs) == 1:
        return expert_outputs[0]

    merged = torch.empty_like(expert_outputs[0])
    for i in range(len(expert_outputs) - 1, -1, -1):
        expert_output = expert_outputs[i]
        merged[modality_masks[i]] = expert_output
    return merged


class SimpleRMSNorm(torch.nn.Module):
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
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore
