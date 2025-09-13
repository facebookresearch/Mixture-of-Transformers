###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
from typing import Optional, Tuple

import torch

from src.utils import merge_modalities, SimpleRMSNorm


class SimpleModalityUntiedAttention(torch.nn.Module):
    """
    Modality-specific attention with decoupled query, key, value, and output projections,
    along with modality-specific normalization layers.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        dropout: float,
        norm_eps: float = 1e-5,
        qk_normalization: bool = False,
        n_modalities: int = 2,
    ):
        super().__init__()

        self.n_modalities = n_modalities

        # Initialize modality-specific query, key, value, and output projections
        self.local_experts_wq = self._create_experts(dim, n_heads * head_dim)
        self.local_experts_wk = self._create_experts(dim, n_heads * head_dim)
        self.local_experts_wv = self._create_experts(dim, n_heads * head_dim)
        self.local_experts_wo = self._create_experts(n_heads * head_dim, dim)

        # QK normalization (if enabled)
        self.head_dim = head_dim
        if qk_normalization:
            self.local_experts_q_normalization = self._create_norms(
                head_dim, self.n_modalities, eps=norm_eps
            )
            self.local_experts_k_normalization = self._create_norms(
                head_dim, self.n_modalities, eps=norm_eps
            )

        # Final output normalization for each modality
        self.local_experts_attention_norm = self._create_norms(dim, self.n_modalities)

        # Inner attention mechanism
        self.attention_comp = torch.nn.MultiheadAttention(
            head_dim=head_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

    def _create_experts(self, input_dim, output_dim):
        """
        Helper to create modality-specific linear projections.
        """
        return torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    input_dim,
                    output_dim,
                    bias=False,
                    params_dtype=torch.get_default_dtype(),
                )
                for _ in range(self.n_modalities)
            ]
        )

    def _create_norms(self, dim, n_modalities, eps=1e-5):
        """
        Helper to create modality-specific normalization layers.
        """
        return torch.nn.ModuleList(
            [SimpleRMSNorm(dim, eps) for _ in range(n_modalities)]
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        modality_masks: torch.Tensor,
    ):
        # Extract modality-specific tokens
        expert_outputs_xq, expert_outputs_xk, expert_outputs_xv = self._process_qkv(
            x, modality_masks
        )

        #  Merge modality-specific Q, K, V outputs into unified tensors for attention computation
        xq = merge_modalities(expert_outputs_xq, modality_masks)
        xk = merge_modalities(expert_outputs_xk, modality_masks)
        xv = merge_modalities(expert_outputs_xv, modality_masks)

        # Compute attention output
        attn_output, attn_output_weights = self.attention_comp(
            xq, xk, xv, attn_mask=attn_mask
        )

        # Process final output with modality-specific projections and normalization
        attn_output = self._process_final_output(attn_output, modality_masks)

        return attn_output, attn_output_weights

    def _process_qkv(self, x, modality_masks):
        """
        Process query, key, and value projections for each modality.
        """
        expert_outputs_xq, expert_outputs_xk, expert_outputs_xv = [], [], []
        for i in range(self.n_modalities):
            expert_input = x[modality_masks[i]]
            xq = self.local_experts_wq[i](expert_input)
            xk = self.local_experts_wk[i](expert_input)
            xv = self.local_experts_wv[i](expert_input)

            # Apply QK normalization if enabled
            if hasattr(self, "local_experts_q_normalization"):
                # Apply normalization to both xq and xk
                xq = qk_normalize_tensor(
                    xq, self.local_experts_q_normalization[i], self.head_dim
                )
                xk = qk_normalize_tensor(
                    xk, self.local_experts_k_normalization[i], self.head_dim
                )

            expert_outputs_xq.append(xq)
            expert_outputs_xk.append(xk)
            expert_outputs_xv.append(xv)

        return expert_outputs_xq, expert_outputs_xk, expert_outputs_xv

    def _process_final_output(self, output, modality_masks):
        """
        Process final attention output with modality-specific `wo` projections and normalization.
        """
        output = output.flatten(0, 1)

        expert_outputs = []
        for i in range(self.n_modalities):
            expert_input = output[modality_masks[i]]
            expert_output = self.local_experts_wo[i](expert_input)
            expert_output = self.local_experts_attention_norm[i](expert_output)
            expert_outputs.append(expert_output)
        return merge_modalities(expert_outputs, modality_masks)


def qk_normalize_tensor(tensor, normalization_layer, head_dim):
    """
    Normalize a tensor by reshaping it for LayerNorm compatibility.

    Args:
        tensor (torch.Tensor): Input tensor to normalize. Assumes last dim contains `num_heads * head_dim`.
        normalization_layer (nn.LayerNorm): LayerNorm instance for normalization.
        head_dim (int): Dimension of the head for reshaping.

    Returns:
        torch.Tensor: Normalized tensor reshaped back to its original structure.
    """
    # Infer the shape dynamically
    original_shape = tensor.shape  # Save the original shape
    num_heads = tensor.size(-1) // head_dim
    reshaped_tensor = tensor.reshape(
        *original_shape[:-1], num_heads, head_dim
    )  # Reshape last dim
    normalized_tensor = normalization_layer(reshaped_tensor)  # Apply normalization
    return normalized_tensor.reshape(*original_shape)  # Restore original shape
