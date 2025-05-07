class ModalityUntiedAttention(torch.nn.Module):
    """
    Modality-specific attention with decoupled query, key, value, and output projections,
    along with modality-specific normalization layers.
    """

    def __init__(
        self,
        args: ModelArgs,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float,
        efficient_attn: Optional[str],
        use_rope: bool,
        init_args: InitArgs,
        init_depth: Optional[int],
        norm_type: str,
        norm_eps: float = 1e-5,
        qk_normalization: bool = False,
    ):
        super().__init__()

        self.n_modalities = args.n_modalities

        # Initialize modality-specific query, key, value, and output projections
        self.local_experts_wq = self._create_experts(dim, n_heads * head_dim, init_args)
        self.local_experts_wk = self._create_experts(
            dim, n_kv_heads * head_dim, init_args
        )
        self.local_experts_wv = self._create_experts(
            dim, n_kv_heads * head_dim, init_args
        )
        self.local_experts_wo = self._create_experts(
            n_heads * head_dim, dim, init_args, row_parallel=True
        )

        # QK normalization (if enabled)
        self.head_dim = head_dim
        if qk_normalization:
            self.local_experts_q_normalization = self._create_norms(
                head_dim, self.n_modalities
            )
            self.local_experts_k_normalization = self._create_norms(
                head_dim, self.n_modalities
            )

        # Final output normalization for each modality
        self.local_experts_attention_norm = self._create_norms(
            dim, self.n_modalities, norm_type, norm_eps
        )

        # Inner attention mechanism
        self.inner_attention = _InnerAttention(
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            efficient_attn=efficient_attn,
            use_rope=use_rope,
        )

    def _create_experts(self, input_dim, output_dim, init_args, row_parallel=False):
        """
        Helper to create modality-specific linear projections.
        """
        cls = RowParallelLinear if row_parallel else ColumnParallelLinear
        init_fn = get_init_fn(init_args, input_dim, None)
        return torch.nn.ModuleList(
            [
                cls(
                    input_dim,
                    output_dim,
                    bias=False,
                    init_method=init_fn,
                    params_dtype=torch.get_default_dtype(),
                )
                for _ in range(self.n_modalities)
            ]
        )

    def _create_norms(self, dim, n_modalities, norm_type="rmsnorm", eps=1e-5):
        """
        Helper to create modality-specific normalization layers.
        """
        return torch.nn.ModuleList(
            [
                build_norm_fn(
                    norm_type,
                    dim,
                    eps,
                    True,  # affine
                )
                for _ in range(n_modalities)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        attn_bias: Optional,
        modality_masks: torch.Tensor,
        freqs_cis: Optional[torch.Tensor],
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):

        _, dim = x.shape
        if freqs_cis.ndim == 2:
            slen, _ = freqs_cis.shape
        else:
            _, slen, _ = freqs_cis.shape

        # Extract modality-specific tokens
        expert_outputs_xq, expert_outputs_xk, expert_outputs_xv = self._process_qkv(
            x, modality_masks
        )

        # Merge and reshape for attention computation
        xq, xk, xv = self._merge_qkv(
            expert_outputs_xq,
            expert_outputs_xk,
            expert_outputs_xv,
            x.size(0),
            modality_masks,
            x.dtype,
        )

        xq, xk, xv = (
            xq.unflatten(0, (-1, slen)),
            xk.unflatten(0, (-1, slen)),
            xv.unflatten(0, (-1, slen)),
        )

        # Compute attention output
        output, new_cache = self.inner_attention(
            xq, xk, xv, mask, attn_bias, freqs_cis, cache
        )

        # Process final output with modality-specific projections and normalization
        output = self._process_final_output(output, modality_masks)

        return output, new_cache

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

    def _merge_qkv(
        self,
        expert_outputs_xq,
        expert_outputs_xk,
        expert_outputs_xv,
        batch_size,
        modality_masks,
        dtype,
    ):
        """
        Merge modality-specific Q, K, V outputs into unified tensors.
        """
        xq_merged = self._merge_modalities(
            batch_size,
            expert_outputs_xq,
            modality_masks,
            dtype,
        )
        xk_merged = self._merge_modalities(
            batch_size,
            expert_outputs_xk,
            modality_masks,
            dtype,
        )
        xv_merged = self._merge_modalities(
            batch_size,
            expert_outputs_xv,
            modality_masks,
            dtype,
        )
        return xq_merged, xk_merged, xv_merged

    def _merge_modalities(self, batch_size, expert_outputs, modality_masks, dtype):
        """
        Merge modality-specific outputs into a unified tensor.
        """
        merged = torch.empty(
            (batch_size, expert_outputs[0].size(1)),
            device=expert_outputs[0].device,
            dtype=dtype,
        )
        for i, expert_output in enumerate(expert_outputs):
            merged[modality_masks[i]] = expert_output
        return merged

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
        return self._merge_modalities(
            output.size(0), expert_outputs, modality_masks, output.dtype
        )


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
