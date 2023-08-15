import tensorflow as tf

RouterOutput = Any



class Router(nn.Module):
    """Abstract base router class, defining router API and inner workings.

    Attributes:
      router_weights: Configurable module used to compute router logits from token inputs.
      dtype: Numeric float type for returned combine array. All actual
        computations are performed in float32 of the input for stability.
      ignore_padding_tokens: Whether to ignore padding tokens during routing. Note
        that some routers (e.g. TokensChooseMaskedRouter) will completely ignore
        padding tokens, while others (e.g. TokensChooseScatterRouter and
        ExpertsChooseMaskedRouter) will simply down-weight the probability of
        selecting padding tokens.
    """
    router_weights: RouterWeights
    dtype: jnp.dtype
    ignore_padding_tokens: bool

    def __call__(self, token_inputs, num_experts, expert_capacity, apply_jitter=False) -> RouterOutput:
        """Computes dispatch and combine arrays for routing to experts.

        Args:
          token_inputs: <float>[num_groups, tokens_per_group, hidden_dim] inputs to
            send to experts.
          num_experts: Number of experts.
          expert_capacity: Each group will send this many tokens to each expert.
          apply_jitter: If true, apply jitter noise during routing.

        Returns:
          Router indices or mask arrays (depending on router type).
        """

        # 1. Get router probabilities.
        router_probs, router_logits = self._compute_router_probabilities(token_inputs, num_experts, apply_jitter)

        # 2. No padding mask for now.
        padding_mask = None

        # 3. Compute dispatch and combine arrays.
        dispatch_mask, combine_array, auxiliary_loss = self._compute_routing_instructions(router_probs, padding_mask, expert_capacity)


        z_loss = _router_z_loss(router_logits)

        return dispatch_mask, combine_array, auxiliary_loss, z_loss

    def _compute_router_probabilities(self, token_inputs, num_experts):
        """Computes router probabilities from input tokens.

        Args:
          token_inputs: <float>[num_groups, tokens_per_group, hidden_dim] from which
            router probabilities are computed.
          num_experts: Number of experts.
        Returns:
          - <float32>[num_groups, tokens_per_group, num_experts] probabilities for
            each token and expert. Used for routing tokens to experts.
          - <float>[num_groups, tokens_per_group, num_experts] raw router logits.
            Used for computing router z-loss.
        """
        # For remainder of routing computation we use float32 to ensure stability.
        # See the discussion of "selective precision" in
        # https://arxiv.org/abs/2101.03961.
        token_inputs = jax.lax.convert_element_type(token_inputs, jnp.float32)


        # Shape: [batch, seq_len, num_experts]
        router_logits = self.router_weights(token_inputs, num_experts)

        router_probabilities = jax.nn.softmax(router_logits, axis=-1)

        return router_probabilities, router_logits

    def _compute_routing_instructions(self, router_probs, padding_mask, expert_capacity):
        """Computes masks for the highest probability token per expert.

        Args:
          router_probs: <float32>[batch, seq_len, num_experts]
            probabilities used to determine the routing of tokens to the experts.
          padding_mask: <float32>[batch, seq_len] padding logit mask
            used to identify padding tokens that should be down-weighted by the
            router.
          expert_capacity: Each group will send this many tokens to each expert.

        Returns:
            Dispatch and combine arrays for routing with masked matmuls.
        """

        # This is simply the sequence length
        tokens_per_group = router_probs.shape[1]

        if padding_mask is not None:
            # Because experts choose tokens, we mask probabilities corresponding to
            # tokens before the top-k operation. Note that, unlike for masked-based
            # tokens-choose routing, the experts here may still choose to select the
            # (down-weighted) padding tokens.
            router_probs *= jnp.expand_dims(padding_mask, axis=-1)

        # Swap second and third axes for top-k selection.
        # Transform: [batches, seq_len, num_experts] --> [batches, num_experts, seq_len]
        router_probs_t = jax.vmap(lambda m: m.transpose())(router_probs)

        # Returns probability of capacity for each expert
        # Transform: [batches, num_experts, seq_len] --> [batches, num_experts, expert_capacity].
        expert_gate, expert_index = _top_k(router_probs_t, k=expert_capacity)

        # Convert to one-hot mask of expert indices for each token in each group.
        # Transform: [batches, num_experts, expert_capacity] --> [batch, num_experts, expert_capacity, seq_len].
        dispatch_mask = jax.nn.one_hot(expert_index, tokens_per_group, dtype=jnp.int32)

        # Move axes to conform with shape expected by MoeLayer API.
        # Shape: [batch, num_experts, expert_capacity, seq_len] --> [batch, seq_len, num_experts, expert_capacity]
        dispatch_mask = jnp.moveaxis(dispatch_mask, 3, 1)

        # The combine array will be used for combining expert outputs, scaled by the router probabilities.
        # Transform: [batch, seq_len, num_experts, expert_capacity] --> [batch, num_experts, seq_len, expert_capacity].
        combine_array = jnp.einsum(
            '...ec,...tec->...tec',
            expert_gate,
            dispatch_mask,
            precision=jax.lax.Precision.DEFAULT
        )

        # Return to default dtype now that router computation is complete.
        combine_array = jax.lax.convert_element_type(combine_array, self.dtype)

        # Each expert is choosing tokens until it reaches full capacity, so we don't
        # need an auxiliary loading balancing loss for expert choice routing.
        auxiliary_loss = 0.0

        return dispatch_mask, combine_array, auxiliary_loss
















def _router_z_loss(router_logits):
    num_groups, tokens_per_group, _ = tf.shape(router_logits)
    log_z = tf.math.reduce_logsumexp(router_logits, axis=-1)
    z_loss = tf.math.square(log_z)
    return tf.math.reduce_sum(z_loss, dtype=tf.float32) / (num_groups * tokens_per_group)

def _top_k(array: tf.Tensor, k: int):
    """Returns top k values and their indices along the last axis of the array.

    Args:
        array: Source array.
        k: Number of top values to select.

    Returns:
        - Top k values
        - Associated top k indices.
    """
    top_k_values, top_k_indices = tf.math.top_k(array, k=k)
    top_k_values = _take_along_axis(array, top_k_indices, axis=-1)
    return top_k_values, top_k_indices

def _take_along_axis(array: tf.Tensor, indices: tf.Tensor, axis: int) -> tf.Tensor:
    """Takes values from the input array by matching 1D index and data slices.

    Args:
        array: Source array.
        indices: Indices to take along each 1D slice of array.
        axis: Axis along which to take 1D slices.

    Returns:
        The indexed result.
    """
    if array.ndim != indices.ndim:
        raise ValueError(
            'indices and array must have the same number of dimensions; '
            f'{indices.ndim} vs. {array.ndim}.')

    if (axis != -1 and axis != array.ndim - 1 and  # Not last dimension
        axis != 1 and axis != -array.ndim + 1):  # Not second dimension
        raise ValueError(
            'Only slices along the second or last dimension are supported; '
            f'array.ndim = {array.ndim}, while axis = {axis}.')

    one_hot_length = array.shape[axis]
    one_hot_indices = tf.one_hot(indices, one_hot_length, axis=axis)

    if axis == -1 or array.ndim == 1:
        # Take i elements from last dimension (s).
        result = tf.reduce_sum(
            array * one_hot_indices,
            axis=-2,
            keepdims=False
        )
    else:
        # Take i elements from second dimension (s). We assume here that we always
        # want to slice along the second dimension.
        result = tf.reduce_sum(
            array * one_hot_indices,
            axis=1,
            keepdims=False
        )

    return tf.cast(result, array.dtype)
