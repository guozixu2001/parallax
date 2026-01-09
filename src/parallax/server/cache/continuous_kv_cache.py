"""
Continuous KV Cache using concatenate strategy (non-paged).
Used for speculative decoding with SDPA.
"""

from typing import Tuple

import mlx.core as mx


class ContinuousKVCache:
    """
    Continuous KV cache using concatenate strategy.

    Unlike paged KV cache, this uses a simple concatenate-based approach
    for appending new key-value pairs. Suitable for speculative decoding
    where we need frequent rollback operations.

    Attributes:
        max_seq_len: Maximum sequence length
        num_kv_heads: Number of key-value heads
        head_dim: Dimension of each head
        dtype: Data type for cache
        key_cache: Key cache tensor (1, max_seq_len, num_kv_heads, head_dim)
        value_cache: Value cache tensor (1, max_seq_len, num_kv_heads, head_dim)
        current_length: Current length of cached sequence
    """

    def __init__(self, max_seq_len: int, num_kv_heads: int, head_dim: int, dtype: type):
        """
        Initialize ContinuousKVCache.

        Args:
            max_seq_len: Maximum sequence length
            num_kv_heads: Number of key-value heads
            head_dim: Dimension of each head
            dtype: Data type for cache tensors
        """
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Initialize empty KV cache
        self.key_cache = mx.zeros((1, max_seq_len, num_kv_heads, head_dim), dtype=dtype)
        self.value_cache = mx.zeros((1, max_seq_len, num_kv_heads, head_dim), dtype=dtype)

        # Current length
        self.current_length = 0

    def append(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Append new KV pairs to cache.

        Args:
            keys: (batch, num_tokens, num_kv_heads, head_dim)
            values: (batch, num_tokens, num_kv_heads, head_dim)

        Returns:
            Tuple of (key_cache, value_cache) for the appended slice
        """
        batch, num_tokens, num_kv_heads, head_dim = keys.shape

        # Check dimensions
        assert (
            num_kv_heads == self.num_kv_heads
        ), f"Expected {self.num_kv_heads} KV heads, got {num_kv_heads}"
        assert head_dim == self.head_dim, f"Expected {self.head_dim} head dim, got {head_dim}"

        # Check if we need to expand
        if self.current_length + num_tokens > self.max_seq_len:
            # Dynamic expansion (simplified implementation)
            new_max_len = self.max_seq_len * 2
            new_key_cache = mx.zeros(
                (1, new_max_len, self.num_kv_heads, self.head_dim), dtype=self.dtype
            )
            new_value_cache = mx.zeros(
                (1, new_max_len, self.num_kv_heads, self.head_dim), dtype=self.dtype
            )

            # Copy old data
            new_key_cache[:, : self.current_length, :, :] = self.key_cache[
                :, : self.current_length, :, :
            ]
            new_value_cache[:, : self.current_length, :, :] = self.value_cache[
                :, : self.current_length, :, :
            ]

            self.key_cache = new_key_cache
            self.value_cache = new_value_cache
            self.max_seq_len = new_max_len

        # Append new KV
        start_idx = self.current_length
        end_idx = self.current_length + num_tokens

        self.key_cache[:, start_idx:end_idx, :, :] = keys
        self.value_cache[:, start_idx:end_idx, :, :] = values

        self.current_length += num_tokens

        # Return the appended slice for compatibility
        return (
            self.key_cache[:, start_idx:end_idx, :, :],
            self.value_cache[:, start_idx:end_idx, :, :],
        )

    def rollback(self, target_length: int):
        """
        Rollback cache to a specific length.
        Used when draft tokens are rejected.

        Args:
            target_length: Target length to rollback to
        """
        if target_length < 0:
            raise ValueError(f"Target length must be >= 0, got {target_length}")
        if target_length > self.current_length:
            # Nothing to rollback
            return

        self.current_length = target_length

    def get_cache(self) -> Tuple[mx.array, mx.array]:
        """
        Return current KV cache (only valid portion).

        Returns:
            Tuple of (key_cache, value_cache) in transposed format for SDPA:
            key_cache: (1, n_kv_heads, current_length, head_dim)
            value_cache: (1, n_kv_heads, current_length, head_dim)
        """
        return (
            self.key_cache[:, : self.current_length, :, :].transpose(
                0, 2, 1, 3
            ),  # (1, n_kv_heads, seq_len, head_dim)
            self.value_cache[:, : self.current_length, :, :].transpose(
                0, 2, 1, 3
            ),  # (1, n_kv_heads, seq_len, head_dim)
        )

    def get_length(self) -> int:
        """Return current cache length."""
        return self.current_length

    def get_slice(self, start: int, end: int) -> Tuple[mx.array, mx.array]:
        """
        Get a slice of the cache.

        Args:
            start: Start index
            end: End index (exclusive)

        Returns:
            Tuple of (key_cache_slice, value_cache_slice)
        """
        if start < 0 or end > self.current_length or start > end:
            raise ValueError(
                f"Invalid slice range: [{start}, {end}), current_length={self.current_length}"
            )

        return (
            self.key_cache[:, start:end, :, :],
            self.value_cache[:, start:end, :, :],
        )

    def clear(self):
        """Clear all cached data."""
        self.current_length = 0

    def __repr__(self) -> str:
        return (
            f"ContinuousKVCache(max_seq_len={self.max_seq_len}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"current_length={self.current_length})"
        )
