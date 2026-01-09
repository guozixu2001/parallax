"""
Speculative Cache Manager for target model in speculative decoding.
Uses mlx_lm's KVCache (concatenate-based) instead of paged KV cache.
"""

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class SpeculativeCacheManager:
    """
    Cache manager for speculative decoding target model.

    Uses mlx_lm's KVCache (concatenate-based) instead of paged KV cache.
    Manages KV cache at request level with rollback support.
    """

    def __init__(
        self,
        num_layers: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: type = mx.float16,
    ):
        """
        Initialize SpeculativeCacheManager.

        Args:
            num_layers: Number of transformer layers
            max_seq_len: Maximum sequence length (not used by KVCache, kept for compatibility)
            num_kv_heads: Number of key-value heads (not used by KVCache, kept for compatibility)
            head_dim: Dimension of each head (not used by KVCache, kept for compatibility)
            dtype: Data type for cache (not used by KVCache, kept for compatibility)
        """
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        self.request_caches: Dict[str, List[KVCache]] = {}

        # For compatibility with paged cache interface
        self.needs_slots = False

        logger.debug(
            f"SpeculativeCacheManager initialized: "
            f"num_layers={num_layers}, max_seq_len={max_seq_len}, "
            f"num_kv_heads={num_kv_heads}, head_dim={head_dim}"
        )

    def allocate_request(
        self, request_id: str, prompt_len: int, token_ids: Optional[List[int]] = None
    ) -> Tuple[bool, int]:
        """
        Allocate cache for a new request.

        Args:
            request_id: Request ID
            prompt_len: Length of the prompt
            token_ids: Optional token IDs for prefix cache matching (not implemented yet)

        Returns:
            Tuple of (success, matched_tokens)
            - success: True if allocation succeeded
            - matched_tokens: Number of tokens matched from prefix cache (always 0 for now)
        """
        if request_id in self.request_caches:
            # Already allocated
            return True, 0

        # Create KVCache for each layer
        layer_caches = [KVCache() for _ in range(self.num_layers)]

        self.request_caches[request_id] = layer_caches

        logger.debug(
            f"Allocated cache for request {request_id}: "
            f"prompt_len={prompt_len}, num_layers={self.num_layers}"
        )

        return True, 0

    def has_request(self, request_id: str) -> bool:
        """Check if request exists in cache."""
        return request_id in self.request_caches

    def get_layer_cache(
        self, request_id: str, layer_idx: int
    ) -> Optional[Tuple[mx.array, mx.array]]:
        """
        Get KV cache for a specific layer and request.

        Args:
            request_id: Request ID
            layer_idx: Layer index

        Returns:
            Tuple of (key_cache, value_cache) or None if request not found
        """
        if request_id not in self.request_caches:
            return None

        if layer_idx < 0 or layer_idx >= self.num_layers:
            return None

        layer_cache = self.request_caches[request_id][layer_idx]
        # KVCache.state returns (keys, values) tuple
        return layer_cache.state

    def get_caches(self) -> Optional[List[KVCache]]:
        """
        Get caches for the current batch.

        For SpeculativeCacheManager with per-request caches, this returns
        the cache objects of the first allocated request. This is a compatibility
        method for the batch processing API.

        Returns:
            List of KVCache objects for each layer,
            or None if no requests are allocated
        """
        if not self.request_caches:
            logger.debug("get_caches(): No requests allocated yet")
            return None

        # Get the first request's cache objects (not tuples)
        first_request_id = next(iter(self.request_caches))
        caches = self.request_caches.get(first_request_id)
        logger.debug(
            f"get_caches(): Returning {len(caches) if caches else 0} layer caches "
            f"for request {first_request_id}"
        )
        return caches

    def get_context_length(self, request_id: str) -> int:
        """
         Get current context length for a request.

         Args:
        request_id: Request ID

         Returns:
             Current context length, or 0 if request not found
        """
        if request_id not in self.request_caches:
            return 0

        # Get length from the first layer cache (all layers should have same length)
        layer_caches = self.request_caches[request_id]
        if not layer_caches:
            return 0

        # KVCache has __len__ method that returns offset
        return len(layer_caches[0])

    def rollback_to_position(self, request_id: str, target_length: int):
        """
        Rollback KV cache to a specific position.

        Used when draft tokens are rejected in speculative decoding.

        Args:
            request_id: Request ID
            target_length: Target length to rollback to
        """
        if request_id not in self.request_caches:
            logger.warning(f"Attempted to rollback non-existent request {request_id}")
            return

        # Get current length from actual cache
        current_length = self.get_context_length(request_id)

        if target_length >= current_length:
            # Nothing to rollback
            return

        if target_length < 0:
            raise ValueError(f"Target length must be >= 0, got {target_length}")

        # Calculate how many tokens to trim
        num_to_trim = current_length - target_length

        # Rollback each layer cache using trim()
        for layer_idx in range(self.num_layers):
            layer_cache = self.request_caches[request_id][layer_idx]
            layer_cache.trim(num_to_trim)

        logger.debug(f"Rolled back request {request_id} from {current_length} to {target_length}")

    def free_request(self, request_id: str):
        """
        Free all resources for a request.

        Args:
            request_id: Request ID
        """
        if request_id in self.request_caches:
            del self.request_caches[request_id]
            logger.debug(f"Freed cache for request {request_id}")

    def update_request_tokens(self, request_id: str, token_ids: List[int]):
        """
        Update request tokens (for compatibility with prefix cache interface).

        Not implemented in continuous mode.

        Args:
            request_id: Request ID
            token_ids: Token IDs to update
        """
        # Prefix cache not implemented in continuous mode

    def __repr__(self) -> str:
        return (
            f"SpeculativeCacheManager("
            f"num_layers={self.num_layers}, "
            f"max_seq_len={self.max_seq_len}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"num_requests={len(self.request_caches)})"
        )
