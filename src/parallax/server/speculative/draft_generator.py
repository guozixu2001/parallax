"""
Draft token generation for speculative decoding.
"""

from typing import List, Optional

import mlx.core as mx
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class DraftGenerator:
    """
    Generate draft tokens using draft model.

    Used only on first peer for speculative decoding.
    """

    def __init__(
        self,
        draft_model,
        cache_manager,
        max_draft_tokens: int = 3,
    ):
        """
        Initialize DraftGenerator.

        Args:
            draft_model: Draft model instance (small model)
            cache_manager: Cache manager for draft model
            max_draft_tokens: Maximum number of draft tokens to generate
        """
        self.draft_model = draft_model
        self.cache_manager = cache_manager
        self.max_draft_tokens = max_draft_tokens

        logger.debug(
            f"DraftGenerator initialized: max_draft_tokens={max_draft_tokens}"
        )

    def generate_draft_tokens(
        self,
        request_id: str,
        last_token: int,
        num_tokens: Optional[int] = None,
    ) -> List[int]:
        """
        Generate K draft tokens autoregressively.

        Args:
            request_id: Request ID
            last_token: Last generated token
            num_tokens: Number of draft tokens to generate (default: max_draft_tokens)

        Returns:
            List of K draft token IDs
        """
        if num_tokens is None:
            num_tokens = self.max_draft_tokens

        if num_tokens <= 0:
            return []

        draft_ids = []
        current_input = mx.array([[last_token]])

        for step in range(num_tokens):
            # Get KV cache objects for this request (KVCache objects)
            if request_id not in self.cache_manager.request_caches:
                logger.error(f"No cache found for request {request_id}")
                break

            layer_caches = self.cache_manager.request_caches[request_id]

            # Get current context length for RoPE
            current_len = self.cache_manager.get_context_length(request_id)
            context_lengths = mx.array([current_len], dtype=mx.int32)

            # Draft model forward (SDPA only, no paged attention)
            draft_logits = self.draft_model(
                h_or_tokens=current_input,
                cache=layer_caches,  # Pass list of KVCache objects
                mask=None,  # Decode phase doesn't need mask
                block_tables=None,  # Draft model doesn't use paged attention
                context_lengths=context_lengths,  # Needed for RoPE positioning
                slot_mapping=None,
            )

            # draft_logits shape: (1, 1, vocab_size)
            # Sample next token (greedy for now, can be extended to support sampling)
            next_token = int(mx.argmax(draft_logits[0, 0, :]))
            draft_ids.append(next_token)

            logger.debug(
                f"Draft token {step + 1}/{num_tokens}: {next_token} for request {request_id}"
            )

            # Note: Cache is automatically updated by the model's attention layer
            # via cache.append() in qwen3.py line 264, which increments current_length

            # Update input for next iteration
            current_input = mx.array([[next_token]])

        logger.debug(
            f"Generated {len(draft_ids)} draft tokens for request {request_id}: {draft_ids}"
        )

        return draft_ids

    def generate_prefill_draft_tokens(
        self,
        request_id: str,
        input_tokens: List[int],
        num_draft_tokens: Optional[int] = None,
    ) -> List[int]:
        """
        Generate draft tokens from a prefill input.

        This is used when the draft model needs to process a prefill phase
        before generating draft tokens.

        Args:
            request_id: Request ID
            input_tokens: Input token IDs for prefill
            num_draft_tokens: Number of draft tokens to generate after prefill

        Returns:
            List of draft token IDs
        """
        if num_draft_tokens is None:
            num_draft_tokens = self.max_draft_tokens

        # Step 1: Run prefill phase
        # TODO: Implement prefill phase with proper cache management

        # Step 2: Generate draft tokens from the last token
        if input_tokens:
            last_token = input_tokens[-1]
            return self.generate_draft_tokens(request_id, last_token, num_draft_tokens)
        else:
            return []
