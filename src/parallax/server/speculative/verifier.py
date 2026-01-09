"""
Verification logic for speculative decoding.
"""

from typing import List, Tuple

import mlx.core as mx

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


def verify_draft_tokens(hidden_states: mx.array, draft_tokens: List[int]) -> Tuple[int, int]:
    """
    Verify K draft tokens and generate 1 bonus token.

    This is the core verification algorithm for speculative decoding.
    It checks each draft token sequentially against the target model's prediction.

    Args:
        hidden_states: (K+1, vocab_size) - Target model logits
            - hidden_states[0:K]: Logits for verifying K draft tokens
            - hidden_states[K]: Logit for generating bonus token
        draft_tokens: List of K draft token IDs from draft model

    Returns:
        Tuple of (num_accepted, bonus_token):
            - num_accepted: Number of draft tokens accepted (0 to K)
            - bonus_token: Newly sampled token at position K

    Example:
        >>> draft_tokens = [10, 20, 30, 40]
        >>> hidden_states = model.forward([...])  # (5, vocab_size)
        >>> num_accepted, bonus_token = verify_draft_tokens(hidden_states, draft_tokens)
        >>> print(f"Accepted {num_accepted}/{len(draft_tokens)} tokens")
        >>> print(f"Bonus token: {bonus_token}")
    """
    K = len(draft_tokens)

    if hidden_states.shape[0] != K + 1:
        raise ValueError(
            f"Expected hidden_states shape ({K+1}, vocab_size), " f"got {hidden_states.shape}"
        )

    num_accepted = 0

    # Verify each draft token sequentially
    for k in range(K):
        # Get target model's prediction at position k
        target_logit = hidden_states[k]  # (vocab_size,)

        # Get the token with maximum probability
        target_token = int(mx.argmax(target_logit))

        if target_token == draft_tokens[k]:
            # Accept: Draft model's prediction matches target model
            num_accepted += 1
        else:
            # Reject: Mismatch found, reject this and all remaining drafts
            logger.debug(
                f"Mismatch at position {k}: " f"draft={draft_tokens[k]}, target={target_token}"
            )
            break

    # Generate bonus token at position num_accepted
    # If all tokens accepted: bonus at position K
    # If some rejected: bonus at position where we first disagreed
    bonus_logit = hidden_states[num_accepted]  # (vocab_size,)
    bonus_token = int(mx.argmax(bonus_logit))

    logger.debug(
        f"Verification result: {num_accepted}/{K} draft tokens accepted, "
        f"bonus_token={bonus_token}"
    )

    return num_accepted, bonus_token


def verify_draft_tokens_batch(
    hidden_states_batch: mx.array, draft_tokens_batch: List[List[int]]
) -> List[Tuple[int, int]]:
    """
    Verify draft tokens for a batch of requests.

    Args:
        hidden_states_batch: (batch, K+1, vocab_size) - Logits for each request
        draft_tokens_batch: List of draft token lists for each request

    Returns:
        List of (num_accepted, bonus_token) tuples for each request
    """
    if len(hidden_states_batch) != len(draft_tokens_batch):
        raise ValueError(
            f"Batch size mismatch: " f"{len(hidden_states_batch)} vs {len(draft_tokens_batch)}"
        )

    results = []

    for i in range(len(hidden_states_batch)):
        hidden_states = hidden_states_batch[i]
        draft_tokens = draft_tokens_batch[i]

        num_accepted, bonus_token = verify_draft_tokens(hidden_states, draft_tokens)
        results.append((num_accepted, bonus_token))

    return results


def calculate_acceptance_rate(num_accepted: int, num_draft_tokens: int) -> float:
    """
    Calculate the acceptance rate for speculative decoding.

    Args:
        num_accepted: Number of accepted draft tokens
        num_draft_tokens: Total number of draft tokens

    Returns:
        Acceptance rate (0.0 to 1.0)
    """
    if num_draft_tokens == 0:
        return 0.0

    return num_accepted / num_draft_tokens
