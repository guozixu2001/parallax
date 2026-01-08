"""
Speculative decoding module for Parallax.

This module contains components for speculative decoding including:
- DraftGenerator: Generate draft tokens using draft model
- Verifier: Verify draft tokens using target model
"""

from parallax.server.speculative.draft_generator import DraftGenerator
from parallax.server.speculative.verifier import verify_draft_tokens

__all__ = [
    "DraftGenerator",
    "verify_draft_tokens",
]
