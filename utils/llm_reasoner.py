"""Backward-compatible exports for LLM collaboration reasoning."""

from utils.data import COLLABORATION_CATEGORIES
from proj_test.llm_reasoner import (
    build_collaboration_prompt,
    call_llm_reasoner,
    get_default_llm_config,
)

# Export for backward compatibility
__all__ = [
    "COLLAB_CATEGORIES",
    "build_collaboration_prompt",
    "call_llm_reasoner",
    "get_default_llm_config",
]

# Alias for compatibility
COLLAB_CATEGORIES = COLLABORATION_CATEGORIES
