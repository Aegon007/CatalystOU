"""Utilities for CatalystOU experiments.

Keep this package initializer lightweight. Data loading and metric code should
not require optional LLM dependencies such as the OpenAI SDK.
"""

from .logger import setup_logger
from .data_utils import (
    ResearcherProfile,
    CollaborationProfile,
    ProfileLoader,
    PROFILE_LIST_FIELDS,
    PROFILE_SUMMARY_FIELD,
    PROFILE_FIELDS,
    COLLABORATION_CATEGORIES,
    COLLABORATION_SUMMARY_FIELD,
    COLLABORATION_FIELDS,
)

__all__ = [
    "setup_logger",
    "ResearcherProfile",
    "CollaborationProfile",
    "ProfileLoader",
    "PROFILE_LIST_FIELDS",
    "PROFILE_SUMMARY_FIELD",
    "PROFILE_FIELDS",
    "COLLABORATION_CATEGORIES",
    "COLLABORATION_SUMMARY_FIELD",
    "COLLABORATION_FIELDS",
    "BaseLLMProvider",
    "OpenAIProvider",
    "create_llm_provider",
    "register_llm_provider",
]


def __getattr__(name):
    """Lazily expose LLM provider classes only when explicitly requested."""
    if name in {"BaseLLMProvider", "OpenAIProvider", "create_llm_provider", "register_llm_provider"}:
        from .llm import (
            BaseLLMProvider,
            OpenAIProvider,
            create_llm_provider,
            register_llm_provider,
        )

        values = {
            "BaseLLMProvider": BaseLLMProvider,
            "OpenAIProvider": OpenAIProvider,
            "create_llm_provider": create_llm_provider,
            "register_llm_provider": register_llm_provider,
        }
        return values[name]
    raise AttributeError(f"module 'utils' has no attribute {name!r}")
