"""Utilities for CatalystOU experiments.

Keep this package initializer lightweight. Data loading and metric code should
not require optional LLM dependencies such as the OpenAI SDK.
"""

from .logger_utils import setup_logger
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
]
