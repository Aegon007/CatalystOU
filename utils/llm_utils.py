"""Simple shared LLM helpers for CatalystOU experiments.

This module keeps the research code straightforward by providing a single,
lightweight entry point for creating providers and building common prompt-based
reasoning calls.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .data_utils import COLLABORATION_CATEGORIES
from .logger import setup_logger

logger = setup_logger(__name__, log_file="llm_utils.log")


def get_default_llm_config(model_name: str = "gpt-4", provider: str = "openai") -> Dict[str, Any]:
    """Return a simple default configuration for experiment runs."""
    temperature_support = model_name not in {"gpt-5-nano"}
    api_mode = "responses" if model_name.startswith("gpt-5") else "chat"
    return {
        "provider": provider,
        "model_name": model_name,
        "temperature": 0.3,
        "max_tokens": 4096,
        "temperature_support": temperature_support,
        "max_retries": 3,
        "timeout": 600.0,
        "api_mode": api_mode,
    }


def ensure_collaboration_schema(output: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure the collaboration output contains the expected schema fields."""
    for category in COLLABORATION_CATEGORIES:
        if category not in output:
            output[category] = []
        if not isinstance(output[category], list):
            output[category] = [str(output[category])]

    if "Summary Collaboration Themes" not in output:
        output["Summary Collaboration Themes"] = ""

    return output


def create_provider(provider_type: str, config: Dict[str, Any]):
    """Create an LLM provider using the shared package interface if available."""
    raise RuntimeError("LLM provider dependencies are not available")
