"""
Simple shared LLM helpers for CatalystOU experiments.

This module keeps the research code straightforward by providing a single,
lightweight entry point for creating providers and building common prompt-based
reasoning calls.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .data_utils import COLLABORATION_CATEGORIES
from .logger_utils import setup_logger

logger = setup_logger(__name__, log_file="llm_utils.log")


def _load_model_config_file() -> Dict[str, Any]:
    """Load model-specific LLM settings from the utils JSON config file."""
    config_path = Path(__file__).resolve().parent / "llm_config.json"
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


MODEL_CONFIGS = _load_model_config_file()


def get_model_capabilities(model_name: str) -> Dict[str, Any]:
    """Return capability hints for a model name from the JSON config table."""
    normalized = (model_name or "").lower()
    config = MODEL_CONFIGS.get(normalized)
    if config:
        return {
            "supports_temperature": bool(config.get("temperature") is not None),
            "supports_tools": bool(config.get("supports_tools", False)),
            "api_mode": config.get("api_mode", "chat"),
        }

    if normalized.startswith("gpt-5") or normalized.startswith("gpt-4") or normalized.startswith("o1") or normalized.startswith("o3"):
        return {
            "supports_temperature": False if normalized.startswith("gpt-5") else True,
            "supports_tools": True,
            "api_mode": "responses",
        }
    if any(token in normalized for token in ["qwen", "phi", "gemma", "llama", "mistral"]):
        return {
            "supports_temperature": True,
            "supports_tools": False,
            "api_mode": "chat",
        }
    return {
        "supports_temperature": True,
        "supports_tools": False,
        "api_mode": "chat",
    }


def get_default_llm_config(
    model_name: str = "gpt-4",
    provider: str = "openai",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a simple default configuration for experiment runs."""
    provider_name = (provider or "openai").strip().lower()
    if provider_name in {"lmstudio", "local", "openai-compatible", "openai_compatible"}:
        provider_name = "lmstudio"

    normalized_model = (model_name or "").strip().lower()
    config = MODEL_CONFIGS.get(normalized_model, MODEL_CONFIGS.get("default", {}))
    if config:
        provider_name = config.get("provider", provider_name)

    capabilities = get_model_capabilities(model_name)
    resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if provider_name == "lmstudio" and not resolved_base_url:
        resolved_base_url = None

    resolved_api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if provider_name == "lmstudio" and not resolved_api_key:
        resolved_api_key = "lm-studio"

    temperature = config.get("temperature", 0.3)
    max_tokens = config.get("max_tokens", 4096)
    max_retries = config.get("max_retries", 3)
    timeout = config.get("timeout", 600.0)
    api_mode = config.get("api_mode", capabilities["api_mode"])

    return {
        "provider": provider_name,
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "temperature_support": capabilities["supports_temperature"],
        "supports_tools": capabilities["supports_tools"],
        "max_retries": max_retries,
        "timeout": timeout,
        "api_mode": api_mode,
        "base_url": resolved_base_url,
        "api_key": resolved_api_key,
    }


def create_async_openai_client(config: Optional[Dict[str, Any]] = None):
    """Create an AsyncOpenAI-compatible client for remote OpenAI or local OpenAI-compatible servers."""
    try:
        from openai import AsyncOpenAI
    except Exception as exc:  # pragma: no cover - optional dependency fallback
        raise RuntimeError("The openai package is required for LLM calls") from exc

    resolved = config or {}
    client_kwargs: Dict[str, Any] = {
        "api_key": resolved.get("api_key") or "unused",
        "timeout": resolved.get("timeout", 600.0),
    }
    if resolved.get("base_url"):
        client_kwargs["base_url"] = resolved["base_url"]
    return AsyncOpenAI(**client_kwargs)


def build_completion_payload(
    *,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    config: Optional[Dict[str, Any]] = None,
    max_output_tokens: int = 10240,
) -> Dict[str, Any]:
    """Build a request payload compatible with both OpenAI Responses and chat/completions APIs."""
    resolved = config or get_default_llm_config(model_name=model_name)
    api_mode = resolved.get("api_mode", "chat")
    payload: Dict[str, Any] = {
        "model": model_name,
        "max_tokens": max_output_tokens,
    }
    if resolved.get("temperature_support") and resolved.get("temperature") is not None:
        payload["temperature"] = resolved.get("temperature", 0.3)

    if api_mode == "responses":
        payload["input"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload["max_output_tokens"] = max_output_tokens
        return payload

    payload["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return payload


def extract_response_text(response: Any, api_mode: Optional[str] = None) -> str:
    """Extract text from either OpenAI Responses API or chat/completions responses."""
    mode = api_mode or "chat"
    if mode == "responses":
        texts: list[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for block in getattr(item, "content", []) or []:
                    if getattr(block, "type", None) == "output_text":
                        texts.append(getattr(block, "text", ""))
        return "\n".join(texts).strip()

    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                texts.append(str(item.get("text", "")))
            else:
                texts.append(str(item))
        return "\n".join(texts).strip()
    return str(content).strip()


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
