"""
Small LLM helper layer for CatalystOU research scripts.

Model settings live in utils/llm_config.json. Scripts pass a model name; this
module loads the config, creates an OpenAI-compatible client, sends the request,
and returns text or JSON.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Optional
from openai import AsyncOpenAI
from .data_utils import COLLABORATION_CATEGORIES
from .logger_utils import setup_logger

logger = setup_logger(__name__, log_file="llm_utils.log")
LLM_CONFIG_PATH = Path(__file__).resolve().parent / "llm_config.json"


def load_llm_configs() -> Dict[str, Any]:
    """Load utils/llm_config.json."""
    if not LLM_CONFIG_PATH.exists():
        return {}

    try:
        with LLM_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            configs = json.load(handle)
        if isinstance(configs, dict):
            return configs
    except Exception:
        pass

    return {}


def _env_value(name: Any) -> Optional[str]:
    return os.getenv(str(name))


def get_llm_config(model_name: Optional[str] = None) -> Dict[str, Any]:
    """Return the resolved config for a model with automatic provider routing."""
    if not model_name:
        model_name = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
    requested_model = model_name.strip()
    configs = load_llm_configs()
    config_key = requested_model.lower()

    if config_key in configs:
        model_config = dict(configs[config_key])
    else:
        alt_key = config_key.split("/")[-1]
        if alt_key in configs:
            model_config = dict(configs[alt_key])
        else:
            model_config = dict(configs.get("default", {}))

    model_config["model_name"] = model_config.get("model_name") or requested_model

    # Resolve API Key
    api_key = (
        model_config.get("api_key")
        or _env_value("LLM_API_KEY")
        or _env_value("OPENROUTER_API_KEY")
        or _env_value("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError(
            f"No API key found for '{requested_model}'. Please set LLM_API_KEY in your .env file."
        )
    model_config["api_key"] = api_key

    # Resolve Base URL
    if not model_config.get("base_url"):
        env_base = (
            _env_value("LLM_API_URL")
            or _env_value("OPENAI_BASE_URL")
            or _env_value("OPENROUTER_BASE_URL")
        )
        if env_base:
            model_config["base_url"] = env_base
        elif str(api_key).startswith("sk-or-"):
            model_config["base_url"] = "https://openrouter.ai/api/v1"
        elif str(api_key).startswith("AIza"):
            model_config["base_url"] = "https://generativelanguage.googleapis.com/v1beta/openai/"

    # Normalize OpenRouter URL if needed
    if model_config.get("base_url") and "openrouter.ai" in model_config["base_url"]:
        if not model_config["base_url"].rstrip("/").endswith("/api/v1"):
            model_config["base_url"] = "https://openrouter.ai/api/v1"

    # Resolve Temperature and Seed (defaulting to 0.0 temp and seed 42 for deterministic evaluation)
    if _env_value("LLM_TEMPERATURE") is not None:
        try:
            model_config["temperature"] = float(_env_value("LLM_TEMPERATURE"))
        except ValueError:
            pass
    elif "temperature" not in model_config:
        model_config["temperature"] = 0.0

    if _env_value("LLM_SEED") is not None:
        try:
            model_config["seed"] = int(_env_value("LLM_SEED"))
        except ValueError:
            pass
    elif "seed" not in model_config:
        model_config["seed"] = 42

    if "api_mode" not in model_config:
        model_config["api_mode"] = "chat"

    return model_config


def create_async_openai_client(model_name: str | Dict[str, Any] | None = None) -> AsyncOpenAI:
    """Create an AsyncOpenAI client from a model name or resolved config."""
    if isinstance(model_name, dict):
        config = dict(model_name)
    else:
        config = get_llm_config(model_name)

    client_args: Dict[str, Any] = {
        "api_key": config.get("api_key") or "unused",
        "timeout": config.get("timeout", 600.0),
        "max_retries": config.get("max_retries", 3),
    }
    if config.get("base_url"):
        client_args["base_url"] = config["base_url"]
        if "openrouter.ai" in config["base_url"]:
            client_args["default_headers"] = {
                "HTTP-Referer": "https://catalystou.local",
                "X-Title": "CatalystOU",
            }

    return AsyncOpenAI(**client_args)


def build_completion_payload(
    *,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    config: Optional[Dict[str, Any]] = None,
    max_output_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the standard OpenAI-compatible request payload."""
    config = config or get_llm_config(model_name)
    token_limit = (
        max_output_tokens
        or config.get("max_tokens")
        or config.get("max_output_tokens")
        or 4096
    )

    payload: Dict[str, Any] = {
        "model": config.get("model_name", model_name),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    if config.get("temperature") is not None:
        payload["temperature"] = config["temperature"]

    if config.get("seed") is not None:
        payload["seed"] = config["seed"]

    # Google AI Studio and standard non-reasoning models require 'max_tokens'
    model_lower = str(payload["model"]).lower()
    if "o1" in model_lower or "o3" in model_lower:
        payload["max_completion_tokens"] = token_limit
    else:
        payload["max_tokens"] = token_limit

    return payload


def extract_response_text(response: Any) -> str:
    """Extract assistant text from chat/completions output."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "") if message else ""
    if isinstance(content, list):
        text = "\n".join(
            str(item.get("text") or item.get("content") or "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
        if text:
            return text
    elif content and str(content).strip():
        return str(content).strip()

    # Fallback to reasoning / thinking output if content is empty
    reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
    if reasoning:
        return str(reasoning).strip()
    return ""


async def call_llm(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    max_attempts: int = 5,
) -> str:
    """Call the configured LLM with retry logic."""
    config = config or get_llm_config(model_name)
    client = create_async_openai_client(config)
    payload = build_completion_payload(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config=config,
        max_output_tokens=max_output_tokens,
    )

    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.chat.completions.create(**payload)
            return extract_response_text(response)
        except Exception as e:
            err_str = str(e).lower()
            is_transient = (
                "429" in err_str
                or "rate_limit" in err_str
                or "quota" in err_str
                or "503" in err_str
                or "502" in err_str
            )
            if is_transient and attempt < max_attempts:
                sleep_time = (2 ** attempt) + random.uniform(1.0, 3.0)
                logger.warning(f"Retryable error on attempt {attempt}/{max_attempts}: {e}. Retrying in {sleep_time:.2f}s...")
                await asyncio.sleep(sleep_time)
            else:
                logger.error(f"LLM call failed definitively on model '{model_name}': {e}")
                raise e


def parse_json_text(text: str) -> Dict[str, Any]:
    """Parse a JSON object, sanitizing backslashes and markdown fences."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    try:
        data = json.loads(cleaned, strict=False)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Repair invalid escape backslashes (e.g. \alpha, \mu, \d)
    repaired = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", cleaned)

    try:
        data = json.loads(repaired, strict=False)
        if isinstance(data, dict):
            return data
        raise ValueError("Parsed JSON is not a dictionary.")
    except json.JSONDecodeError as err:
        raise ValueError(f"Failed to parse LLM JSON response: {err}\nSnippet: {cleaned[:300]}")


async def call_llm_json(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    max_attempts: int = 5,
) -> Dict[str, Any]:
    """Call LLM and parse response as a JSON dictionary."""
    raw_text = await call_llm(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_output_tokens=max_output_tokens,
        config=config,
        max_attempts=max_attempts,
    )
    return parse_json_text(raw_text)


def ensure_collaboration_schema(output: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing collaboration fields with default lists."""
    for category in COLLABORATION_CATEGORIES:
        if category not in output:
            output[category] = []
        if not isinstance(output[category], list):
            output[category] = [str(output[category])]

    output.setdefault("Summary Collaboration Themes", "")
    return output