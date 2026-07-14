"""
Small LLM helper layer for CatalystOU research scripts.

Model settings live in utils/llm_config.json. Scripts pass a model name; this
module loads the config, creates an OpenAI-compatible client, sends the request,
and returns text or JSON.
"""

from __future__ import annotations

import json
import os
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
        raise FileNotFoundError(f"LLM config file not found: {LLM_CONFIG_PATH}")

    with LLM_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        configs = json.load(handle)

    if not isinstance(configs, dict):
        raise ValueError("llm_config.json must contain a JSON object")
    
    return configs


def _env_value(name: Any) -> Optional[str]:
    env_value = os.getenv(str(name))
    return env_value


def get_llm_config(model_name: str = "gpt-5-nano") -> Dict[str, Any]:
    """Return the resolved config for a model listed in llm_config.json."""
    requested_model = model_name.strip()

    configs = load_llm_configs()
    config_key = requested_model.lower()
    if config_key not in configs.keys():
        available = ", ".join(sorted(key for key in configs if key != "default"))
        logger.warning(f"Unknown LLM model '{model_name}'. Available models: {available}")
        model_config = dict(configs.get("default", {}))
    else:
        model_config = dict(configs[config_key])

    provider = str(model_config.get("provider", "openai")).strip().lower()

    api_key = model_config.get("api_key")
    if not api_key:
        if provider == "lmstudio":
            api_key = _env_value("LLM_API_KEY") or _env_value("OPENAI_API_KEY") or "lm-studio"
        elif provider == "openai":
            api_key = _env_value("LLM_API_KEY") or _env_value("OPENAI_API_KEY")

    if provider == "lmstudio" and not api_key:
        raise ValueError(f"No API key found for {config_key}.")
    if provider == "openai" and not api_key:
        raise ValueError(f"No API key found for {config_key}.")

    model_config["api_key"] = api_key

    if not model_config.get("base_url"):
        for env_name in model_config.get("base_url_env", []) or []:
            env_value = _env_value(env_name)
            if env_value:
                model_config["base_url"] = env_value
                break

    return model_config


def create_async_openai_client(model_name: str | Dict[str, Any] = "gpt-5-nano"):
    """Create an AsyncOpenAI client from a model name or resolved config."""

    if isinstance(model_name, dict):
        config = dict(model_name)
    else:
        config = get_llm_config(model_name)

    client_args = {
        "api_key": config.get("api_key") or "unused",
        "timeout": config.get("timeout", 600.0),
        "max_retries": config.get("max_retries", 3),
    }
    if config.get("base_url"):
        client_args["base_url"] = config["base_url"]
    return AsyncOpenAI(**client_args)


def create_llm_connection(model_name: str = "gpt-5-nano"):
    """Return (client, config) for scripts that need the raw client."""
    if isinstance(model_name, dict):
        config = dict(model_name)
    else:
        config = get_llm_config(model_name)
    return create_async_openai_client(config), config


def build_completion_payload(*, model_name: str, system_prompt: str, user_prompt: str, config: Optional[Dict[str, Any]] = None, max_output_tokens: Optional[int] = None) -> Dict[str, Any]:
    """
        Build the OpenAI-compatible request payload for a prompt.
        in a function definition, a bare * means all parameters after it must be passed by name.
    """
    config = config or get_llm_config(model_name)
    token_limit = max_output_tokens or config.get("max_tokens", 4096)
    payload: Dict[str, Any] = {"model": config.get("model_name", model_name)}

    if config.get("temperature") is not None:
        payload["temperature"] = config["temperature"]

    if config.get("api_mode") == "responses":
        payload["input"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload["max_output_tokens"] = token_limit
    else:
        payload["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload["max_tokens"] = token_limit

    return payload


def extract_response_text(response: Any, api_mode: str = "chat") -> str:
    """Extract assistant text from Responses API or chat/completions output."""
    if api_mode == "responses":
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text).strip()

        texts: list[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for block in getattr(item, "content", []) or []:
                    if getattr(block, "type", None) in {"output_text", "text"}:
                        texts.append(getattr(block, "text", ""))
        return "\n".join(texts).strip()

    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "") if message else ""
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or item.get("content") or "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content).strip()


async def call_llm(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Call the configured LLM and return response text."""
    config = config or get_llm_config(model_name)
    client = create_async_openai_client(config)
    payload = build_completion_payload(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config=config,
        max_output_tokens=max_output_tokens,
    )

    if config["api_mode"] == "responses":
        response = await client.responses.create(**payload)
    else:
        response = await client.chat.completions.create(**payload)
    return extract_response_text(response, api_mode=config["api_mode"])


def parse_json_text(text: str) -> Dict[str, Any]:
    """Parse a JSON object, allowing simple markdown code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("LLM response JSON must be an object")
    return data


async def call_llm_json(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call the configured LLM and parse the response as a JSON object."""
    return parse_json_text(
        await call_llm(
            model_name,
            system_prompt,
            user_prompt,
            max_output_tokens=max_output_tokens,
            config=config,
        )
    )


def ensure_collaboration_schema(output: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing collaboration fields with empty values."""
    for category in COLLABORATION_CATEGORIES:
        if category not in output:
            output[category] = []
        if not isinstance(output[category], list):
            output[category] = [str(output[category])]

    output.setdefault("Summary Collaboration Themes", "")
    return output
