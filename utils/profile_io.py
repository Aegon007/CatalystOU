# profile_io.py
"""Profile I/O utilities for experiment JSON files."""

import json
from pathlib import Path
from typing import Any, Dict


def load_profile_json(path: str | Path) -> Dict[str, Any]:
    """
    Load a JSON profile from file path.
    Expecting a dict with string keys and list-of-string values.

    Args:
        path: File path to JSON profile

    Returns:
        Loaded profile dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        JSONDecodeError: If JSON is invalid
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    """
    Save JSON object to file path with pretty formatting.

    Args:
        obj: Dictionary to save
        path: Target file path
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
