# profile_io.py
import json


def load_profile_json(path: str) -> dict:
    """
    Load a JSON profile from file path.
    Expecting a dict with string keys and list-of-string values.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(obj: dict, path: str):
    """
    Save JSON object to file path with pretty formatting.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
