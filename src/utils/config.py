import copy
import json
import os
from typing import Any, Dict, List

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at '{path}' must be a mapping object.")
    return data


def _parse_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none" or lowered == "null":
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def parse_overrides(overrides: List[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"Invalid override '{item}'. Expected format key.subkey=value."
            )
        key, value = item.split("=", 1)
        cursor = parsed
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _parse_scalar(value)
    return parsed


def deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    output = copy.deepcopy(base)
    for key, value in updates.items():
        if (
            key in output
            and isinstance(output[key], dict)
            and isinstance(value, dict)
        ):
            output[key] = deep_merge(output[key], value)
        else:
            output[key] = value
    return output


def save_resolved_config(config: Dict[str, Any], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    target = os.path.join(output_dir, "resolved_config.json")
    with open(target, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
    return target
