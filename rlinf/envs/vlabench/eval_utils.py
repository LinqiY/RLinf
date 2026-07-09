from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def jsonable(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def stable_config_hash(config: Any) -> str:
    payload = json.dumps(jsonable(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def episode_config_id(source: str | None, task_name: str, episode_idx: int, config: Any) -> str:
    source_name = source or "inline"
    digest = stable_config_hash(config)
    return f"{source_name}:{task_name}:{episode_idx}:sha1={digest}"


def ensure_dir_for(path: str | os.PathLike[str]) -> None:
    directory = os.path.dirname(str(path))
    if directory:
        os.makedirs(directory, exist_ok=True)


def write_json(path: str | os.PathLike[str], payload: Any) -> None:
    ensure_dir_for(path)
    with open(path, "w") as f:
        json.dump(jsonable(payload), f, indent=2, ensure_ascii=False)


def append_jsonl(path: str | os.PathLike[str], payload: Any) -> None:
    ensure_dir_for(path)
    with open(path, "a") as f:
        f.write(json.dumps(jsonable(payload), ensure_ascii=False) + "\n")


def default_debug_dir(result_path: str | os.PathLike[str]) -> str:
    return str(Path(result_path).parent / "debug")
