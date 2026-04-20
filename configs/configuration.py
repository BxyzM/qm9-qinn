"""
Loads default parameters from defaults.py, merges a user-supplied YAML file
on top, and exposes the result as a dot-notation namespace object.

Author: Dr. Aritra Bal (ETP)
Date: March 03, 2026
"""

import copy
import pathlib
from types import SimpleNamespace
from typing import Any

import yaml
from loguru import logger

from configs.defaults import DEFAULTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base; override wins on conflicts."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _dict_to_namespace(d: Any) -> Any:
    """Recursively convert nested dicts to SimpleNamespace objects."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_namespace(i) for i in d]
    return d


def _namespace_to_dict(obj: Any) -> Any:
    """Recursively convert SimpleNamespace objects back to nested dicts."""
    if isinstance(obj, SimpleNamespace):
        return {k: _namespace_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_namespace_to_dict(i) for i in obj]
    return obj


# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------

class Config:
    """
    Configuration object built from defaults + YAML overrides.

    Attributes are accessible via dot notation: config.setup.train, etc.
    Any top-level key present in defaults.py or the YAML file is exposed
    directly as an attribute of this object.
    """

    def __init__(self, yaml_path: str) -> None:
        """
        Args:
            yaml_path: Path to user YAML file. Keys present override defaults;
                       missing keys retain their default values.
        """
        cfg = copy.deepcopy(DEFAULTS)

        with open(yaml_path) as f:
            yaml_cfg = yaml.safe_load(f) or {}

        cfg = _deep_merge(cfg, yaml_cfg)
        self._ns = _dict_to_namespace(cfg)

        # Expose all top-level groups as direct attributes.
        for k, v in vars(self._ns).items():
            setattr(self, k, v)

        logger.info(f"Config loaded from '{yaml_path}' | run_id={self.setup.run_id}")

    def save(self, path: str = None) -> None:
        """
        Save the full resolved configuration as a YAML file.

        Args:
            path: Destination path. Defaults to <model_dir>/config.yaml.
        """
        if path is None:
            path = str(pathlib.Path(self.paths.model_dir) / "config.yaml")

        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(_namespace_to_dict(self._ns), f, default_flow_style=False, sort_keys=False)

        logger.info(f"Config saved to '{path}'")
