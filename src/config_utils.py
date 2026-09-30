"""
Shared utilities for loading configuration files.

This module avoids duplicating the same logic in:
- scripts/train.py
- scripts/evaluate.py
"""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml_config(path):
    """
    Load a YAML files and validate its contents
    top-level object is a mapping/dictionary.
    """

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid YAML configuration: {path}"
        )

    return config


def resolve_project_path(path):
    """
    Convert a relative path to an absolute path
    relative to the repository root.

    Absolute paths are returned unchanged.
    """

    path = Path(path)

    if path.is_absolute():
        return str(path)

    return str(PROJECT_ROOT / path)