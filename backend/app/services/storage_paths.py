"""Helpers for resolving persistent storage paths."""

import os
from pathlib import Path


def get_data_root_dir() -> str:
    """Return the configured data root directory."""
    configured_root = (os.getenv("AGENTIC_DATA_DIR") or "").strip()
    if configured_root:
        return configured_root

    return "data"


def resolve_data_path(*parts: str) -> str:
    """Resolve a path relative to the configured data root."""
    root_path = Path(get_data_root_dir())
    return str(root_path.joinpath(*parts))
