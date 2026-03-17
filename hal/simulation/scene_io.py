"""Utilities for reading and writing ``ENVIRONMENT.md`` JSON payloads.

Supported payload formats
-------------------------
1. Legacy object-only dict (v0)
2. Structured envelope (v1) with an ``objects`` section:

     {
         "schema_version": "oea.environment.v1",
         "scene_graph": {...},
         "robots": {...},
         "objects": {...}
     }

``load_scene_from_md`` remains backward-compatible and always returns an
object dict suitable for current HAL drivers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ── Markdown fences kept as constants (avoids backtick confusion in f-strings) ──
_FENCE_OPEN = "```json"
_FENCE_CLOSE = "```"

_BLOCK_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


def _load_json_block(path: Path) -> dict:
    """Parse and return the JSON block from a Markdown file."""
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8")
    match = _BLOCK_RE.search(content)
    if not match:
        return {}
    try:
        data = json.loads(match.group(1))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def load_environment_doc(path: Path) -> dict:
    """Return the full environment document dict from ``ENVIRONMENT.md``."""
    return _load_json_block(path)


def _extract_objects(doc: dict) -> dict[str, dict]:
    """Extract HAL object scene from either v0 or v1 environment docs."""
    if not isinstance(doc, dict):
        return {}

    # v1: structured envelope
    objects = doc.get("objects")
    if isinstance(objects, dict):
        return objects

    # v0: flat object dict
    reserved = {"schema_version", "updated_at", "scene_graph", "robots", "map", "tf", "nav_state"}
    if any(k in doc for k in reserved):
        # Looks like a structured schema but missing/invalid objects section
        return {}
    return doc


def load_scene_from_md(path: Path) -> dict[str, dict]:
    """Return the scene dict parsed from *path* (ENVIRONMENT.md).

    Returns an empty dict if the file does not exist or contains no
    valid JSON code block.
    """
    return _extract_objects(load_environment_doc(path))


def save_environment_doc(path: Path, environment: dict) -> None:
    """Write a full environment document dict to ``ENVIRONMENT.md``."""
    env_json = json.dumps(environment, indent=2, ensure_ascii=False)
    content = (
        "# Environment State\n\n"
        "Auto-updated by HAL Watchdog after each action execution.\n"
        "Edit the JSON block below to set up or reset the test scene.\n\n"
        f"{_FENCE_OPEN}\n{env_json}\n{_FENCE_CLOSE}\n"
    )
    path.write_text(content, encoding="utf-8")


def save_scene_to_md(path: Path, scene: dict[str, dict]) -> None:
    """Write *scene* to *path* (ENVIRONMENT.md) as a JSON code block.

    Preserves the human-readable header so that the LLM agent can still
    understand the file's purpose.
    """
    save_environment_doc(
        path,
        {
            "schema_version": "oea.environment.v1",
            "scene_graph": {"nodes": [], "edges": []},
            "robots": {},
            "objects": scene,
        },
    )
