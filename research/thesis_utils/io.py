"""Standard artifact I/O helpers (json + figure + handoff state)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import ARTIFACTS_DIR, FIGURES_DIR, SHARED_STATE_DIR


def _ensure_suffix(name: str, suffix: str) -> str:
    return name if name.endswith(suffix) else f"{name}{suffix}"


def save_json(name: str, payload: dict[str, Any], subdir: Path = ARTIFACTS_DIR) -> Path:
    """JSON으로 저장한다. ``name``에 확장자 포함 안 해도 됨."""
    subdir.mkdir(parents=True, exist_ok=True)
    path = subdir / _ensure_suffix(name, ".json")
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path


def load_json(name: str, subdir: Path = ARTIFACTS_DIR) -> dict[str, Any]:
    """JSON 산출물을 로드한다."""
    path = subdir / _ensure_suffix(name, ".json")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_figure(fig: Any, name: str, subdir: Path = FIGURES_DIR, dpi: int = 150) -> Path:
    """matplotlib Figure를 PNG로 저장한다."""
    subdir.mkdir(parents=True, exist_ok=True)
    path = subdir / _ensure_suffix(name, ".png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_state(name: str, payload: dict[str, Any]) -> Path:
    """노트북 간 핸드오프용. SHARED_STATE_DIR에 저장."""
    return save_json(name, payload, subdir=SHARED_STATE_DIR)


def load_state(name: str) -> dict[str, Any]:
    """이전 노트북이 남긴 핸드오프 state를 로드."""
    return load_json(name, subdir=SHARED_STATE_DIR)


__all__ = [
    "save_json",
    "load_json",
    "save_figure",
    "save_state",
    "load_state",
]
