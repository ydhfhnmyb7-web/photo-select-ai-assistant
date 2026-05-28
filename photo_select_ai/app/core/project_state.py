from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT


STATE_PATH = PROJECT_ROOT / "data" / "project_state.json"


@dataclass
class ProjectState:
    project_folder: str = ""
    selected_path: str = ""
    filter_name: str = "全部"
    sort_name: str = "文件名"
    view_mode: str = "单图审片"
    thumbnail_size_label: str = "中"
    thumbnail_mode: str = "contain"
    dirty: bool = False
    clean_shutdown: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def load_project_state() -> ProjectState:
    if not STATE_PATH.exists():
        return ProjectState()
    try:
        raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return ProjectState()
    state = ProjectState()
    for key, value in raw.items():
        if hasattr(state, key):
            setattr(state, key, value)
    return state


def save_project_state(state: ProjectState) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")
