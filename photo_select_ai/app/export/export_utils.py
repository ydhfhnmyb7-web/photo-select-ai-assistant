from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


WINDOWS_ILLEGAL_CHARS = '<>:"/\\|?*'


QUALITY_FOLDER_NAMES = {
    "S": "S_强烈推荐",
    "A": "A_可交付",
    "B": "B_备选",
    "C": "C_留档练习",
    "X": "X_废片",
}


def safe_name(value: str | None, default: str) -> str:
    text = str(value or "").strip() or default
    cleaned = "".join("_" if char in WINDOWS_ILLEGAL_CHARS else char for char in text)
    cleaned = cleaned.rstrip(" .")
    return cleaned or default


def safe_file_name(filename: str) -> str:
    path = Path(filename)
    stem = safe_name(path.stem, "未命名")
    suffix = "".join("_" if char in WINDOWS_ILLEGAL_CHARS else char for char in path.suffix)
    return f"{stem}{suffix or '.jpg'}"


def unique_path(target_path: Path) -> Path:
    if not target_path.exists():
        return target_path
    stem = target_path.stem
    suffix = target_path.suffix
    parent = target_path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter:03d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def timestamped_output_dir(base_dir: Path) -> Path:
    base_dir = Path(base_dir)
    if not base_dir.exists() or not any(base_dir.iterdir()):
        return base_dir
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    candidate = base_dir.with_name(f"{base_dir.name}_{stamp}")
    counter = 1
    while candidate.exists():
        candidate = base_dir.with_name(f"{base_dir.name}_{stamp}_{counter:02d}")
        counter += 1
    return candidate


def quality_folder_name(code: str | None) -> str:
    return QUALITY_FOLDER_NAMES.get(str(code or "").strip(), "未评级")


def readable_list(values: list[str] | tuple[str, ...] | set[str] | str | None) -> str:
    if not values:
        return ""
    if isinstance(values, str):
        try:
            loaded = json.loads(values)
        except json.JSONDecodeError:
            return values
        values = loaded if isinstance(loaded, list) else [str(loaded)]
    return "、".join(str(value) for value in values if str(value))


def readable_json(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return value
        return json.dumps(loaded, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def copy_original(source: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = unique_path(destination_dir / safe_file_name(source.name))
    shutil.copy2(source, target)
    return target


def copy_jpg_preview(source: Path, destination_dir: Path, max_size: int = 2400) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    target_name = f"{safe_name(source.stem, '未命名')}.jpg"
    target = unique_path(destination_dir / target_name)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        image.thumbnail((max_size, max_size))
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(target, "JPEG", quality=92, optimize=True)
    return target
