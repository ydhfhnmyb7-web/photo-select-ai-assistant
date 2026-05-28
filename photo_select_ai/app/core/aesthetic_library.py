from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import PROJECT_ROOT


LIBRARY_PATH = PROJECT_ROOT / "data" / "aesthetic_library.json"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


AESTHETIC_TAGS = [
    "我喜欢",
    "我不喜欢",
    "构图好",
    "光线好",
    "表情好",
    "色彩好",
    "氛围好",
    "商业可交付",
    "作品集级别",
    "废片",
    "可修图",
]


@dataclass
class AestheticLibrary:
    folders: list[str] = field(default_factory=list)
    images: dict[str, list[str]] = field(default_factory=dict)


def load_aesthetic_library() -> AestheticLibrary:
    if not LIBRARY_PATH.exists():
        return AestheticLibrary()
    try:
        raw = json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))
        folders = [str(item) for item in raw.get("folders", [])]
        images = {
            str(path): [str(tag) for tag in tags]
            for path, tags in (raw.get("images") or {}).items()
            if isinstance(tags, list)
        }
        return AestheticLibrary(folders=folders, images=images)
    except Exception:
        return AestheticLibrary()


def save_aesthetic_library(library: AestheticLibrary) -> None:
    LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIBRARY_PATH.write_text(
        json.dumps({"folders": library.folders, "images": library.images}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add_aesthetic_folder(folder: Path) -> AestheticLibrary:
    library = load_aesthetic_library()
    value = str(folder)
    if value not in library.folders:
        library.folders.append(value)
    save_aesthetic_library(library)
    return library


def remove_aesthetic_folder(folder: Path) -> AestheticLibrary:
    library = load_aesthetic_library()
    value = str(folder)
    library.folders = [item for item in library.folders if item != value]
    save_aesthetic_library(library)
    return library


def scan_aesthetic_library() -> dict:
    library = load_aesthetic_library()
    found = 0
    skipped = 0
    for folder_text in library.folders:
        folder = Path(folder_text)
        if not folder.exists():
            skipped += 1
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                library.images.setdefault(str(path), [])
                found += 1
    save_aesthetic_library(library)
    return {"folders": len(library.folders), "images": len(library.images), "scanned": found, "missing_folders": skipped}


def tag_aesthetic_image(image_path: Path, tags: list[str]) -> None:
    library = load_aesthetic_library()
    valid_tags = [tag for tag in tags if tag in AESTHETIC_TAGS]
    library.images[str(image_path)] = valid_tags
    save_aesthetic_library(library)


def aesthetic_report() -> str:
    library = load_aesthetic_library()
    counter = Counter(tag for tags in library.images.values() for tag in tags)
    lines = [
        "审美素材库报告",
        "",
        f"素材文件夹：{len(library.folders)} 个",
        f"已扫描图片：{len(library.images)} 张",
        "",
        "标签分布：",
    ]
    if counter:
        lines.extend(f"- {tag}: {count} 张" for tag, count in counter.most_common())
    else:
        lines.append("- 暂无标签。")
    lines.extend(
        [
            "",
            "数据来源规则：软件不会自动爬取网络图片；只使用你手动添加的本地素材文件夹。",
            "请确认素材库中的图片来自你的作品、你拥有使用权的数据，或公开授权数据集。",
        ]
    )
    return "\n".join(lines)
