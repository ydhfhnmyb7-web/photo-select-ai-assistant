from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Future RAW hooks. The MVP does not decode these formats yet.
RAW_EXTENSIONS = {".raw", ".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".orf", ".rw2"}


def scan_image_files(folder: Path) -> list[Path]:
    files: list[Path] = []
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files, key=lambda item: str(item).lower())


def scan_raw_candidates(folder: Path) -> list[Path]:
    return sorted(
        [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in RAW_EXTENSIONS],
        key=lambda item: str(item).lower(),
    )


def read_bgr_image(image_path: Path) -> np.ndarray:
    """Read an image with Unicode-path support on Windows."""
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is not None:
        return image

    # Pillow fallback helps with formats OpenCV may not decode in some builds.
    with Image.open(image_path) as pil_image:
        pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
        rgb = np.array(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def read_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        return image.size


def describe_orientation(width: int, height: int) -> str:
    if width > height:
        return "横幅"
    if height > width:
        return "竖幅"
    return "方图"


def generate_thumbnail(image_path: Path, thumbnail_dir: Path, max_size: tuple[int, int] = (256, 256)) -> Path:
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = thumbnail_dir / _thumbnail_filename(image_path)

    source_mtime = int(image_path.stat().st_mtime)
    if thumb_path.exists() and int(thumb_path.stat().st_mtime) >= source_mtime:
        return thumb_path

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        image = _ensure_rgb(image)
        image.save(thumb_path, "JPEG", quality=88, optimize=True)
    return thumb_path


def average_hash(image_path: Path, hash_size: int = 8) -> str:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        image = image.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32)

    avg = pixels.mean()
    bits = pixels > avg
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    width = hash_size * hash_size // 4
    return f"{value:0{width}x}"


def _thumbnail_filename(image_path: Path) -> str:
    stat = image_path.stat()
    digest_source = f"{image_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8", "ignore")
    digest = hashlib.sha1(digest_source).hexdigest()[:16]
    return f"{image_path.stem}_{digest}.jpg"


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGB", image.size, "white")
        alpha = image.getchannel("A")
        background.paste(image.convert("RGB"), mask=alpha)
        return background
    return image.convert("RGB")
