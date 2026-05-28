from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import Image


EXIF_DATE_TAGS = (36867, 36868, 306)


def read_taken_at(image_path: Path) -> str:
    """Read the best available EXIF capture time."""
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
            for tag in EXIF_DATE_TAGS:
                value = exif.get(tag)
                if not value:
                    continue
                parsed = _normalize_exif_datetime(str(value))
                if parsed:
                    return parsed
    except Exception:
        return ""
    return ""


def _normalize_exif_datetime(value: str) -> str:
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return value.strip()
