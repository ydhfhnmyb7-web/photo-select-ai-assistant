from __future__ import annotations

from pathlib import Path
from typing import Any


def analyze_photo_with_ai(image_path: str | Path) -> dict[str, Any]:
    """Future extension point for OpenAI Vision API or local vision models.

    The MVP deliberately returns neutral placeholder values so the rest of the
    application can be wired against a stable contract without requiring an API key.
    """
    _ = Path(image_path)
    return {
        "aesthetic_score": None,
        "composition_score": None,
        "emotion_score": None,
        "commercial_value_score": None,
        "tags": [],
        "retouching_advice": "",
        "recommended_use": "",
    }
