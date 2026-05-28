from __future__ import annotations

import json
from dataclasses import dataclass, field


REVIEW_STATUS_UNREVIEWED = "未审"
REVIEW_STATUS_AI_REVIEWED = "AI已审"
REVIEW_STATUS_HUMAN_CONFIRMED = "人工已确认"
REVIEW_STATUS_NEEDS_REVIEW = "待复核"

QUALITY_RATING_DEFAULT = ""


def encode_multi_select(values: list[str]) -> str:
    return json.dumps([str(value) for value in values if str(value).strip()], ensure_ascii=False)


def decode_multi_select(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(loaded, list):
        return []
    return [str(item) for item in loaded if str(item).strip()]


@dataclass
class ReviewFields:
    photo_type: str = ""
    subtype: str = ""
    quality_rating: str = QUALITY_RATING_DEFAULT
    delivery_use: list[str] = field(default_factory=list)
    issue_tags: list[str] = field(default_factory=list)
    commercial_score: float = 0.0
    portfolio_score: float = 0.0
    ai_suggestion: str = ""
    human_decision: str = ""
    review_status: str = REVIEW_STATUS_UNREVIEWED
    best_in_group: bool = False
    review_note: str = ""

