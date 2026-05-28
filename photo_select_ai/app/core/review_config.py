from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config.review_taxonomy import (
    DELIVERY_USE_OPTIONS,
    ISSUE_TAG_OPTIONS,
    PHOTO_TYPES,
    QUALITY_RATINGS,
    SUBTYPES_BY_PHOTO_TYPE,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
REVIEW_PRESETS_PATH = CONFIG_DIR / "review_presets.yaml"
DEFAULT_PRESET_NAME = "默认通用标准"
DEFAULT_SUBTYPE = "未细分"


def get_photo_types() -> list[str]:
    return list(PHOTO_TYPES)


def get_subtypes(photo_type: str) -> list[str]:
    return list(SUBTYPES_BY_PHOTO_TYPE.get(photo_type, [DEFAULT_SUBTYPE]))


def get_quality_ratings() -> list[dict[str, str]]:
    return [{"code": code, "label": label, "display": f"{code}：{label}"} for code, label in QUALITY_RATINGS]


def get_delivery_uses() -> list[str]:
    return list(DELIVERY_USE_OPTIONS)


def get_issue_tags() -> list[str]:
    return list(ISSUE_TAG_OPTIONS)


def get_review_preset(photo_type: str) -> dict[str, int]:
    presets = load_review_presets()
    selected = presets.get(photo_type) or presets.get(DEFAULT_PRESET_NAME) or {}
    return dict(selected)


def validate_review_fields(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    photo_type = str(data.get("photo_type") or "")
    subtype = str(data.get("subtype") or "")
    quality_rating = str(data.get("quality_rating") or "")
    delivery_use = _as_list(data.get("delivery_use"))
    issue_tags = _as_list(data.get("issue_tags"))

    if photo_type and photo_type not in PHOTO_TYPES:
        errors.append(f"照片类型不合法：{photo_type}")

    valid_subtypes = get_subtypes(photo_type)
    if subtype and subtype not in valid_subtypes:
        errors.append(f"细分类别不合法：{subtype}")

    valid_ratings = {code for code, _label in QUALITY_RATINGS}
    if quality_rating and quality_rating not in valid_ratings:
        errors.append(f"品质等级不合法：{quality_rating}")

    invalid_delivery = [value for value in delivery_use if value not in DELIVERY_USE_OPTIONS]
    if invalid_delivery:
        errors.append(f"交付用途不合法：{', '.join(invalid_delivery)}")

    invalid_issues = [value for value in issue_tags if value not in ISSUE_TAG_OPTIONS]
    if invalid_issues:
        errors.append(f"问题标签不合法：{', '.join(invalid_issues)}")

    return errors


def load_review_presets() -> dict[str, dict[str, int]]:
    if not REVIEW_PRESETS_PATH.exists():
        return {}

    text = REVIEW_PRESETS_PATH.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
        return _normalize_presets(loaded)
    except Exception:
        return _parse_simple_yaml_mapping(text)


def _normalize_presets(value: Any) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict):
        return {}
    presets: dict[str, dict[str, int]] = {}
    for preset_name, weights in value.items():
        if not isinstance(weights, dict):
            continue
        presets[str(preset_name)] = {str(key): int(weight) for key, weight in weights.items()}
    return presets


def _parse_simple_yaml_mapping(text: str) -> dict[str, dict[str, int]]:
    presets: dict[str, dict[str, int]] = {}
    current_name = ""
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_name = line[:-1].strip()
            presets[current_name] = {}
            continue
        if current_name and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            try:
                presets[current_name][key] = int(value)
            except ValueError:
                continue
    return presets


def _as_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple) or isinstance(value, set):
        return [str(item) for item in value]
    return [str(value)]

