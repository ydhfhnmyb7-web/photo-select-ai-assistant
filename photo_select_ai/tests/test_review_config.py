from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.review_config import (
    get_delivery_uses,
    get_issue_tags,
    get_photo_types,
    get_review_preset,
    get_subtypes,
    validate_review_fields,
)


def test_photo_types_can_be_read() -> None:
    photo_types = get_photo_types()
    assert "婚纱照" in photo_types
    assert "个人写真" in photo_types
    assert "无法判断" in photo_types


def test_portrait_subtypes() -> None:
    subtypes = get_subtypes("个人写真")
    assert "生日写真" in subtypes
    assert "情绪写真" in subtypes
    assert "特写" in subtypes


def test_wedding_subtypes() -> None:
    subtypes = get_subtypes("婚纱照")
    assert "主纱" in subtypes
    assert "秀禾/中式" in subtypes
    assert "双人互动" in subtypes


def test_unknown_photo_type_returns_default_subtype() -> None:
    assert get_subtypes("不存在的类型") == ["未细分"]


def test_portrait_preset_can_be_read() -> None:
    preset = get_review_preset("个人写真")
    assert preset["表情情绪"] == 25
    assert preset["姿态自然"] == 20


def test_unknown_photo_type_returns_default_preset() -> None:
    preset = get_review_preset("不存在的类型")
    assert preset["清晰度"] == 25
    assert preset["后期潜力"] == 15


def test_delivery_uses_and_issue_tags() -> None:
    assert "精修候选" in get_delivery_uses()
    assert "不导出" in get_delivery_uses()
    assert "虚焦" in get_issue_tags()
    assert "作品感不足" in get_issue_tags()


def test_validate_review_fields_reports_errors_without_crashing() -> None:
    errors = validate_review_fields(
        {
            "photo_type": "个人写真",
            "subtype": "主纱",
            "quality_rating": "Q",
            "delivery_use": ["精修候选", "未知用途"],
            "issue_tags": ["虚焦", "未知问题"],
        }
    )
    assert any("细分类别不合法" in error for error in errors)
    assert any("品质等级不合法" in error for error in errors)
    assert any("交付用途不合法" in error for error in errors)
    assert any("问题标签不合法" in error for error in errors)


if __name__ == "__main__":
    test_photo_types_can_be_read()
    test_portrait_subtypes()
    test_wedding_subtypes()
    test_unknown_photo_type_returns_default_subtype()
    test_portrait_preset_can_be_read()
    test_unknown_photo_type_returns_default_preset()
    test_delivery_uses_and_issue_tags()
    test_validate_review_fields_reports_errors_without_crashing()
    print("test_review_config_ok")
