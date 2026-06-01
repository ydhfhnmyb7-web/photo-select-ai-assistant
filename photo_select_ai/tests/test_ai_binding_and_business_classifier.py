from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from PIL import Image
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analyzers.analyzer_result import AnalyzerResult
from app.analyzers.analyzer_pipeline import CURRENT_AI_SUGGESTION_VERSION
from app.analyzers.business_type_classifier import classify_business_type
from app.ui.main_window import MainWindow
from app.core.mvp_models import PhotoItem


def _face_result(face_count: int, **raw) -> AnalyzerResult:
    return AnalyzerResult(
        module_name="face",
        score=80,
        confidence=0.8,
        tags=[f"检测到{face_count}张人脸"],
        problems=[],
        reason="mock face result",
        raw_data={"face_count": face_count, **raw},
    )


def test_single_person_cannot_be_family() -> None:
    root = Path("_tmp_business_classifier_test")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        path = root / "single.jpg"
        Image.new("RGB", (600, 900), (180, 170, 160)).save(path)
        result = classify_business_type(path, _face_result(1, person_count=1, scene_hint="棚拍"))
        assert result.raw_data["photo_type"] != "家庭合影"
        assert result.raw_data["photo_type"] != "会议/活动照"
        assert result.raw_data["subtype"] != "多人家庭"
        assert result.raw_data["photo_type"] == "无法判断"
        assert result.raw_data["candidate_photo_type"] == "个人写真"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_single_bride_prioritizes_wedding() -> None:
    root = Path("_tmp_business_classifier_test")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        path = root / "single_bride.jpg"
        Image.new("RGB", (900, 1300), (210, 210, 205)).save(path)
        result = classify_business_type(
            path,
            _face_result(
                1,
                person_count=1,
                scene_hint="棚拍",
                clothing_hint="婚纱",
                background_hint="棚拍背景",
                full_body_or_half_body="全身",
            ),
        )
        assert result.raw_data["photo_type"] == "婚纱写真"
        assert result.raw_data["subtype"] in {"单人新娘", "主纱", "棚内婚纱"}
        assert result.confidence >= 0.65
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_studio_suit_portrait_is_not_meeting_event() -> None:
    root = Path("_tmp_business_classifier_test")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        path = root / "studio_suit.jpg"
        Image.new("RGB", (800, 1100), (170, 165, 160)).save(path)
        result = classify_business_type(
            path,
            _face_result(
                1,
                person_count=1,
                scene_hint="棚拍",
                clothing_hint="西装",
                background_hint="棚拍背景",
                full_body_or_half_body="半身",
            ),
        )
        assert result.raw_data["photo_type"] == "商务形象照"
        assert result.raw_data["photo_type"] != "会议/活动照"
        assert result.confidence >= 0.65
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_bouquet_studio_prioritizes_wedding_portrait() -> None:
    root = Path("_tmp_business_classifier_test")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        path = root / "bouquet_bride.jpg"
        Image.new("RGB", (900, 1300), (205, 202, 198)).save(path)
        result = classify_business_type(
            path,
            _face_result(
                1,
                person_count=1,
                scene_hint="棚拍",
                prop_hint="花束",
                background_hint="棚拍背景",
                full_body_or_half_body="全身",
            ),
        )
        assert result.raw_data["photo_type"] == "婚纱写真"
        assert result.raw_data["photo_type"] != "会议/活动照"
        assert result.confidence >= 0.65
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_family_requires_three_or_more_people_and_remains_low_confidence() -> None:
    root = Path("_tmp_business_classifier_test")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        path = root / "group.jpg"
        Image.new("RGB", (1200, 800), (120, 120, 120)).save(path)
        result = classify_business_type(path, _face_result(3, person_count=3))
        assert result.raw_data["candidate_photo_type"] == "家庭合影"
        assert result.raw_data["photo_type"] == "无法判断"
        assert "建议人工确认" in result.problems
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_low_confidence_outputs_unknown() -> None:
    root = Path("_tmp_business_classifier_test")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        path = root / "plain_single.jpg"
        Image.new("RGB", (640, 900), (160, 150, 145)).save(path)
        result = classify_business_type(path, _face_result(1, person_count=1))
        assert result.confidence < 0.65
        assert result.raw_data["photo_type"] == "无法判断"
        assert result.raw_data["subtype"] == "未细分"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_meeting_event_requires_clear_meeting_scene() -> None:
    root = Path("_tmp_business_classifier_test")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        path = root / "meeting_group.jpg"
        Image.new("RGB", (1200, 800), (105, 105, 108)).save(path)
        result = classify_business_type(
            path,
            _face_result(
                8,
                person_count=8,
                scene_hint="会议",
                background_hint="会议室",
                has_banner=True,
            ),
        )
        assert result.raw_data["photo_type"] == "会议/活动照"
        assert result.confidence >= 0.65
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_ai_suggestion_does_not_stick_between_photos() -> None:
    root = Path("_tmp_ai_binding_test")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        path_a = root / "a.jpg"
        path_b = root / "b.jpg"
        Image.new("RGB", (120, 120), (255, 255, 255)).save(path_a)
        Image.new("RGB", (120, 120), (20, 20, 20)).save(path_b)
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        item_a = PhotoItem(path=path_a, filename=path_a.name, thumbnail_path=None, width=120, height=120, file_size_mb=0.01)
        item_b = PhotoItem(path=path_b, filename=path_b.name, thumbnail_path=None, width=120, height=120, file_size_mb=0.01)
        item_a.ai_suggestion = json.dumps(
            {
                "version": CURRENT_AI_SUGGESTION_VERSION,
                "analyzer_name": "analyzer_pipeline",
                "image_path": str(path_a),
                "decision": {
                    "photo_type": "婚纱写真",
                    "subtype": "单人新娘",
                    "quality_rating": "A",
                    "delivery_use": ["客户可选"],
                    "issue_tags": [],
                    "confidence": 0.8,
                    "reason": "mock",
                },
            },
            ensure_ascii=False,
        )
        window.items = [item_a, item_b]
        window.current_source_index = 0
        window.show_item(item_a)
        assert "AI建议，仅供参考" in window.review_ai_suggestion_label.text()
        assert "婚纱写真" in window.review_ai_suggestion_label.text()
        window.current_source_index = 1
        window.show_item(item_b)
        assert window.review_ai_suggestion_label.text().startswith("暂无 AI 建议")
        item_b.ai_suggestion = item_a.ai_suggestion
        window.show_item(item_b)
        assert "建议重新分析" in window.review_ai_suggestion_label.text()
        assert "未分析" in window.ai_labels["module_business"].text()
        item_b.photo_type = "人工类型"
        item_b.delivery_use = ["不导出"]
        item_b.issue_tags = ["重复照片"]
        assert window._clear_stale_ai_suggestion_fields(item_b) is True
        assert item_b.ai_suggestion == ""
        assert item_b.photo_type == "人工类型"
        assert item_b.delivery_use == ["不导出"]
        assert item_b.issue_tags == ["重复照片"]
        window.close()
        app.processEvents()
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_single_person_cannot_be_family()
    test_single_bride_prioritizes_wedding()
    test_studio_suit_portrait_is_not_meeting_event()
    test_bouquet_studio_prioritizes_wedding_portrait()
    test_family_requires_three_or_more_people_and_remains_low_confidence()
    test_low_confidence_outputs_unknown()
    test_meeting_event_requires_clear_meeting_scene()
    test_ai_suggestion_does_not_stick_between_photos()
    print("ai binding and business classifier tests passed")
