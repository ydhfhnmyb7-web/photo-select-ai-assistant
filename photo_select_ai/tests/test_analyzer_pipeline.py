from __future__ import annotations

import shutil
import sys
import json
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analyzers.analyzer_pipeline import (
    AI_ANALYZER_NAME,
    CURRENT_AI_SUGGESTION_VERSION,
    ai_suggestion_status,
    analyze_photo,
    apply_analysis_to_item,
)
from app.core.mvp_models import PhotoItem
from app.core.review_models import REVIEW_STATUS_AI_REVIEWED, REVIEW_STATUS_HUMAN_CONFIRMED


def test_analyzer_pipeline_basic_and_manual_protection() -> None:
    root = Path("_tmp_analyzer_pipeline_test")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        path = root / "sample.jpg"
        Image.new("RGB", (640, 480), (180, 185, 190)).save(path)
        result = analyze_photo(path)
        assert result["version"] == CURRENT_AI_SUGGESTION_VERSION
        assert result["analyzer_name"] == AI_ANALYZER_NAME
        assert "basic_quality" in result
        assert "decision" in result
        assert result["basic_quality"]["score"] >= 0
        assert ai_suggestion_status(result, path)[0] == "current"
        assert ai_suggestion_status({"image_path": str(path), "decision": {}}, path)[0] == "legacy"
        assert ai_suggestion_status(result, root / "other.jpg")[0] == "mismatch"

        item = PhotoItem(path=path, filename=path.name, thumbnail_path=None, width=640, height=480, file_size_mb=0.01)
        item.ai_suggestion = json.dumps({"image_path": str(path), "decision": {"photo_type": "全家福"}}, ensure_ascii=False)
        apply_analysis_to_item(item, result)
        assert item.review_status == REVIEW_STATUS_AI_REVIEWED
        assert item.ai_suggestion
        refreshed = json.loads(item.ai_suggestion)
        assert refreshed["version"] == CURRENT_AI_SUGGESTION_VERSION
        assert refreshed["decision"].get("photo_type") != "全家福"

        confirmed = PhotoItem(
            path=path,
            filename=path.name,
            thumbnail_path=None,
            width=640,
            height=480,
            file_size_mb=0.01,
            photo_type="婚纱照",
            quality_rating="S",
            review_status=REVIEW_STATUS_HUMAN_CONFIRMED,
        )
        apply_analysis_to_item(confirmed, result)
        assert confirmed.review_status == REVIEW_STATUS_HUMAN_CONFIRMED
        assert confirmed.photo_type == "婚纱照"
        assert confirmed.quality_rating == "S"
        assert confirmed.ai_suggestion
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_analyzer_pipeline_basic_and_manual_protection()
    print("analyzer pipeline tests passed")
