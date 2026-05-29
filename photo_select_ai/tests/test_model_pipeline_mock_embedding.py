from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import AppConfig
from app.core.mvp_models import PhotoItem
from app.model_pipeline.embedding_cache import EmbeddingCache
from app.model_pipeline.embedding_extractor import FallbackEmbeddingExtractor, MockEmbeddingExtractor
from app.model_pipeline.model_pipeline_v1 import run_model_pipeline_v1


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (320, 240), color).save(path)


def _make_pattern_image(path: Path) -> None:
    image = Image.new("RGB", (320, 240), (20, 40, 200))
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 18, 150, 210), fill=(245, 240, 20))
    draw.ellipse((190, 48, 288, 168), fill=(10, 10, 10))
    draw.line((0, 235, 318, 0), fill=(255, 255, 255), width=8)
    image.save(path)


def _item(path: Path) -> PhotoItem:
    return PhotoItem(
        path=path,
        filename=path.name,
        thumbnail_path=None,
        width=320,
        height=240,
        file_size_mb=0.01,
    )


def test_model_pipeline_mock_embedding_groups_and_preserves_manual_fields() -> None:
    root = Path("_tmp_model_pipeline_mock_test")
    cache_root = root / "cache"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        paths = [root / "same_1.jpg", root / "same_2.jpg", root / "same_3.jpg", root / "different.jpg"]
        for path in paths:
            _make_image(path, (180, 180, 180))
        items = [_item(path) for path in paths]
        items[1].similar_group_id = "manual_group"
        items[1].similar_group_status = "duplicate"
        items[1].best_in_group = True
        items[1].human_group_decision = '{"decision_source":"human","similar_group_status":"duplicate"}'
        items[1].delivery_use = ["不导出"]
        items[1].issue_tags = ["重复照片"]
        items[1].review_status = "人工已确认"

        config = AppConfig(embedding_similarity_threshold=0.86)
        cache = EmbeddingCache(cache_root)
        result = run_model_pipeline_v1(
            items,
            config=config,
            extractor=MockEmbeddingExtractor(),
            cache=cache,
        )

        assert len(result.grouping.groups) == 1
        assert items[0].auto_group_id
        assert items[0].auto_group_id == items[1].auto_group_id == items[2].auto_group_id
        assert items[3].auto_group_id == ""
        assert items[0].grouping_method == "mock_embedding"
        assert items[0].embedding_model_name == "mock_embedding"
        assert items[0].embedding_cache_key
        assert items[0].embedding_dim >= 4
        assert items[1].similar_group_id == "manual_group"
        assert items[1].similar_group_status == "duplicate"
        assert items[1].best_in_group
        assert items[1].human_group_decision
        assert items[1].delivery_use == ["不导出"]
        assert items[1].issue_tags == ["重复照片"]
        assert items[1].review_status == "人工已确认"

        second = run_model_pipeline_v1(
            items,
            config=config,
            extractor=MockEmbeddingExtractor(),
            cache=cache,
        )
        assert second.cache_hits == 4
        assert second.cache_misses == 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_fallback_embedding_marks_low_confidence_fallback_grouping() -> None:
    root = Path("_tmp_model_pipeline_fallback_test")
    cache_root = root / "cache"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        p1 = root / "copy_1.jpg"
        p2 = root / "copy_2.jpg"
        p3 = root / "different.jpg"
        _make_image(p1, (240, 240, 240))
        shutil.copy2(p1, p2)
        _make_pattern_image(p3)
        items = [_item(p1), _item(p2), _item(p3)]

        result = run_model_pipeline_v1(
            items,
            config=AppConfig(embedding_similarity_threshold=0.94),
            extractor=FallbackEmbeddingExtractor(),
            cache=EmbeddingCache(cache_root),
        )

        assert result.method == "ahash_fallback"
        assert items[0].grouping_method == "ahash_fallback"
        assert items[0].auto_group_id == items[1].auto_group_id
        assert items[0].auto_group_confidence <= 0.72
        assert "fallback" in items[0].auto_group_reason.lower()
        assert items[2].auto_group_id == ""
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_model_pipeline_mock_embedding_groups_and_preserves_manual_fields()
    test_fallback_embedding_marks_low_confidence_fallback_grouping()
    print("model pipeline mock embedding tests passed")
