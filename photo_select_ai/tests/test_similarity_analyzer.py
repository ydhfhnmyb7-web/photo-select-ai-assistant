from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analyzers.similarity_analyzer import build_similarity_summary, calculate_similarity_groups, set_group_status
from app.core.mvp_models import PhotoItem


def _make_pattern(path: Path, invert: bool = False) -> None:
    image = Image.new("RGB", (320, 240), (245, 245, 245) if not invert else (30, 30, 30))
    draw = ImageDraw.Draw(image)
    fill = (40, 80, 190) if not invert else (230, 220, 120)
    draw.rectangle((40, 40, 180, 170), fill=fill)
    draw.ellipse((190, 80, 265, 155), fill=(200, 60, 80) if not invert else (60, 180, 210))
    draw.line((10, 220, 310, 20), fill=(20, 20, 20) if not invert else (240, 240, 240), width=5)
    image.save(path)


def _item(path: Path) -> PhotoItem:
    return PhotoItem(path=path, filename=path.name, thumbnail_path=None, width=320, height=240, file_size_mb=0.01)


def test_similarity_grouping_and_manual_status() -> None:
    root = Path("_tmp_similarity_test")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        p1 = root / "same_1.jpg"
        p2 = root / "same_2.jpg"
        p3 = root / "same_3.jpg"
        p4 = root / "different.jpg"
        _make_pattern(p1)
        _make_pattern(p2)
        _make_pattern(p3)
        _make_pattern(p4, invert=True)
        items = [_item(p1), _item(p2), _item(p3), _item(p4)]
        items[1].best_in_group = True
        items[1].similar_group_status = "best"

        result = calculate_similarity_groups(items, mode="standard")
        assert result.groups
        assert items[0].similar_group_id
        assert items[1].similar_group_id == items[0].similar_group_id
        assert items[2].similar_group_id == items[0].similar_group_id
        assert not items[3].similar_group_id
        assert items[1].best_in_group
        assert not items[0].best_in_group
        assert sum(1 for item in items[:3] if item.ai_recommended_best) == 1

        summary = build_similarity_summary(items)
        assert summary["similar_group_count"] == 1
        assert summary["photos_in_groups_count"] == 3

        changed = set_group_status(items, items[2], "best")
        assert len(changed) == 3
        assert items[2].best_in_group
        assert not items[1].best_in_group

        set_group_status(items, items[0], "duplicate")
        assert items[0].similar_group_status == "duplicate"
        assert "重复照片" in items[0].issue_tags
        assert "不导出" in items[0].delivery_use
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_similarity_grouping_and_manual_status()
    print("similarity analyzer tests passed")
