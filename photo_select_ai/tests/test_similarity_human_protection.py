from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analyzers.similarity_analyzer import calculate_similarity_groups
from app.core.mvp_models import PhotoItem
from app.core.review_models import REVIEW_STATUS_HUMAN_CONFIRMED, REVIEW_STATUS_NEEDS_REVIEW


def _make_image(path: Path, variant: int = 0) -> None:
    image = Image.new("RGB", (360, 240), (235, 235, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 30, 170, 190), fill=(30 + variant, 90, 180))
    draw.ellipse((205, 70, 290, 155), fill=(210, 70 + variant, 80))
    draw.line((20, 220, 340, 25), fill=(25, 25, 25), width=6)
    image.save(path)


def _item(path: Path, status: str = "") -> PhotoItem:
    decision = ""
    review_status = ""
    if status:
        review_status = REVIEW_STATUS_NEEDS_REVIEW if status == "review" else REVIEW_STATUS_HUMAN_CONFIRMED
        decision = json.dumps(
            {
                "similar_group_status": status,
                "best_in_group": status == "best",
                "decision_source": "human",
                "updated_at": "2026-05-22 17:30:00",
                "note": f"人工设为{status}",
            },
            ensure_ascii=False,
        )
    return PhotoItem(
        path=path,
        filename=path.name,
        thumbnail_path=None,
        width=360,
        height=240,
        file_size_mb=0.01,
        similar_group_id="旧组999",
        similar_group_status=status,
        best_in_group=status == "best",
        review_status=review_status,
        human_group_decision=decision,
    )


def test_human_similarity_status_survives_recalculation() -> None:
    root = Path("_tmp_similarity_human_protection")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        paths: list[Path] = []
        for index in range(4):
            path = root / f"similar_{index}.jpg"
            _make_image(path, index)
            paths.append(path)
        different = root / "different.jpg"
        Image.new("RGB", (360, 240), (20, 20, 20)).save(different)

        items = [
            _item(paths[0], "best"),
            _item(paths[1], "duplicate"),
            _item(paths[2], "backup"),
            _item(paths[3], "review"),
            _item(different),
        ]
        items[1].delivery_use = ["不导出"]
        items[1].issue_tags = ["重复照片"]
        items[1].quality_rating = "X"
        items[2].quality_rating = "B"
        items[3].similar_group_note = "人工要求复核这一张"

        old_decisions = [item.human_group_decision for item in items[:4]]
        result = calculate_similarity_groups(items, mode="standard")

        assert result.groups
        assert all(item.similar_group_id and item.similar_group_id != "旧组999" for item in items[:4])
        assert not items[4].similar_group_id

        assert items[0].best_in_group is True
        assert items[0].similar_group_status == "best"
        assert items[1].similar_group_status == "duplicate"
        assert items[2].similar_group_status == "backup"
        assert items[3].similar_group_status == "review"
        assert items[3].review_status == REVIEW_STATUS_NEEDS_REVIEW

        assert items[1].delivery_use == ["不导出"]
        assert items[1].issue_tags == ["重复照片"]
        assert items[1].quality_rating == "X"
        assert items[2].quality_rating == "B"
        assert items[3].similar_group_note == "人工要求复核这一张"
        assert [item.human_group_decision for item in items[:4]] == old_decisions

        assert sum(1 for item in items[:4] if item.ai_recommended_best) == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_human_similarity_status_survives_recalculation()
    print("similarity human protection tests passed")
