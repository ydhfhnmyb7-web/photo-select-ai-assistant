from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from PIL import Image
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.mvp_models import PhotoItem
from app.core.review_config import get_delivery_uses, get_issue_tags
from app.core.review_models import REVIEW_STATUS_HUMAN_CONFIRMED
from app.ui.dual_compare_dialog import DualCompareDialog


def _app() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (420, 280), color).save(path)


def _item(path: Path, rank: int) -> PhotoItem:
    return PhotoItem(
        path=path,
        filename=path.name,
        thumbnail_path=None,
        width=420,
        height=280,
        file_size_mb=0.01,
        similar_group_id="组001",
        similar_group_rank=rank,
        similarity_score=0.94 - rank * 0.01,
    )


def test_dual_compare_status_actions_and_human_protection() -> None:
    app = _app()
    root = Path("_tmp_dual_compare_dialog")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        left_path = root / "left.jpg"
        right_path = root / "right.jpg"
        _make_image(left_path, (220, 220, 220))
        _make_image(right_path, (210, 215, 230))

        delivery_no_export = get_delivery_uses()[8]
        issue_duplicate = get_issue_tags()[10]
        items = [_item(left_path, 1), _item(right_path, 2)]
        items[1].similar_group_status = "duplicate"
        items[1].review_status = REVIEW_STATUS_HUMAN_CONFIRMED
        items[1].delivery_use = [delivery_no_export]
        items[1].issue_tags = [issue_duplicate]

        dialog = DualCompareDialog(items, 0, 1)
        emitted: list[tuple[int, int]] = []
        dialog.group_status_changed.connect(lambda changed, index: emitted.append((len(changed), index)))

        dialog._apply_status(items[0], "best")
        assert emitted[-1] == (2, 0)
        assert items[0].best_in_group is True
        assert items[0].similar_group_status == "best"
        assert items[1].similar_group_status == "duplicate"
        assert items[1].delivery_use == [delivery_no_export]
        assert items[1].issue_tags == [issue_duplicate]

        dialog._apply_status(items[0], "backup")
        assert emitted[-1] == (2, 0)
        assert items[0].best_in_group is False
        assert items[0].similar_group_status == "backup"
        assert items[0].quality_rating == "B"

        dialog._apply_status(items[0], "duplicate")
        assert items[0].similar_group_status == "duplicate"
        assert delivery_no_export in items[0].delivery_use
        assert issue_duplicate in items[0].issue_tags
        assert left_path.exists()
        assert right_path.exists()
        dialog.deleteLater()
        app.processEvents()
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_dual_compare_status_actions_and_human_protection()
    print("dual compare dialog tests passed")
