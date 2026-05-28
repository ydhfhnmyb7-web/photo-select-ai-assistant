from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path

from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.mvp_models import PhotoItem
from app.database.db import PhotoRepository
from app.ui.similar_group_panel import SimilarGroupPanel


def _app() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def test_similar_group_panel_lists_and_marks_status() -> None:
    app = _app()
    root = Path("_tmp_similar_group_panel")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    image_a = root / "a.jpg"
    image_b = root / "b.jpg"
    image_c = root / "c.jpg"
    for path in [image_a, image_b, image_c]:
        path.write_bytes(b"not-a-real-image-but-a-safe-original")
    items = [
        PhotoItem(
            path=image_a,
            filename="a.jpg",
            thumbnail_path=None,
            width=100,
            height=80,
            file_size_mb=0.1,
            similar_group_id="组001",
            similar_group_rank=1,
            ai_recommended_best=True,
        ),
        PhotoItem(
            path=image_b,
            filename="b.jpg",
            thumbnail_path=None,
            width=100,
            height=80,
            file_size_mb=0.1,
            similar_group_id="组001",
            similar_group_rank=2,
        ),
        PhotoItem(path=image_c, filename="c.jpg", thumbnail_path=None, width=100, height=80, file_size_mb=0.1),
    ]
    repo = PhotoRepository(root / "cache" / "photoselect.db")
    try:
        panel = SimilarGroupPanel()
        emitted: list[tuple[int, int]] = []

        def save_changed(changed, index):
            emitted.append((len(changed), index))
            repo.save_items(changed)

        panel.group_status_changed.connect(save_changed)
        panel.set_items(items, current_index=1)

        assert panel.group_list.count() == 1
        assert panel.current_group_id == "组001"

        panel._apply_status(items[1], "duplicate")
        assert emitted[-1] == (2, 1)
        assert items[1].similar_group_status == "duplicate"
        assert "重复照片" in items[1].issue_tags
        assert "不导出" in items[1].delivery_use
        assert image_b.exists()

        saved = repo.load_item_map()[str(image_b)]
        assert saved["similar_group_status"] == "duplicate"
        assert "重复照片" in saved["issue_tags"]
        assert "不导出" in saved["delivery_use"]
        panel.deleteLater()
        app.processEvents()
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_similar_group_panel_lists_and_marks_status()
    print("similar group panel tests passed")
