from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.background_task_manager import (
    TASK_MODEL_AUTO_GROUP,
    TASK_NAMES,
    TASK_SIMILAR_PHOTOS,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_ERROR,
    BackgroundTask,
    BackgroundTaskWorker,
    build_background_tasks,
)
from app.core.config import AppConfig
from app.core.mvp_models import PhotoItem


def _make_image(path: Path, color: tuple[int, int, int] = (180, 180, 180)) -> None:
    Image.new("RGB", (320, 240), color).save(path)


def _item(path: Path) -> PhotoItem:
    return PhotoItem(
        path=path,
        filename=path.name,
        thumbnail_path=None,
        width=320,
        height=240,
        file_size_mb=0.01,
    )


def test_model_auto_group_task_registered() -> None:
    assert TASK_MODEL_AUTO_GROUP in TASK_NAMES
    tasks = build_background_tasks([TASK_MODEL_AUTO_GROUP], [])
    assert tasks[0].task_type == TASK_MODEL_AUTO_GROUP
    assert tasks[0].task_name == "模型自动分组 v1"


def test_model_auto_group_task_writes_auto_fields_and_preserves_manual_group_fields() -> None:
    root = Path("_tmp_background_model_auto_group")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        paths = [root / "same_1.jpg", root / "same_2.jpg", root / "same_3.jpg", root / "different.jpg"]
        for path in paths:
            _make_image(path)
        items = [_item(path) for path in paths]
        items[1].similar_group_id = "manual_group"
        items[1].similar_group_status = "duplicate"
        items[1].best_in_group = True
        items[1].human_group_decision = '{"decision_source":"human","similar_group_status":"duplicate"}'
        items[1].delivery_use = ["不导出"]
        items[1].issue_tags = ["重复照片"]
        items[1].quality_rating = "X"
        items[1].review_status = "人工已确认"

        task = BackgroundTask(task_type=TASK_MODEL_AUTO_GROUP, task_name=TASK_NAMES[TASK_MODEL_AUTO_GROUP], total_count=len(items))
        worker = BackgroundTaskWorker([task], items, AppConfig(embedding_backend="mock", embedding_similarity_threshold=0.86), root)
        worker._run_task(task)

        assert task.status == TASK_STATUS_COMPLETED
        assert task.success_count == 4
        assert "分组 1" in task.error_message
        assert items[0].auto_group_id
        assert items[0].auto_group_id == items[1].auto_group_id == items[2].auto_group_id
        assert items[3].auto_group_id == ""
        assert items[0].grouping_method == "mock_embedding"
        assert items[0].embedding_model_name == "mock_embedding"
        assert items[0].embedding_cache_key
        assert items[1].similar_group_id == "manual_group"
        assert items[1].similar_group_status == "duplicate"
        assert items[1].best_in_group
        assert items[1].human_group_decision
        assert items[1].delivery_use == ["不导出"]
        assert items[1].issue_tags == ["重复照片"]
        assert items[1].quality_rating == "X"
        assert items[1].review_status == "人工已确认"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_model_auto_group_task_failure_is_reported_without_replacing_similarity_task() -> None:
    import app.model_pipeline.model_pipeline_v1 as pipeline_module

    original = pipeline_module.run_model_pipeline_v1

    def broken_pipeline(*_args, **_kwargs):
        raise RuntimeError("pipeline exploded")

    root = Path("_tmp_background_model_auto_group_failure")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        path = root / "same_1.jpg"
        _make_image(path)
        items = [_item(path)]
        pipeline_module.run_model_pipeline_v1 = broken_pipeline

        task = BackgroundTask(task_type=TASK_MODEL_AUTO_GROUP, task_name=TASK_NAMES[TASK_MODEL_AUTO_GROUP], total_count=len(items))
        worker = BackgroundTaskWorker([task], items, AppConfig(embedding_backend="mock"), root)
        worker.logger.exception = lambda *_args, **_kwargs: None
        worker._run_task(task)

        assert task.status == TASK_STATUS_ERROR
        assert "pipeline exploded" in task.error_message
        assert worker._handler_for(TASK_SIMILAR_PHOTOS).__name__ == "_run_similarity_task"
    finally:
        pipeline_module.run_model_pipeline_v1 = original
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_model_auto_group_task_registered()
    test_model_auto_group_task_writes_auto_fields_and_preserves_manual_group_fields()
    test_model_auto_group_task_failure_is_reported_without_replacing_similarity_task()
    print("background model auto group task tests passed")
