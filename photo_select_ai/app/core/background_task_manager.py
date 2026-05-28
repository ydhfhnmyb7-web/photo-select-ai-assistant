from __future__ import annotations

import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal, Slot

from app.core.app_logging import get_logger
from app.core.config import AppConfig, PROJECT_ROOT
from app.core.mvp_loader import _thumbnail_cache_size, generate_thumbnail
from app.core.mvp_models import PhotoItem
from app.export.report_generator import build_export_summary


TASK_THUMBNAIL_CACHE = "thumbnail_cache"
TASK_SIMILAR_PHOTOS = "similar_photos"
TASK_AI_PRECLASSIFY = "ai_preclassify"
TASK_FACE_QUALITY = "face_quality"
TASK_POSE_ANALYSIS = "pose_analysis"
TASK_EXPRESSION_ANALYSIS = "expression_analysis"
TASK_PREFERENCE_STATS = "preference_stats"
TASK_EXPORT_PREVIEW = "export_preview"
TASK_EXPORT = "export"

TASK_NAMES = {
    TASK_THUMBNAIL_CACHE: "生成缩略图缓存",
    TASK_SIMILAR_PHOTOS: "计算相似照片",
    TASK_AI_PRECLASSIFY: "AI预分类",
    TASK_FACE_QUALITY: "人脸质量分析",
    TASK_POSE_ANALYSIS: "肢体姿态分析",
    TASK_EXPRESSION_ANALYSIS: "表情状态分析",
    TASK_PREFERENCE_STATS: "审美偏好样本统计",
    TASK_EXPORT_PREVIEW: "生成导出预览",
    TASK_EXPORT: "导出任务",
}

TASK_STATUS_PENDING = "pending"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_PAUSED = "paused"
TASK_STATUS_STOPPING = "stopping"
TASK_STATUS_STOPPED = "stopped"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_ERROR = "error"
TASK_STATUS_CANCELLED = "cancelled"

BACKGROUND_LOG_PATH = PROJECT_ROOT / "logs" / "background_tasks.log"


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class BackgroundTask:
    task_type: str
    task_name: str
    total_count: int = 0
    priority: int = 50
    can_pause: bool = True
    can_cancel: bool = True
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = TASK_STATUS_PENDING
    completed_count: int = 0
    success_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    current_file: str = ""
    error_message: str = ""
    created_at: str = field(default_factory=_now_text)
    started_at: str = ""
    finished_at: str = ""

    def progress_text(self) -> str:
        return f"{self.completed_count}/{self.total_count}" if self.total_count else "0/0"


def build_background_tasks(task_types: list[str], items: list[PhotoItem]) -> list[BackgroundTask]:
    total = len(items)
    tasks: list[BackgroundTask] = []
    for index, task_type in enumerate(task_types):
        task_name = TASK_NAMES.get(task_type, task_type)
        tasks.append(
            BackgroundTask(
                task_type=task_type,
                task_name=task_name,
                total_count=total,
                priority=index,
            )
        )
    return tasks


class BackgroundTaskWorker(QObject):
    task_started = Signal(object)
    task_progress = Signal(object)
    task_paused = Signal(object)
    task_resumed = Signal(object)
    task_stopping = Signal(object)
    task_stopped = Signal(object)
    task_completed = Signal(object)
    task_error = Signal(object)
    queue_changed = Signal(object)
    all_finished = Signal(object)

    def __init__(self, tasks: list[BackgroundTask], items: list[PhotoItem], config: AppConfig, project_folder: Path | None):
        super().__init__()
        self.tasks = tasks
        self.items = items
        self.config = config
        self.project_folder = project_folder
        self._pause_requested = False
        self._stop_requested = False
        self._paused_emitted_for_task = ""
        self.logger = get_logger()

    @Slot()
    def run(self) -> None:
        _write_background_log("后台任务队列开始")
        try:
            for task in self.tasks:
                if self._stop_requested:
                    if task.status == TASK_STATUS_PENDING:
                        task.status = TASK_STATUS_CANCELLED
                        task.finished_at = _now_text()
                    continue
                if task.status not in {TASK_STATUS_PENDING, TASK_STATUS_STOPPED, TASK_STATUS_CANCELLED}:
                    continue
                self._run_task(task)
        finally:
            _write_background_log("后台任务队列结束")
            self.all_finished.emit(self.tasks)

    def pause(self) -> None:
        self._pause_requested = True
        _write_background_log("用户请求暂停后台任务")

    def resume(self) -> None:
        self._pause_requested = False
        self._paused_emitted_for_task = ""
        _write_background_log("用户请求继续后台任务")

    def stop(self) -> None:
        self._stop_requested = True
        _write_background_log("用户请求停止后台任务")

    def _run_task(self, task: BackgroundTask) -> None:
        task.completed_count = 0
        task.success_count = 0
        task.skipped_count = 0
        task.failed_count = 0
        task.current_file = ""
        task.status = TASK_STATUS_RUNNING
        task.started_at = _now_text()
        task.finished_at = ""
        task.error_message = ""
        _write_background_log(f"任务开始：{task.task_name} ({task.task_id})")
        self.task_started.emit(task)
        self.queue_changed.emit(self.tasks)
        try:
            handler = self._handler_for(task.task_type)
            handler(task)
            if task.status not in {TASK_STATUS_STOPPED, TASK_STATUS_CANCELLED, TASK_STATUS_ERROR}:
                task.status = TASK_STATUS_COMPLETED
                task.finished_at = _now_text()
                _write_background_log(
                    f"任务完成：{task.task_name} 成功 {task.success_count} 跳过 {task.skipped_count} 失败 {task.failed_count}"
                )
                self.task_completed.emit(task)
        except Exception as exc:
            task.status = TASK_STATUS_ERROR
            task.error_message = str(exc)
            task.finished_at = _now_text()
            _write_background_log(f"任务出错：{task.task_name} {exc}\n{traceback.format_exc()}")
            self.logger.exception("后台任务出错：%s", task.task_name)
            self.task_error.emit(task)
        finally:
            self.queue_changed.emit(self.tasks)

    def _handler_for(self, task_type: str) -> Callable[[BackgroundTask], None]:
        if task_type == TASK_THUMBNAIL_CACHE:
            return self._run_thumbnail_cache
        if task_type == TASK_EXPORT_PREVIEW:
            return self._run_export_preview
        if task_type == TASK_PREFERENCE_STATS:
            return self._run_preference_stats
        if task_type == TASK_SIMILAR_PHOTOS:
            return self._run_similarity_task
        if task_type in {TASK_AI_PRECLASSIFY, TASK_FACE_QUALITY, TASK_POSE_ANALYSIS, TASK_EXPRESSION_ANALYSIS}:
            return self._run_analyzer_pipeline
        return self._run_placeholder_task

    def _run_thumbnail_cache(self, task: BackgroundTask) -> None:
        thumbnail_dir = self._thumbnail_dir()
        max_size = _thumbnail_cache_size(self.config.thumbnail_size)
        for item in self.items:
            if self._should_stop(task):
                return
            self._wait_if_paused(task)
            task.current_file = item.filename
            try:
                current_thumb = Path(item.thumbnail_path) if item.thumbnail_path else None
                if current_thumb and current_thumb.exists():
                    task.skipped_count += 1
                else:
                    item.thumbnail_path = generate_thumbnail(item.path, thumbnail_dir, max_size=max_size, mode=self.config.thumbnail_mode)
                    task.success_count += 1
            except Exception as exc:
                task.failed_count += 1
                task.error_message = str(exc)
                _write_background_log(f"缩略图失败：{item.path} {exc}")
            finally:
                task.completed_count += 1
                self.task_progress.emit(task)
                self.queue_changed.emit(self.tasks)

    def _run_export_preview(self, task: BackgroundTask) -> None:
        try:
            summary = build_export_summary(self.items)
            task.error_message = (
                f"总照片 {summary.get('total', 0)}，客户可选 {summary.get('client_select_count', 0)}，"
                f"精修候选 {summary.get('retouch_candidate_count', 0)}，待复核 {summary.get('needs_review_count', 0)}"
            )
            task.success_count = len(self.items)
            task.completed_count = len(self.items)
            task.current_file = "导出预览统计已生成"
            self.task_progress.emit(task)
        except Exception as exc:
            task.failed_count = len(self.items)
            task.error_message = str(exc)
            raise

    def _run_preference_stats(self, task: BackgroundTask) -> None:
        for item in self.items:
            if self._should_stop(task):
                return
            self._wait_if_paused(task)
            task.current_file = item.filename
            task.completed_count += 1
            if item.user_label or item.manual_override or item.review_status == "人工已确认":
                task.success_count += 1
            else:
                task.skipped_count += 1
            self.task_progress.emit(task)

    def _run_analyzer_pipeline(self, task: BackgroundTask) -> None:
        from app.analyzers.analyzer_pipeline import analyze_photo, apply_analysis_to_item
        from app.core.review_models import REVIEW_STATUS_HUMAN_CONFIRMED

        for item in self.items:
            if self._should_stop(task):
                return
            self._wait_if_paused(task)
            task.current_file = item.filename
            try:
                if item.review_status == REVIEW_STATUS_HUMAN_CONFIRMED:
                    task.skipped_count += 1
                    task.error_message = "已跳过人工确认照片，人工字段不会被覆盖。"
                    continue
                analysis = analyze_photo(item.path, item)
                apply_analysis_to_item(item, analysis)
                task.success_count += 1
            except Exception as exc:
                task.failed_count += 1
                task.error_message = str(exc)
                _write_background_log(f"分析失败：{item.path} {exc}\n{traceback.format_exc()}")
            finally:
                task.completed_count += 1
                self.task_progress.emit(task)
                self.queue_changed.emit(self.tasks)

    def _run_similarity_task(self, task: BackgroundTask) -> None:
        from app.analyzers.similarity_analyzer import calculate_similarity_groups

        def progress(done: int, total: int, filename: str) -> None:
            task.total_count = max(task.total_count, total)
            task.completed_count = min(done, task.total_count)
            task.current_file = filename
            self.task_progress.emit(task)
            self.queue_changed.emit(self.tasks)

        result = calculate_similarity_groups(
            self.items,
            mode=getattr(self.config, "similarity_mode", "standard"),
            progress_callback=progress,
            cancel_callback=lambda: self._stop_requested,
            pause_callback=lambda: self._wait_if_paused(task),
        )
        if task.status == TASK_STATUS_STOPPED:
            return
        task.total_count = len(self.items)
        task.completed_count = len(self.items)
        task.success_count = sum(1 for item in self.items if item.similar_group_id)
        task.skipped_count = len(result.skipped)
        task.failed_count = 0
        task.current_file = f"生成 {len(result.groups)} 个相似组"
        task.error_message = f"相似组 {len(result.groups)}，阈值 {result.threshold}，跳过 {len(result.skipped)} 张"
        self.task_progress.emit(task)
        self.queue_changed.emit(self.tasks)

    def _run_placeholder_task(self, task: BackgroundTask) -> None:
        message = "模型未接入，已跳过" if task.task_type not in {TASK_SIMILAR_PHOTOS} else "相似分析功能待接入，已跳过"
        for item in self.items:
            if self._should_stop(task):
                return
            self._wait_if_paused(task)
            task.current_file = item.filename
            task.completed_count += 1
            task.skipped_count += 1
            task.error_message = message
            self.task_progress.emit(task)
        _write_background_log(f"{task.task_name}：{message}")

    def _thumbnail_dir(self) -> Path:
        if self.project_folder:
            return self.project_folder / "cache" / "thumbnails"
        if self.items:
            return self.items[0].path.parent / "cache" / "thumbnails"
        return PROJECT_ROOT / "cache" / "thumbnails"

    def _wait_if_paused(self, task: BackgroundTask) -> None:
        while self._pause_requested and not self._stop_requested:
            if self._paused_emitted_for_task != task.task_id:
                task.status = TASK_STATUS_PAUSED
                self._paused_emitted_for_task = task.task_id
                self.task_paused.emit(task)
                self.queue_changed.emit(self.tasks)
            QThread.msleep(120)
        if task.status == TASK_STATUS_PAUSED and not self._stop_requested:
            task.status = TASK_STATUS_RUNNING
            self.task_resumed.emit(task)
            self.queue_changed.emit(self.tasks)

    def _should_stop(self, task: BackgroundTask) -> bool:
        if not self._stop_requested:
            return False
        task.status = TASK_STATUS_STOPPED
        task.finished_at = _now_text()
        self.task_stopped.emit(task)
        self.queue_changed.emit(self.tasks)
        _write_background_log(f"任务停止：{task.task_name} 已完成 {task.completed_count}/{task.total_count}")
        return True


class BackgroundTaskManager(QObject):
    task_started = Signal(object)
    task_progress = Signal(object)
    task_paused = Signal(object)
    task_resumed = Signal(object)
    task_stopping = Signal(object)
    task_stopped = Signal(object)
    task_completed = Signal(object)
    task_error = Signal(object)
    queue_changed = Signal(object)
    all_finished = Signal(object)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.tasks: list[BackgroundTask] = []
        self.thread: QThread | None = None
        self.worker: BackgroundTaskWorker | None = None

    def set_tasks(self, tasks: list[BackgroundTask]) -> None:
        self.tasks = tasks
        self.queue_changed.emit(self.tasks)

    def add_tasks(self, tasks: list[BackgroundTask]) -> None:
        self.tasks.extend(tasks)
        self.queue_changed.emit(self.tasks)

    def clear(self) -> None:
        self.tasks = []
        self.queue_changed.emit(self.tasks)

    def start(self, items: list[PhotoItem], config: AppConfig, project_folder: Path | None) -> bool:
        if self.is_running():
            return False
        runnable = [task for task in self.tasks if task.status in {TASK_STATUS_PENDING, TASK_STATUS_STOPPED}]
        if not runnable:
            return False
        self.thread = QThread(self)
        self.worker = BackgroundTaskWorker(runnable, items, config, project_folder)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.task_started.connect(self.task_started)
        self.worker.task_progress.connect(self.task_progress)
        self.worker.task_paused.connect(self.task_paused)
        self.worker.task_resumed.connect(self.task_resumed)
        self.worker.task_stopping.connect(self.task_stopping)
        self.worker.task_stopped.connect(self.task_stopped)
        self.worker.task_completed.connect(self.task_completed)
        self.worker.task_error.connect(self.task_error)
        self.worker.queue_changed.connect(self.queue_changed)
        thread = self.thread
        worker = self.worker
        self.worker.all_finished.connect(self._on_all_finished)
        self.worker.all_finished.connect(self.all_finished)
        self.worker.all_finished.connect(lambda *_args, target_thread=thread: target_thread.quit())
        self.thread.finished.connect(worker.deleteLater)
        self.thread.finished.connect(thread.deleteLater)
        self.thread.finished.connect(self._clear_worker_refs)
        self.thread.start()
        return True

    def pause(self) -> None:
        if self.worker:
            self.worker.pause()

    def resume(self) -> None:
        if self.worker:
            self.worker.resume()

    def stop(self) -> None:
        if self.worker:
            self.worker.stop()
            for task in self.tasks:
                if task.status == TASK_STATUS_RUNNING:
                    task.status = TASK_STATUS_STOPPING
                    self.task_stopping.emit(task)
            self.queue_changed.emit(self.tasks)

    def is_running(self) -> bool:
        active_statuses = {TASK_STATUS_RUNNING, TASK_STATUS_PAUSED, TASK_STATUS_STOPPING}
        if not any(task.status in active_statuses for task in self.tasks):
            return False
        if self.thread is None:
            return False
        try:
            return self.thread.isRunning()
        except RuntimeError:
            self.thread = None
            self.worker = None
            return False

    def is_paused(self) -> bool:
        return any(task.status == TASK_STATUS_PAUSED for task in self.tasks)

    def track_external_task(self, task_type: str, task_name: str, total_count: int) -> BackgroundTask:
        task = BackgroundTask(task_type=task_type, task_name=task_name, total_count=total_count, can_pause=False, can_cancel=True)
        task.status = TASK_STATUS_RUNNING
        task.started_at = _now_text()
        self.tasks.append(task)
        self.task_started.emit(task)
        self.queue_changed.emit(self.tasks)
        _write_background_log(f"外部任务开始：{task.task_name}")
        return task

    def update_external_task(
        self,
        task: BackgroundTask | None,
        completed: int,
        current_file: str = "",
        success: int | None = None,
        skipped: int | None = None,
        failed: int | None = None,
        error_message: str = "",
    ) -> None:
        if task is None:
            return
        task.completed_count = completed
        task.current_file = current_file
        if success is not None:
            task.success_count = success
        if skipped is not None:
            task.skipped_count = skipped
        if failed is not None:
            task.failed_count = failed
        if error_message:
            task.error_message = error_message
        self.task_progress.emit(task)
        self.queue_changed.emit(self.tasks)

    def finish_external_task(
        self,
        task: BackgroundTask | None,
        status: str = TASK_STATUS_COMPLETED,
        success: int = 0,
        skipped: int = 0,
        failed: int = 0,
        error_message: str = "",
    ) -> None:
        if task is None:
            return
        task.status = status
        task.success_count = success
        task.skipped_count = skipped
        task.failed_count = failed
        task.completed_count = task.total_count if status == TASK_STATUS_COMPLETED else task.completed_count
        task.error_message = error_message
        task.finished_at = _now_text()
        if status == TASK_STATUS_COMPLETED:
            self.task_completed.emit(task)
        elif status in {TASK_STATUS_STOPPED, TASK_STATUS_CANCELLED}:
            self.task_stopped.emit(task)
        else:
            self.task_error.emit(task)
        self.queue_changed.emit(self.tasks)
        _write_background_log(f"外部任务结束：{task.task_name} 状态 {status} 成功 {success} 跳过 {skipped} 失败 {failed}")

    @Slot(object)
    def _on_all_finished(self, tasks: list[BackgroundTask]) -> None:
        self.queue_changed.emit(self.tasks)

    @Slot()
    def _clear_worker_refs(self) -> None:
        self.thread = None
        self.worker = None


def _write_background_log(message: str) -> None:
    BACKGROUND_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{_now_text()}] {message}\n"
    BACKGROUND_LOG_PATH.open("a", encoding="utf-8").write(line)
