from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QAbstractListModel, QItemSelectionModel, QModelIndex, QObject, QPoint, QRect, QSize, QSettings, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QFont, QImageReader, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStyle,
    QStackedWidget,
    QStyledItemDelegate,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.analyzers.analyzer_pipeline import analyze_photo as run_analyzer_pipeline
from app.analyzers.analyzer_pipeline import (
    CURRENT_AI_SUGGESTION_VERSION,
    ai_suggestion_status,
    apply_analysis_to_item,
    compact_ai_suggestion_text,
    module_summary_text,
)
from app.analyzers.similarity_analyzer import set_group_status
from app.core.ai_classifier import LocalAIClassifier, apply_ai_result_to_item
from app.core.aesthetic_library import add_aesthetic_folder, aesthetic_report, scan_aesthetic_library
from app.core.app_logging import get_logger
from app.core.background_task_manager import (
    BACKGROUND_LOG_PATH,
    TASK_AI_PRECLASSIFY,
    TASK_EXPORT,
    TASK_EXPORT_PREVIEW,
    TASK_EXPRESSION_ANALYSIS,
    TASK_FACE_QUALITY,
    TASK_MODEL_AUTO_GROUP,
    TASK_NAMES,
    TASK_POSE_ANALYSIS,
    TASK_PREFERENCE_STATS,
    TASK_SIMILAR_PHOTOS,
    TASK_STATUS_CANCELLED,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_ERROR,
    TASK_STATUS_PAUSED,
    TASK_STATUS_PENDING,
    TASK_STATUS_RUNNING,
    TASK_STATUS_STOPPED,
    TASK_STATUS_STOPPING,
    TASK_THUMBNAIL_CACHE,
    BackgroundTask,
    BackgroundTaskManager,
    build_background_tasks,
)
from app.core.categories import (
    MANUAL_PENDING,
    MANUAL_RETOUCH,
    MANUAL_CROP,
    MANUAL_PORTFOLIO,
    MANUAL_PRACTICE,
    MANUAL_REJECTED,
    MANUAL_SELECTED,
    PRIMARY_DUPLICATE,
    PRIMARY_REVIEW,
)
from app.core.config import AppConfig, load_config, update_config
from app.core.gpu import detect_ai_environment
from app.core.model_manager import ModelManager, gpu_memory_text
from app.core.model_diagnostics import auto_tune_config, gpu_low_usage_explanation, model_profile_advice, run_model_self_test
from app.core.mvp_exporter import ExportOptions, build_export_plan, build_export_summary, export_photos_with_result, write_current_report
from app.core.mvp_loader import import_photo_folder, preview_pixmap_source_size
from app.core.mvp_models import PhotoItem
from app.core.preference_learning import (
    clear_preference_model,
    predict_preference,
    preference_report,
    preference_summary,
    record_preference_sample,
    train_preference_model,
)
from app.core.project_state import ProjectState, load_project_state, save_project_state
from app.core.review_config import (
    get_delivery_uses,
    get_issue_tags,
    get_photo_types,
    get_quality_ratings,
    get_subtypes,
    validate_review_fields,
)
from app.core.review_models import (
    REVIEW_STATUS_HUMAN_CONFIRMED,
    REVIEW_STATUS_NEEDS_REVIEW,
    REVIEW_STATUS_UNREVIEWED,
)
from app.database.db import PhotoRepository
from app.ui.shortcut_manager import (
    DELIVERY_USE_SHORTCUTS,
    ISSUE_TAG_SHORTCUTS,
    PHOTO_TYPE_SHORTCUTS,
    SHORTCUT_HELP_TEXT,
    SHORTCUT_HINT_TEXT,
    ShortcutManager,
)
from app.ui.similar_group_panel import SimilarGroupPanel


THUMBNAIL_SIZES = {
    "小": (120, 90),
    "中": (160, 120),
    "大": (220, 165),
}

FILTERS = [
    "全部",
    "人工：精修候选",
    "人工：可交付",
    "人工：备选",
    "人工：重复",
    "人工：废片",
    "人工：作品集候选",
    "人工：练习片",
    "人工：待复核",
    "AI：低置信度",
    "AI：相似候选组",
    "AI业务：婚纱写真",
    "AI业务：个人写真",
    "AI业务：情侣/双人写真",
    "AI业务：家庭合影",
    "AI业务：商务形象照",
    "AI业务：会议/活动照",
    "AI业务：无法判断",
]

SORTS = ["文件名", "拍摄时间", "AI业务建议", "人工确认", "置信度", "相似组"]


def _quality_rating_map() -> dict[str, str]:
    return {row["code"]: row["display"] for row in get_quality_ratings()}


def _quality_rating_text(code: str) -> str:
    if not code:
        return "-"
    return _quality_rating_map().get(code, code)


def _join_values(values: list[str] | tuple[str, ...] | None) -> str:
    return "、".join(value for value in (values or []) if value) or "-"


def _review_type_text(item: PhotoItem) -> str:
    if item.photo_type and item.subtype:
        return f"{item.photo_type} / {item.subtype}"
    if item.photo_type:
        return item.photo_type
    if item.ai_primary_category:
        secondary = f" / {item.ai_secondary_category}" if item.ai_secondary_category else ""
        return f"AI建议：{item.ai_primary_category}{secondary}"
    return "-"


def _review_status_text(item: PhotoItem) -> str:
    return item.review_status or REVIEW_STATUS_UNREVIEWED


def _safe_int_score(value: float | int | str | None) -> int:
    try:
        return max(0, min(100, int(float(value or 0))))
    except (TypeError, ValueError):
        return 0


class ImportWorker(QObject):
    progress = Signal(str, int, int, str)
    log = Signal(str)
    finished = Signal(object, object, object)
    failed = Signal(str)

    def __init__(self, folder: Path, config: AppConfig):
        super().__init__()
        self.folder = folder
        self.config = config
        self.cancel_requested = False

    @Slot()
    def run(self) -> None:
        try:
            repo = PhotoRepository(self.folder / "cache" / "photoselect.db")
            saved = repo.load_item_map()
            items, skipped = import_photo_folder(
                self.folder,
                saved,
                self.config,
                progress_callback=lambda stage, done, total, name: self.progress.emit(stage, done, total, name),
                cancel_callback=lambda: self.cancel_requested,
            )
            total = len(items)
            for start in range(0, total, self.config.import_batch_size):
                if self.cancel_requested:
                    break
                batch = items[start : start + self.config.import_batch_size]
                self.progress.emit("写入数据库", min(start + len(batch), total), total, "")
                repo.save_items(batch)
            self.progress.emit("完成", len(items), total, "")
            self.finished.emit(items, skipped, self.cancel_requested)
        except Exception as exc:
            get_logger().exception("导入失败")
            self.failed.emit(str(exc))

    @Slot()
    def cancel(self) -> None:
        self.cancel_requested = True


class AIWorker(QObject):
    progress = Signal(int, int, str)
    log = Signal(str)
    finished = Signal(object, bool, object)
    failed = Signal(str)

    def __init__(self, paths: list[Path], config: AppConfig):
        super().__init__()
        self.paths = paths
        self.config = config
        self.cancel_requested = False

    @Slot()
    def run(self) -> None:
        try:
            classifier = LocalAIClassifier(self.config)
            self.log.emit(f"AI来源：{classifier.device_label}；batch_size={self.config.batch_size}")
            results = classifier.classify_paths(
                self.paths,
                progress_callback=lambda done, total, name: self.progress.emit(done, total, name),
                cancel_callback=lambda: self.cancel_requested,
            )
            self.finished.emit(results, self.cancel_requested, classifier.performance_stats)
        except Exception as exc:
            get_logger().exception("AI分析错误")
            self.failed.emit(str(exc))

    @Slot()
    def cancel(self) -> None:
        self.cancel_requested = True


class ModelDownloadWorker(QObject):
    progress = Signal(str, int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.cancel_requested = False

    @Slot()
    def run(self) -> None:
        try:
            manager = ModelManager(self.config)
            status = manager.download_recommended_models(
                progress_callback=lambda stage, percent, detail: self.progress.emit(stage, percent, detail),
                cancel_callback=lambda: self.cancel_requested,
            )
            self.finished.emit(status)
        except Exception as exc:
            get_logger().exception("模型下载失败")
            self.failed.emit(str(exc))

    @Slot()
    def cancel(self) -> None:
        self.cancel_requested = True


class DiagnosticsWorker(QObject):
    progress = Signal(str, int, str)
    finished = Signal(str, object)
    failed = Signal(str)

    def __init__(self, mode: str, config: AppConfig, paths: list[Path]):
        super().__init__()
        self.mode = mode
        self.config = config
        self.paths = paths
        self.cancel_requested = False

    @Slot()
    def run(self) -> None:
        try:
            if self.mode == "tune":
                self.progress.emit("自动调优", 10, "准备测试 batch_size")
                result = auto_tune_config(self.config, self.paths)
            else:
                self.progress.emit("模型自测", 10, "检查环境和模型")
                result = run_model_self_test(self.config, self.paths)
            self.progress.emit("完成", 100, self.mode)
            self.finished.emit(self.mode, result)
        except Exception as exc:
            get_logger().exception("诊断任务失败")
            self.failed.emit(str(exc))

    @Slot()
    def cancel(self) -> None:
        self.cancel_requested = True


class ExportWorker(QObject):
    progress = Signal(int, int, str, str)
    finished = Signal(str, str, object)
    failed = Signal(str)

    def __init__(
        self,
        folder: Path,
        items: list[PhotoItem],
        config: AppConfig,
        output_dir: Path | None = None,
        options: ExportOptions | None = None,
    ):
        super().__init__()
        self.folder = folder
        self.items = items
        self.config = config
        self.output_dir = output_dir
        self.options = options or ExportOptions()
        self.cancel_requested = False

    @Slot()
    def run(self) -> None:
        try:
            result = export_photos_with_result(
                self.folder,
                self.items,
                self.config,
                self.output_dir,
                progress_callback=lambda done, total, name, target: self.progress.emit(done, total, name, target),
                cancel_callback=lambda: self.cancel_requested,
                options=self.options,
            )
            report = result.csv_path or result.markdown_path or result.log_path
            self.finished.emit(str(result.output_dir), str(report), result)
        except Exception as exc:
            get_logger().exception("导出失败")
            self.failed.emit(str(exc))

    @Slot()
    def cancel(self) -> None:
        self.cancel_requested = True


class PhotoListModel(QAbstractListModel):
    def __init__(self) -> None:
        super().__init__()
        self.items: list[PhotoItem] = []
        self.filtered_indexes: list[int] = []
        self.filter_name = "全部"
        self.sort_name = "文件名"

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self.filtered_indexes)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        item = self.item_at_row(index.row())
        if role == Qt.DisplayRole:
            return item.filename
        if role == Qt.UserRole:
            return item
        return None

    def item_at_row(self, row: int) -> PhotoItem:
        return self.items[self.filtered_indexes[row]]

    def source_index_at_row(self, row: int) -> int:
        return self.filtered_indexes[row]

    def row_for_source_index(self, source_index: int) -> int:
        try:
            return self.filtered_indexes.index(source_index)
        except ValueError:
            return -1

    def set_items(self, items: list[PhotoItem]) -> None:
        self.beginResetModel()
        self.items = items
        self._rebuild()
        self.endResetModel()

    def refresh(self) -> None:
        self.beginResetModel()
        self._rebuild()
        self.endResetModel()

    def _rebuild(self) -> None:
        indexes = [index for index, item in enumerate(self.items) if self._matches_filter(item)]
        indexes.sort(key=lambda idx: self._sort_key(self.items[idx]))
        self.filtered_indexes = indexes

    def _matches_filter(self, item: PhotoItem) -> bool:
        name = self.filter_name
        if name == "全部":
            return True
        legacy_name = name
        if name.startswith("人工："):
            legacy_name = name.removeprefix("人工：")
        elif name.startswith("AI业务："):
            return item.ai_primary_category == name.removeprefix("AI业务：")
        if legacy_name == "精修候选":
            return item.manual_category == MANUAL_SELECTED
        if legacy_name == "可交付":
            return item.manual_category == MANUAL_RETOUCH
        if legacy_name == "备选":
            return item.manual_category == MANUAL_PENDING
        if legacy_name == "重复":
            return item.manual_category == MANUAL_CROP or item.similar_group_status == "duplicate"
        if legacy_name == "废片":
            return item.manual_category == MANUAL_REJECTED
        if legacy_name == "作品集候选":
            return item.manual_category == MANUAL_PORTFOLIO
        if legacy_name == "练习片":
            return item.manual_category == MANUAL_PRACTICE
        if legacy_name in {"待复核", "人工复核"}:
            return item.review_status == REVIEW_STATUS_NEEDS_REVIEW
        if name == "AI：低置信度":
            return bool(item.ai_primary_category) and item.ai_confidence < 0.65
        if name == "AI：相似候选组":
            return bool(item.auto_group_id)
        return item.ai_primary_category == name or item.final_category.startswith(name)

    def _sort_key(self, item: PhotoItem):
        if self.sort_name == "拍摄时间":
            return item.taken_at or item.filename
        if self.sort_name in {"AI分类", "AI业务建议"}:
            return (item.ai_primary_category, item.ai_secondary_category, item.filename)
        if self.sort_name in {"手动分类", "人工确认"}:
            return (item.manual_category, item.filename)
        if self.sort_name == "置信度":
            return (-item.ai_confidence, item.filename)
        if self.sort_name == "相似组":
            return (item.similar_group_id or "999", item.filename)
        return item.filename.lower()


class ThumbnailDelegate(QStyledItemDelegate):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.thumb_size = QSize(160, 120)
        self.mode = "contain"
        self._pixmap_cache: dict[str, QPixmap] = {}

    def set_thumbnail_options(self, size: tuple[int, int], mode: str) -> None:
        self.thumb_size = QSize(*size)
        self.mode = mode
        self._pixmap_cache.clear()

    def sizeHint(self, option, index) -> QSize:
        return QSize(self.thumb_size.width() + 24, self.thumb_size.height() + 120)

    def paint(self, painter: QPainter, option, index) -> None:
        item: PhotoItem = index.data(Qt.UserRole)
        painter.save()

        selected = bool(option.state & QStyle.State_Selected)
        rect = option.rect.adjusted(6, 6, -6, -6)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor("#2f3540") if selected else QColor("#252a33"))
        painter.setPen(QPen(QColor("#4f8cff") if selected else QColor("#363d49"), 2 if selected else 1))
        painter.drawRoundedRect(rect, 8, 8)

        image_rect = QRect(rect.left() + 10, rect.top() + 10, self.thumb_size.width(), self.thumb_size.height())
        painter.fillRect(image_rect, QColor("#15181e"))
        pixmap = self._load_pixmap(item)
        if not pixmap.isNull():
            if self.mode == "cover":
                scaled = pixmap.scaled(image_rect.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                source_x = max(0, (scaled.width() - image_rect.width()) // 2)
                source_y = max(0, (scaled.height() - image_rect.height()) // 2)
                painter.drawPixmap(image_rect, scaled, QRect(source_x, source_y, image_rect.width(), image_rect.height()))
            else:
                scaled = pixmap.scaled(image_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                target = QRect(QPoint(0, 0), scaled.size())
                target.moveCenter(image_rect.center())
                painter.drawPixmap(target, scaled)

        self._draw_badge(painter, image_rect, item)
        text_x = rect.left() + 10
        text_y = image_rect.bottom() + 10
        painter.setPen(QColor("#e8ecf3"))
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRect(text_x, text_y, rect.width() - 20, 18), Qt.TextSingleLine, item.filename)
        font.setBold(False)
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QColor("#b7c0cf"))
        rating = _quality_rating_text(item.quality_rating)
        type_text = _review_type_text(item)
        status = _review_status_text(item)
        group_status = {
            "best": "最佳",
            "backup": "备选",
            "duplicate": "淘汰",
            "review": "复核",
            "rejected": "淘汰",
        }.get(item.similar_group_status, "")
        group = f"相似组：{item.similar_group_id}{'｜' + group_status if group_status else ''}" if item.similar_group_id else "相似组：-"
        painter.drawText(QRect(text_x, text_y + 20, rect.width() - 20, 18), Qt.TextSingleLine, f"{rating} · {status}")
        painter.drawText(QRect(text_x, text_y + 40, rect.width() - 20, 18), Qt.TextSingleLine, type_text)
        painter.drawText(QRect(text_x, text_y + 60, rect.width() - 20, 18), Qt.TextSingleLine, group)
        painter.restore()

    def _load_pixmap(self, item: PhotoItem) -> QPixmap:
        path = str(item.thumbnail_path or item.path)
        pixmap = self._pixmap_cache.get(path)
        if pixmap is None:
            pixmap = QPixmap(path)
            self._pixmap_cache[path] = pixmap
            if len(self._pixmap_cache) > 400:
                self._pixmap_cache.clear()
        return pixmap

    def _draw_badge(self, painter: QPainter, image_rect: QRect, item: PhotoItem) -> None:
        if item.manual_category == MANUAL_SELECTED:
            text, color = "精修", QColor("#16a34a")
        elif item.manual_category == MANUAL_RETOUCH:
            text, color = "交付", QColor("#22c55e")
        elif item.manual_category == MANUAL_CROP:
            text, color = "重复", QColor("#f97316")
        elif item.manual_category == MANUAL_REJECTED:
            text, color = "废片", QColor("#dc2626")
        elif item.manual_category == MANUAL_PORTFOLIO:
            text, color = "作品", QColor("#a855f7")
        elif item.manual_category == MANUAL_PRACTICE:
            text, color = "练习", QColor("#64748b")
        elif item.similar_group_status == "best" or item.best_in_group:
            text, color = "最佳", QColor("#22c55e")
        elif item.similar_group_status == "duplicate":
            text, color = "重复", QColor("#f97316")
        elif item.similar_group_status == "review":
            text, color = "复核", QColor("#facc15")
        elif item.final_category.startswith(PRIMARY_REVIEW):
            text, color = "复核", QColor("#facc15")
        else:
            text, color = "备选", QColor("#2563eb")

        badge = QRect(image_rect.left() + 8, image_rect.bottom() - 28, 48, 20)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(badge, 5, 5)
        painter.setPen(QColor("#111827") if text == "复核" else QColor("#ffffff"))
        font = painter.font()
        font.setBold(True)
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(badge, Qt.AlignCenter, text)

class CompareTile(QFrame):
    clicked = Signal(int)
    mark_requested = Signal(int, str)

    def __init__(self, slot_index: int):
        super().__init__()
        self.slot_index = slot_index
        self.source_index = -1
        self.setObjectName("CompareTile")
        self.setFrameShape(QFrame.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.image_label = QLabel("未选择")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(220)
        self.image_label.setObjectName("CompareImage")
        self.info_label = QLabel("-")
        self.info_label.setWordWrap(True)
        button_row = QHBoxLayout()
        self.keep_button = QPushButton("精修")
        self.pending_button = QPushButton("备选")
        self.reject_button = QPushButton("废片")
        button_row.addWidget(self.keep_button)
        button_row.addWidget(self.pending_button)
        button_row.addWidget(self.reject_button)
        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.info_label)
        layout.addLayout(button_row)
        self.keep_button.clicked.connect(lambda: self.mark_requested.emit(self.source_index, MANUAL_SELECTED))
        self.pending_button.clicked.connect(lambda: self.mark_requested.emit(self.source_index, MANUAL_PENDING))
        self.reject_button.clicked.connect(lambda: self.mark_requested.emit(self.source_index, MANUAL_REJECTED))

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self.source_index)
        super().mousePressEvent(event)

    def set_item(self, item: PhotoItem | None, source_index: int, pixmap: QPixmap | None, focused: bool) -> None:
        self.source_index = source_index
        self.setProperty("focused", focused)
        self.style().unpolish(self)
        self.style().polish(self)
        if item is None:
            self.image_label.setText("未选择")
            self.image_label.setPixmap(QPixmap())
            self.info_label.setText("-")
            return
        if pixmap:
            self.image_label.setPixmap(pixmap)
        self.info_label.setText(
            f"{item.filename}\n{item.resolution_text} · {item.file_size_mb:.2f} MB\n"
            f"手动：{item.manual_category}｜AI：{item.ai_category_path or '-'}｜置信度：{item.ai_confidence_text}\n"
            f"相似组：{item.similar_group_id or '-'}｜质量：{', '.join(item.ai_quality_tags) or '-'}"
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.logger = get_logger()
        self.config = load_config()
        self.model_manager = ModelManager(self.config)
        self.ai_env = detect_ai_environment(self.config.use_gpu)
        self.background_task_manager = BackgroundTaskManager(self)

        self.setWindowTitle("PhotoSelect AI Assistant")
        self.resize(1500, 900)
        self.setMinimumSize(1100, 720)
        self.selected_folder: Path | None = None
        self.export_output_dir: Path | None = None
        self.last_report_path: Path | None = None
        self.repository: PhotoRepository | None = None
        self.items: list[PhotoItem] = []
        self.current_source_index = -1
        self.focus_source_index = -1
        self.current_preview_pixmap: QPixmap | None = None
        self.zoom_factor = 0.0
        self.compare_zoom_factor = 1.0

        self.import_thread: QThread | None = None
        self.import_worker: ImportWorker | None = None
        self.ai_thread: QThread | None = None
        self.ai_worker: AIWorker | None = None
        self.export_thread: QThread | None = None
        self.export_worker: ExportWorker | None = None
        self.model_download_thread: QThread | None = None
        self.model_download_worker: ModelDownloadWorker | None = None
        self.diagnostics_thread: QThread | None = None
        self.diagnostics_worker: DiagnosticsWorker | None = None
        self._loading_details = False
        self._review_controls_dirty = False
        self.dirty = False
        self.import_started_at: datetime | None = None
        self.current_ai_total = 0
        self.last_saved_text = "未保存"
        self.last_ai_performance: dict = {}
        self.idle_tasks_paused = False
        self.active_export_task: BackgroundTask | None = None
        self.background_ask_prompted = False
        self.shortcut_manager: ShortcutManager | None = None
        self._last_preference_record_key = ""

        self.model = PhotoListModel()
        self.left_delegate = ThumbnailDelegate(self)
        self.grid_delegate = ThumbnailDelegate(self)
        self.autosave_timer = QTimer(self)
        self.delayed_save_timer = QTimer(self)
        self.idle_timer = QTimer(self)
        self.delayed_save_timer.setSingleShot(True)

        self._build_ui()
        self._sync_config_controls()
        self._connect_signals()
        self._install_shortcuts()
        self._apply_style()
        self._apply_thumbnail_options()
        self._refresh_ai_status()
        self.autosave_timer.setInterval(self.config.autosave_interval_seconds * 1000)
        self.autosave_timer.timeout.connect(lambda: self.save_all(auto=True))
        self.delayed_save_timer.timeout.connect(lambda: self.save_all(auto=True))
        self.idle_timer.setInterval(15000)
        self.idle_timer.timeout.connect(self.run_idle_task_tick)
        self.autosave_timer.start()
        self.idle_timer.start()
        self._refresh_background_buttons()
        self.mark_project_running()
        QTimer.singleShot(0, self.maybe_restore_project)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self.workflow_context_labels: dict[str, QLabel] = {}
        workflow_panel = self._build_workflow_panel()
        self._extract_workflow_pages(workflow_panel)
        self.similar_group_panel.setObjectName("SimilarGroupPage")
        root.addWidget(self._build_top_status_bar())

        self.body_splitter = QSplitter(Qt.Horizontal)
        self.body_splitter.setObjectName("BodyMainSplitter")
        self.body_splitter.addWidget(self._build_sidebar_navigation())
        self.main_stack = QStackedWidget()
        self.main_stack.setObjectName("MainContentStack")
        self.review_workspace = self._build_review_workspace()
        self.review_board = self._build_review_board()
        self.page_widgets = {
            "project": self.project_import_page,
            "ai": self.ai_analysis_page,
            "review": self.review_board,
            "export": self.export_review_page,
        }
        for page in ["project", "ai", "review", "export"]:
            self.main_stack.addWidget(self.page_widgets[page])
        self.current_main_section = "review"
        self.current_review_subsection = "photo"
        self.body_splitter.addWidget(self.main_stack)
        self.body_splitter.setStretchFactor(0, 0)
        self.body_splitter.setStretchFactor(1, 1)
        self.body_splitter.setSizes([180, 1320])
        root.addWidget(self.body_splitter, 1)

        self.setCentralWidget(central)
        self.cancel_import_button.setEnabled(False)
        self.cancel_ai_button.setEnabled(False)
        self.cancel_model_download_button.setEnabled(False)
        self.cancel_export_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.export_csv_button.setEnabled(False)
        self.open_report_button.setEnabled(False)
        self.current_ai_button.setEnabled(False)
        self.ai_analyze_button.setEnabled(False)
        self.model_auto_group_button.setEnabled(False)
        self.reanalyze_unconfirmed_button.setEnabled(False)
        self.clear_stale_ai_button.setEnabled(False)
        self.reanalyze_current_suggestion_button.setEnabled(False)
        self.show_group_button.setEnabled(False)
        self.recommended_keep_button.setEnabled(False)
        self.group_backup_quick_button.setEnabled(False)
        self.group_duplicate_quick_button.setEnabled(False)
        self._restore_layout_state()
        self._refresh_workflow_contexts()

    def _extract_workflow_pages(self, workflow_tabs: QTabWidget) -> None:
        pages: list[QWidget] = []
        while workflow_tabs.count():
            page = workflow_tabs.widget(0)
            workflow_tabs.removeTab(0)
            pages.append(page)
        self.project_import_page = self._wrap_page_scroll(
            self._build_import_settings_page(pages[0], pages[4]),
            "ProjectImportPage",
        )
        self.ai_analysis_page = self._wrap_page_scroll(
            self._build_ai_analysis_results_page(pages[1]),
            "AIAnalysisPage",
        )
        self.review_actions_page = pages[2]
        self.review_actions_page.setParent(self)
        self.style_training_page = self._wrap_page_scroll(pages[3], "StyleTrainingPage")
        self.export_review_page = self._wrap_page_scroll(
            self._build_export_organize_page(self.export_page),
            "ExportReviewPage",
        )

    def _build_page_shell(self, title: str, hint: str, context_key: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)
        title_label = QLabel(title)
        title_label.setObjectName("WorkflowPageTitle")
        title_label.setWordWrap(True)
        hint_label = QLabel(hint)
        hint_label.setObjectName("WorkflowPageHint")
        hint_label.setWordWrap(True)
        context = QLabel(self._workflow_context_text())
        context.setObjectName("WorkflowContext")
        context.setWordWrap(True)
        self.workflow_context_labels[context_key] = context
        layout.addWidget(title_label)
        layout.addWidget(hint_label)
        layout.addWidget(context)
        return page, layout

    def _build_import_settings_page(self, project_widget: QWidget, settings_widget: QWidget) -> QWidget:
        page, layout = self._build_page_shell(
            "1. 导入与设置 · 项目中心",
            "从这里建立项目、确认本批照片业务目标，并查看整体进度。没有导入照片时，先按项目建立向导走一遍。",
            "project",
        )

        self.project_landing_container = QWidget()
        landing_layout = QVBoxLayout(self.project_landing_container)
        landing_layout.setContentsMargins(0, 0, 0, 0)
        landing_layout.setSpacing(14)

        hero_card, hero_layout = self._make_card("", "HeroCard")
        hero_title = QLabel("PhotoSelect AI Assistant")
        hero_title.setObjectName("HeroTitle")
        hero_subtitle = QLabel("AI辅助摄影选片、相似候选组、人工审核与安全导出")
        hero_subtitle.setObjectName("HeroSubtitle")
        hero_subtitle.setWordWrap(True)
        self.project_empty_state_label = QLabel("建立一个选片项目后，软件会保留照片上下文，并引导你完成 AI 分析、人工审核和导出。")
        self.project_empty_state_label.setWordWrap(True)
        self.project_empty_state_label.setObjectName("EmptyStateBody")
        hero_buttons = QHBoxLayout()
        self.hero_new_project_button = self._make_button("新建选片项目", "选择照片文件夹并建立新项目。")
        self.hero_open_project_button = self._make_button("打开已有项目", "打开一个已有照片文件夹或项目目录。")
        self.hero_continue_project_button = self._make_button("继续上次项目", "恢复上次筛片项目。")
        self.hero_new_project_button.setObjectName("PrimaryButton")
        self.hero_open_project_button.setObjectName("SecondaryButton")
        self.hero_continue_project_button.setObjectName("SecondaryButton")
        hero_buttons.addWidget(self.hero_new_project_button)
        hero_buttons.addWidget(self.hero_open_project_button)
        hero_buttons.addWidget(self.hero_continue_project_button)
        hero_buttons.addStretch(1)
        hero_layout.addWidget(hero_title)
        hero_layout.addWidget(hero_subtitle)
        hero_layout.addWidget(self.project_empty_state_label)
        hero_layout.addLayout(hero_buttons)
        landing_layout.addWidget(hero_card)

        action_grid = QGridLayout()
        action_grid.setSpacing(12)
        for index, (title, body) in enumerate(
            [
                ("导入照片", "选择照片文件夹并建立项目。原图只读取，不删除。"),
                ("AI 分析", "生成评分建议、业务类型和 AI 相似候选组。"),
                ("人工审核与导出", "人工确认后安全复制/导出整理结果。"),
            ]
        ):
            card, card_layout = self._make_card(title, "ActionCard")
            body_label = QLabel(body)
            body_label.setObjectName("MetricNote")
            body_label.setWordWrap(True)
            card_layout.addWidget(body_label)
            action_grid.addWidget(card, 0, index)
        landing_layout.addLayout(action_grid)

        self.project_safety_card, safety_layout = self._make_card("安全边界", "SafetyNoticeCard")
        safety_label = QLabel("默认不移动源文件｜AI结果只是建议｜人工确认后才导出")
        safety_label.setObjectName("SafetyNoticeText")
        safety_label.setWordWrap(True)
        safety_layout.addWidget(safety_label)
        landing_layout.addWidget(self.project_safety_card)
        layout.addWidget(self.project_landing_container)

        self.project_dashboard_card, dashboard_layout = self._make_card("项目 Dashboard", "DashboardCard")
        dashboard_grid = QGridLayout()
        self.dashboard_total_card, self.dashboard_total_value, self.dashboard_total_note = self._make_metric_card("照片总数", "0", "导入后显示")
        self.dashboard_ai_card, self.dashboard_ai_value, self.dashboard_ai_note = self._make_metric_card("AI建议", "0", "已分析照片")
        self.dashboard_manual_card, self.dashboard_manual_value, self.dashboard_manual_note = self._make_metric_card("人工确认", "0", "已确认/待复核")
        self.dashboard_group_card, self.dashboard_group_value, self.dashboard_group_note = self._make_metric_card("AI相似候选组", "0", "仅为候选，不是人工重复")
        self.dashboard_export_card, self.dashboard_export_value, self.dashboard_export_note = self._make_metric_card("导出准备", "0", "可导出用途数量")
        for index, card in enumerate(
            [
                self.dashboard_total_card,
                self.dashboard_ai_card,
                self.dashboard_manual_card,
                self.dashboard_group_card,
                self.dashboard_export_card,
            ]
        ):
            dashboard_grid.addWidget(card, index // 3, index % 3)
        dashboard_layout.addLayout(dashboard_grid)
        self.next_step_card, next_step_layout = self._make_card("下一步建议", "ActionCard")
        self.next_step_label = QLabel("请先导入照片。")
        self.next_step_label.setObjectName("EmptyStateBody")
        self.next_step_label.setWordWrap(True)
        self.next_step_button = self._make_button("新建选片项目", "根据当前状态执行下一步。")
        self.next_step_button.setObjectName("PrimaryButton")
        next_step_layout.addWidget(self.next_step_label)
        next_step_layout.addWidget(self.next_step_button)
        dashboard_layout.addWidget(self.next_step_card)
        layout.addWidget(self.project_dashboard_card)

        self.project_setup_group = QGroupBox("项目基础设置")
        self.project_setup_group.setObjectName("WorkflowCard")
        setup_layout = QGridLayout(self.project_setup_group)
        self.business_style_combo = QComboBox()
        self.business_style_combo.addItems(["婚纱写真", "个人写真", "情侣/双人写真", "家庭合影", "商务形象照", "会议/活动照", "混合/无法判断"])
        self.process_goal_combo = QComboBox()
        self.process_goal_combo.addItems(["快速初筛", "精修候选", "客户选片", "作品集筛选", "相似照片整理"])
        self.model_strength_combo = QComboBox()
        self.model_strength_combo.addItems(["保守：少合并，宁愿多拆", "平衡：默认", "激进：更多合并，适合快速整理"])
        self.model_strength_combo.setCurrentText("平衡：默认")
        self.model_default_label = QLabel(self._model_default_text())
        self.model_default_label.setWordWrap(True)
        setup_cards = [
            ("业务风格", self.business_style_combo),
            ("处理目标", self.process_goal_combo),
            ("模型强度", self.model_strength_combo),
        ]
        for index, (title, widget) in enumerate(setup_cards):
            card, card_layout = self._make_card(title, "ActionCard")
            card_layout.addWidget(widget)
            setup_layout.addWidget(card, 0, index)
        defaults_card, defaults_layout = self._make_card("当前默认参数", "ActionCard")
        defaults_layout.addWidget(self.model_default_label)
        setup_layout.addWidget(defaults_card, 1, 0, 1, 3)
        layout.addWidget(self.project_setup_group)

        self.project_import_card, import_layout = self._make_card("项目操作", "DashboardCard")
        import_layout.addWidget(project_widget)
        layout.addWidget(self.project_import_card)

        self.advanced_settings_group = QGroupBox("高级设置 / 模型 / 调试")
        self.advanced_settings_group.setObjectName("WorkflowCard")
        self.advanced_settings_group.setCheckable(True)
        self.advanced_settings_group.setChecked(False)
        advanced_layout = QVBoxLayout(self.advanced_settings_group)
        advanced_layout.addWidget(settings_widget)
        self.advanced_settings_group.toggled.connect(settings_widget.setVisible)
        settings_widget.setVisible(False)
        layout.addWidget(self.advanced_settings_group)
        layout.addStretch(1)
        return page

    def _build_ai_analysis_results_page(self, ai_widget: QWidget) -> QWidget:
        page, layout = self._build_page_shell(
            "2. AI分析结果",
            "这里运行 AI 预分析和 AI 相似候选组。AI 结果只是建议，不等于人工确认结果。",
            "ai",
        )
        self.ai_action_group = QGroupBox("AI分析入口")
        action_layout = QVBoxLayout(self.ai_action_group)
        action_layout.addWidget(ai_widget)
        layout.addWidget(self.ai_action_group)

        self.ai_overview_group = QGroupBox("结果总览：AI建议，不是人工结论")
        overview_layout = QGridLayout(self.ai_overview_group)
        self.ai_progress_summary_label = QLabel("进度：尚未运行 AI 分析。")
        self.ai_score_overview_label = QLabel("A. AI评分建议：暂无")
        self.ai_group_overview_label = QLabel("B. AI相似候选组：暂无")
        self.ai_business_overview_label = QLabel("C. 业务类型建议：暂无")
        for label in [
            self.ai_progress_summary_label,
            self.ai_score_overview_label,
            self.ai_group_overview_label,
            self.ai_business_overview_label,
        ]:
            label.setWordWrap(True)
        overview_layout.addWidget(self.ai_progress_summary_label, 0, 0, 1, 2)
        overview_layout.addWidget(self.ai_score_overview_label, 1, 0)
        overview_layout.addWidget(self.ai_group_overview_label, 1, 1)
        overview_layout.addWidget(self.ai_business_overview_label, 2, 0, 1, 2)
        layout.addWidget(self.ai_overview_group)
        self.ai_empty_state_card, ai_empty_layout = self._make_card("还没有 AI 分析结果", "EmptyStateCard")
        ai_empty_body = QLabel("导入照片后，先运行 AI 分析。AI 会生成评分建议、业务类型和相似候选组，但不会自动变成人工确认。")
        ai_empty_body.setObjectName("EmptyStateBody")
        ai_empty_body.setWordWrap(True)
        self.ai_empty_start_button = self._make_button("开始 AI 分析", "批量分析未人工确认照片。")
        self.ai_empty_start_button.setObjectName("PrimaryButton")
        self.ai_empty_group_button = self._make_button("生成 AI 相似候选组", "使用模型 embedding 生成相似候选组建议。")
        self.ai_empty_group_button.setObjectName("SecondaryButton")
        ai_empty_buttons = QHBoxLayout()
        ai_empty_buttons.addWidget(self.ai_empty_start_button)
        ai_empty_buttons.addWidget(self.ai_empty_group_button)
        ai_empty_buttons.addStretch(1)
        ai_empty_layout.addWidget(ai_empty_body)
        ai_empty_layout.addLayout(ai_empty_buttons)
        layout.addWidget(self.ai_empty_state_card)
        layout.addStretch(1)
        return page

    def _build_export_organize_page(self, export_widget: QWidget) -> QWidget:
        page, layout = self._build_page_shell(
            "4. 导出与整理",
            "导出默认只复制到新文件夹，并生成 CSV / Markdown 复盘报告。源文件移动默认关闭，且本版本不自动删除任何原图。",
            "export",
        )
        self.export_safety_group = QGroupBox("源文件安全")
        safety_layout = QVBoxLayout(self.export_safety_group)
        self.export_copy_safety_label = QLabel("推荐：复制到新文件夹 + 生成报告。人工“重复 / 废片 / 不导出”只影响导出清单，不会删除原图。")
        self.export_copy_safety_label.setWordWrap(True)
        self.move_source_checkbox = QCheckBox("高级：移动源文件分类（默认关闭，需要二次确认；当前版本不执行自动移动）")
        self.move_source_checkbox.setChecked(False)
        self.move_source_checkbox.setEnabled(False)
        self.move_source_checkbox.setToolTip("为了保护原图，v0.6.0 仅保留入口说明；如未来启用，必须先生成 dry-run 计划并二次确认。")
        self.lr_c1_list_checkbox = QCheckBox("生成 Lightroom / Capture One 可用清单（随 CSV 报告记录，后续可扩展专用格式）")
        self.lr_c1_list_checkbox.setChecked(True)
        safety_layout.addWidget(self.export_copy_safety_label)
        safety_layout.addWidget(self.lr_c1_list_checkbox)
        safety_layout.addWidget(self.move_source_checkbox)
        layout.addWidget(self.export_safety_group)
        self.export_empty_state_card, export_empty_layout = self._make_card("还没有可导出的人工确认结果", "EmptyStateCard")
        export_empty_body = QLabel("请先进入审核与修正，确认精修候选、客户可选、直接交付、仅留档、重复或废片等人工结果。")
        export_empty_body.setObjectName("EmptyStateBody")
        export_empty_body.setWordWrap(True)
        self.export_empty_review_button = self._make_button("进入审核与修正", "进入人工审核工作台。")
        self.export_empty_review_button.setObjectName("PrimaryButton")
        export_empty_layout.addWidget(export_empty_body)
        export_empty_layout.addWidget(self.export_empty_review_button)
        layout.addWidget(self.export_empty_state_card)
        self.export_controls_widget = export_widget
        layout.addWidget(export_widget)
        layout.addStretch(1)
        return page

    def _wrap_page_scroll(self, widget: QWidget, object_name: str) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setObjectName(object_name)
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def _make_card(self, title: str = "", object_name: str = "DashboardCard") -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName(object_name)
        card.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        if title:
            label = QLabel(title)
            label.setObjectName("CardTitle")
            label.setWordWrap(True)
            layout.addWidget(label)
        return card, layout

    def _make_metric_card(self, title: str, value: str = "0", note: str = "") -> tuple[QFrame, QLabel, QLabel]:
        card, layout = self._make_card(title, "MetricCard")
        value_label = QLabel(value)
        value_label.setObjectName("MetricValue")
        value_label.setWordWrap(True)
        note_label = QLabel(note)
        note_label.setObjectName("MetricNote")
        note_label.setWordWrap(True)
        layout.addWidget(value_label)
        layout.addWidget(note_label)
        layout.addStretch(1)
        return card, value_label, note_label

    def _build_empty_state_card(self, title: str, body: str, action_text: str = "") -> QFrame:
        card, layout = self._make_card(title, "EmptyStateCard")
        body_label = QLabel(body)
        body_label.setObjectName("EmptyStateBody")
        body_label.setWordWrap(True)
        layout.addWidget(body_label)
        if action_text:
            action_label = QLabel(action_text)
            action_label.setObjectName("EmptyStateAction")
            action_label.setWordWrap(True)
            layout.addWidget(action_label)
        return card

    def _build_top_status_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("TopStatusBar")
        bar.setFixedHeight(44)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(10)
        self.project_status_label = QLabel(self._project_status_text())
        self.project_status_label.setObjectName("TopProjectLabel")
        self.project_status_label.setWordWrap(False)
        self.total_status_label = QLabel("总照片数：0")
        self.current_status_label = QLabel("当前：第 0 张 / 共 0 张")
        self.background_compact_label = QLabel("后台：空闲｜0/0｜成功 0｜失败 0")
        self.background_compact_label.setObjectName("CompactTaskStatus")
        self.saved_status_label = QLabel("未保存")
        self.gpu_status_label = QLabel("AI状态：可用" if self.ai_env.ai_ready else "AI状态：未就绪")
        self.gpu_status_label.setVisible(False)
        self.top_background_button = self._make_button("后台任务", "打开后台任务窗口，查看进度并控制暂停、继续、停止。")
        self.top_settings_button = self._make_button("设置", "打开设置 / 高级调试页面。")
        layout.addWidget(self.project_status_label, 3)
        layout.addWidget(self.total_status_label)
        layout.addWidget(self.current_status_label)
        layout.addWidget(self.background_compact_label, 2)
        layout.addWidget(self.saved_status_label)
        layout.addStretch(1)
        layout.addWidget(self.top_background_button)
        layout.addWidget(self.top_settings_button)
        self.top_background_button.clicked.connect(self.show_background_tasks_dialog)
        self.top_settings_button.clicked.connect(lambda: self._activate_workflow_section("settings"))
        return bar

    def _project_status_text(self) -> str:
        if not self.selected_folder:
            return "项目：-"
        path = Path(self.selected_folder)
        parent = path.parent.name
        name = path.name
        display = f"{parent}/{name}" if parent else name
        if len(display) > 42:
            display = f".../{name[:36]}" if len(name) > 36 else f".../{name}"
        return f"项目：{display}"

    def _model_default_text(self) -> str:
        device = "CUDA" if self.ai_env.cuda_available else "CPU / fallback"
        return (
            f"OpenCLIP/CUDA：{device}；"
            f"AI相似候选阈值：{self.config.embedding_similarity_threshold:.2f}；"
            f"分组策略：{self.config.grouping_strategy}；"
            "缓存：启用。"
        )

    def _workflow_context_text(self) -> str:
        total = len(getattr(self, "items", []))
        if total <= 0:
            return "当前照片：-｜序号：0/0｜AI建议：-｜人工确认：-｜系统状态：尚未导入照片"
        if 0 <= self.current_source_index < total:
            item = self.items[self.current_source_index]
            current = self.current_source_index + 1
            ai_text = item.final_recommendation or item.ai_category_path or item.ai_primary_category or "-"
            human_text = " / ".join(
                value
                for value in [
                    _quality_rating_text(item.quality_rating) if item.quality_rating else "",
                    _join_values(item.delivery_use) if item.delivery_use else "",
                    _review_status_text(item),
                ]
                if value
            ) or "-"
            return (
                f"当前照片：{item.filename}｜序号：{current}/{total}｜"
                f"AI建议：{ai_text}｜人工确认：{human_text}｜系统状态：{_review_status_text(item)}"
            )
        return f"当前照片：未选择｜序号：0/{total}｜AI建议：-｜人工确认：-｜系统状态：等待选择照片"

    def _refresh_workflow_contexts(self) -> None:
        text = self._workflow_context_text()
        for label in getattr(self, "workflow_context_labels", {}).values():
            label.setText(text)
        if hasattr(self, "model_default_label"):
            self.model_default_label.setText(self._model_default_text())
        self.update_project_dashboard()

    def update_project_dashboard(self) -> None:
        if not hasattr(self, "dashboard_total_value"):
            return
        total = len(self.items)
        analyzed = sum(1 for item in self.items if item.ai_suggestion or item.ai_primary_category or item.ai_confidence)
        human_confirmed = sum(1 for item in self.items if item.review_status == REVIEW_STATUS_HUMAN_CONFIRMED)
        needs_review = sum(1 for item in self.items if item.review_status == REVIEW_STATUS_NEEDS_REVIEW)
        auto_group_ids = {item.auto_group_id for item in self.items if item.auto_group_id}
        export_ready = sum(1 for item in self.items if item.delivery_use and "不导出" not in item.delivery_use)
        has_project = total > 0
        if hasattr(self, "project_landing_container"):
            self.project_landing_container.setVisible(not has_project)
        if hasattr(self, "project_dashboard_card"):
            self.project_dashboard_card.setVisible(has_project)
        if hasattr(self, "project_setup_group"):
            self.project_setup_group.setVisible(has_project)
        if hasattr(self, "project_import_card"):
            self.project_import_card.setVisible(has_project)
        self.dashboard_total_value.setText(str(total))
        self.dashboard_total_note.setText(str(self.selected_folder or "尚未选择项目"))
        self.dashboard_ai_value.setText(f"{analyzed}/{total}" if total else "0")
        self.dashboard_ai_note.setText("AI建议，仅供参考")
        self.dashboard_manual_value.setText(str(human_confirmed))
        self.dashboard_manual_note.setText(f"待复核 {needs_review} 张")
        self.dashboard_group_value.setText(str(len(auto_group_ids)))
        self.dashboard_group_note.setText("AI相似候选组，需人工确认")
        self.dashboard_export_value.setText(str(export_ready))
        self.dashboard_export_note.setText("按人工交付用途统计")
        if hasattr(self, "next_step_label"):
            if not has_project:
                self.next_step_label.setText("请先建立一个选片项目。")
                self.next_step_button.setText("新建选片项目")
            elif analyzed <= 0:
                self.next_step_label.setText("照片已导入。建议先运行 AI 分析，生成评分建议、业务类型和相似候选组。")
                self.next_step_button.setText("开始 AI 分析")
            elif human_confirmed <= 0:
                self.next_step_label.setText("AI建议已生成。下一步进入审核与修正，由你确认最终工作流结果。")
                self.next_step_button.setText("进入审核与修正")
            else:
                self.next_step_label.setText("已有人工确认结果。可以生成导出计划并安全复制整理结果。")
                self.next_step_button.setText("导出与整理")
        if hasattr(self, "project_empty_state_label"):
            if total:
                self.project_empty_state_label.setText(f"当前项目已导入 {total} 张照片。下一步建议：运行 AI 分析或进入审核与修正。")
            else:
                self.project_empty_state_label.setText("尚未导入照片。建议先选择照片文件夹，再运行 AI 分析，最后进入人工审核。")
        if hasattr(self, "review_empty_state_card"):
            self.review_empty_state_card.setVisible(total == 0)
        if hasattr(self, "review_stack"):
            self.review_stack.setVisible(total > 0)
        if hasattr(self, "ai_empty_state_card"):
            self.ai_empty_state_card.setVisible(analyzed == 0)
        if hasattr(self, "ai_action_group"):
            self.ai_action_group.setVisible(analyzed > 0)
        if hasattr(self, "ai_overview_group"):
            self.ai_overview_group.setVisible(analyzed > 0)
        if hasattr(self, "export_empty_state_card"):
            self.export_empty_state_card.setVisible(human_confirmed == 0)
        if hasattr(self, "export_controls_widget"):
            self.export_controls_widget.setVisible(human_confirmed > 0)
        if hasattr(self, "export_safety_group"):
            self.export_safety_group.setVisible(human_confirmed > 0)

    def _build_sidebar_navigation(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("SidebarNavigation")
        sidebar.setMinimumWidth(160)
        sidebar.setMaximumWidth(210)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(8, 10, 8, 10)
        layout.setSpacing(8)
        title = QLabel("工作流")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        self.nav_buttons: dict[str, QPushButton] = {}
        sections = [
            ("project", "1. 导入与设置"),
            ("ai", "2. AI分析结果"),
            ("review", "3. 审核与修正"),
            ("export", "4. 导出与整理"),
        ]
        for key, text in sections:
            button = QPushButton(text)
            button.setObjectName("SidebarNavButton")
            button.setCheckable(True)
            button.setMinimumHeight(36)
            button.setToolTip(f"切换到{text}模块")
            button.clicked.connect(lambda _checked=False, section=key: self._activate_workflow_section(section))
            self.nav_buttons[key] = button
            layout.addWidget(button)
        layout.addStretch(1)
        return sidebar

    def _build_review_board(self) -> QWidget:
        board = QWidget()
        board.setObjectName("ReviewCorrectionPage")
        layout = QVBoxLayout(board)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        header = QHBoxLayout()
        self.review_photo_mode_button = QPushButton("照片审核")
        self.review_similar_mode_button = QPushButton("AI相似候选组")
        for button in [self.review_photo_mode_button, self.review_similar_mode_button]:
            button.setCheckable(True)
            button.setMinimumHeight(32)
            button.setObjectName("WorkflowSubNavButton")
            header.addWidget(button)
        header.addStretch(1)
        self.review_context_label = QLabel(self._workflow_context_text())
        self.review_context_label.setObjectName("WorkflowContext")
        self.review_context_label.setWordWrap(True)
        self.workflow_context_labels["review"] = self.review_context_label
        self.review_empty_state_card, review_empty_layout = self._make_card("暂无可审核照片", "EmptyStateCard")
        review_empty_body = QLabel("请先导入照片，或检查当前筛选条件。导入后这里会显示照片列表、大图预览和人工审核面板。")
        review_empty_body.setObjectName("EmptyStateBody")
        review_empty_body.setWordWrap(True)
        self.review_empty_project_button = self._make_button("返回项目中心", "回到导入与设置页面。")
        self.review_empty_project_button.setObjectName("SecondaryButton")
        self.review_empty_import_button = self._make_button("导入照片", "选择照片文件夹。")
        self.review_empty_import_button.setObjectName("PrimaryButton")
        review_empty_buttons = QHBoxLayout()
        review_empty_buttons.addWidget(self.review_empty_import_button)
        review_empty_buttons.addWidget(self.review_empty_project_button)
        review_empty_buttons.addStretch(1)
        review_empty_layout.addWidget(review_empty_body)
        review_empty_layout.addLayout(review_empty_buttons)
        layout.addLayout(header)
        layout.addWidget(self.review_context_label)
        layout.addWidget(self.review_empty_state_card)

        self.review_stack = QStackedWidget()
        self.review_stack.setObjectName("ReviewCorrectionStack")
        self.review_stack.addWidget(self.review_workspace)
        self.review_stack.addWidget(self.similar_group_panel)
        layout.addWidget(self.review_stack, 1)

        self.review_photo_mode_button.clicked.connect(lambda: self._set_review_subsection("photo"))
        self.review_similar_mode_button.clicked.connect(lambda: self._set_review_subsection("similar"))
        self._set_review_subsection("photo", persist=False)
        return board

    def _set_review_subsection(self, subsection: str, persist: bool = True) -> None:
        if not hasattr(self, "review_stack"):
            return
        self.current_review_subsection = "similar" if subsection == "similar" else "photo"
        self.review_stack.setCurrentIndex(1 if self.current_review_subsection == "similar" else 0)
        if hasattr(self, "review_photo_mode_button"):
            self.review_photo_mode_button.setChecked(self.current_review_subsection == "photo")
            self.review_similar_mode_button.setChecked(self.current_review_subsection == "similar")
        if self.current_review_subsection == "similar":
            self.refresh_similarity_panel()
        if persist:
            self._save_layout_state()

    def _build_flow_navigation_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("FlowNavBar")
        bar.setFixedHeight(42)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)
        self.nav_buttons: dict[str, QPushButton] = {}
        sections = [
            ("project", "① 导入照片"),
            ("ai", "② 自动分析"),
            ("review", "③ 人工复核"),
            ("similar", "④ 相似组筛选"),
            ("export", "⑤ 导出结果"),
            ("style", "⑥ 训练我的风格"),
            ("settings", "设置 / 高级 / 调试"),
        ]
        for key, text in sections:
            button = QPushButton(text)
            button.setObjectName("FlowNavButton")
            button.setCheckable(True)
            button.setMinimumHeight(32)
            button.clicked.connect(lambda _checked=False, section=key: self._activate_workflow_section(section))
            self.nav_buttons[key] = button
            layout.addWidget(button)
        layout.addStretch(1)
        return bar

    def _build_top_panel_container(self, workflow_panel: QWidget) -> QWidget:
        panel = QFrame()
        panel.setObjectName("TopPanelContainer")
        panel.setMaximumHeight(68)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        compact = QHBoxLayout()
        self.background_compact_label = QLabel("后台任务：空闲｜进度 0/0｜成功 0｜跳过 0｜失败 0")
        self.background_compact_label.setObjectName("CompactTaskStatus")
        self.background_compact_label.setWordWrap(False)
        compact.addWidget(self.background_compact_label, 1)
        self.compact_background_run_button = self._make_button("立即运行", "按当前勾选项运行后台任务。")
        self.compact_background_pause_button = self._make_button("暂停", "当前文件处理完成后暂停。")
        self.compact_background_resume_button = self._make_button("继续", "从暂停处继续运行。")
        self.compact_background_stop_button = self._make_button("停止", "当前文件处理完成后停止。")
        self.top_panel_toggle_button = self._make_button("展开详情", "展开或收起后台任务、设置和高级调试内容。")
        for button in [
            self.compact_background_run_button,
            self.compact_background_pause_button,
            self.compact_background_resume_button,
            self.compact_background_stop_button,
            self.top_panel_toggle_button,
        ]:
            button.setMinimumHeight(30)
            compact.addWidget(button)
        layout.addLayout(compact)

        self.top_detail_scroll = QScrollArea()
        self.top_detail_scroll.setObjectName("TopDetailScroll")
        self.top_detail_scroll.setWidgetResizable(True)
        self.top_detail_scroll.setMaximumHeight(150)
        self.top_detail_scroll.setWidget(workflow_panel)
        self.top_detail_scroll.setVisible(False)
        layout.addWidget(self.top_detail_scroll)

        self.top_panel_expanded = False
        self.compact_background_run_button.clicked.connect(self.start_background_tasks)
        self.compact_background_pause_button.clicked.connect(self.pause_background_tasks)
        self.compact_background_resume_button.clicked.connect(self.resume_background_tasks)
        self.compact_background_stop_button.clicked.connect(self.stop_background_tasks)
        self.top_panel_toggle_button.clicked.connect(lambda: self._set_top_panel_expanded(not self.top_panel_expanded))
        return panel

    def _build_review_workspace(self) -> QWidget:
        workspace = QWidget()
        workspace.setObjectName("ReviewWorkspacePage")
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(FILTERS)
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(SORTS)
        self.view_combo = QComboBox()
        self.view_combo.addItems(["单图审片", "多图对比", "缩略图网格"])
        self.thumb_size_combo = QComboBox()
        self.thumb_size_combo.addItems(["小", "中", "大"])
        self.thumb_size_combo.setCurrentText("中")
        self.thumb_mode_combo = QComboBox()
        self.thumb_mode_combo.addItems(["contain", "cover"])
        self.thumb_mode_combo.setCurrentText(self.config.thumbnail_mode)

        filter_cards = QHBoxLayout()
        filter_cards.setSpacing(12)
        manual_filter_card, manual_filter_layout = self._make_card("人工状态筛选", "ActionCard")
        manual_filter_hint = QLabel("人工确认、待复核、重复和交付状态")
        manual_filter_hint.setObjectName("MetricNote")
        manual_filter_hint.setWordWrap(True)
        manual_filter_layout.addWidget(manual_filter_hint)
        manual_filter_layout.addWidget(self.filter_combo)

        ai_filter_card, ai_filter_layout = self._make_card("AI建议筛选", "ActionCard")
        ai_filter_hint = QLabel("AI低置信度、AI相似候选组和业务类型建议")
        ai_filter_hint.setObjectName("MetricNote")
        ai_filter_hint.setWordWrap(True)
        ai_filter_layout.addWidget(ai_filter_hint)
        ai_filter_layout.addWidget(self.sort_combo)

        view_filter_card, view_filter_layout = self._make_card("视图与缩略图", "ActionCard")
        view_filter_hint = QLabel("切换审片视图、缩略图大小和显示方式")
        view_filter_hint.setObjectName("MetricNote")
        view_filter_hint.setWordWrap(True)
        thumb_row = QHBoxLayout()
        thumb_row.setSpacing(8)
        thumb_row.addWidget(self.view_combo)
        thumb_row.addWidget(self.thumb_size_combo)
        thumb_row.addWidget(self.thumb_mode_combo)
        view_filter_layout.addWidget(view_filter_hint)
        view_filter_layout.addLayout(thumb_row)

        for card in [manual_filter_card, ai_filter_card, view_filter_card]:
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            filter_cards.addWidget(card)
        layout.addLayout(filter_cards)

        progress_bar = QHBoxLayout()
        self.work_progress = QProgressBar()
        self.work_progress.setRange(0, 100)
        self.stage_label = QLabel("阶段：空闲")
        self.current_file_label = QLabel("当前文件：-")
        self.count_label = QLabel("0 / 0")
        self.elapsed_label = QLabel("已用时间：00:00")
        progress_bar.addWidget(self.work_progress, 1)
        progress_bar.addWidget(self.stage_label)
        progress_bar.addWidget(self.current_file_label)
        progress_bar.addWidget(self.count_label)
        progress_bar.addWidget(self.elapsed_label)
        layout.addLayout(progress_bar)

        self.horizontal_splitter = QSplitter(Qt.Horizontal)
        self.horizontal_splitter.setObjectName("HorizontalWorkSplitter")
        self.horizontal_splitter.addWidget(self._build_left_panel())
        self.horizontal_splitter.addWidget(self._build_center_panel())
        self.horizontal_splitter.addWidget(self._build_right_scroll())
        self.horizontal_splitter.setStretchFactor(0, 0)
        self.horizontal_splitter.setStretchFactor(1, 1)
        self.horizontal_splitter.setStretchFactor(2, 0)
        self.horizontal_splitter.setSizes([240, 1120, 360])
        layout.addWidget(self.horizontal_splitter, 1)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(80)
        layout.addWidget(self.log_edit)
        return workspace

    def _set_top_panel_expanded(self, expanded: bool, persist: bool = True) -> None:
        self.top_panel_expanded = expanded
        if persist:
            self._save_layout_state()

    def _activate_workflow_section(self, section: str, persist: bool = True) -> None:
        if section == "settings":
            section = "project"
            if hasattr(self, "advanced_settings_group"):
                self.advanced_settings_group.setChecked(True)
        if section == "similar":
            section = "review"
            self._set_review_subsection("similar", persist=False)
        elif section == "review":
            self._set_review_subsection("photo", persist=False)
        if not hasattr(self, "page_widgets") or section not in self.page_widgets:
            section = "review"
        self.current_main_section = section
        self.main_stack.setCurrentWidget(self.page_widgets[section])
        if section == "review" and getattr(self, "current_review_subsection", "photo") == "similar":
            self.refresh_similarity_panel()
        if section == "export":
            self.update_export_stats()
        self._set_active_nav_button(section)
        self._refresh_workflow_contexts()
        if persist:
            self._save_layout_state()

    def _set_active_nav_button(self, section: str) -> None:
        for key, button in getattr(self, "nav_buttons", {}).items():
            button.setChecked(key == section)

    def on_workflow_tab_changed(self, index: int) -> None:
        self._save_layout_state()

    def _save_layout_state(self) -> None:
        if not hasattr(self, "body_splitter"):
            return
        settings = QSettings("PhotoSelectAI", "PhotoSelectAIAssistant")
        settings.setValue("layout/version", 4)
        settings.setValue("layout/body_splitter", self.body_splitter.saveState())
        settings.setValue("layout/horizontal_splitter", self.horizontal_splitter.saveState())
        settings.setValue("layout/main_section", getattr(self, "current_main_section", "review"))
        settings.setValue("layout/review_subsection", getattr(self, "current_review_subsection", "photo"))
        settings.setValue("layout/window_geometry", self.saveGeometry())

    def _restore_layout_state(self) -> None:
        settings = QSettings("PhotoSelectAI", "PhotoSelectAIAssistant")
        try:
            layout_version = int(settings.value("layout/version", 0))
        except (TypeError, ValueError):
            layout_version = 0
        body_state = settings.value("layout/body_splitter") if layout_version == 4 else None
        horizontal_state = settings.value("layout/horizontal_splitter") if layout_version in {3, 4} else None
        geometry = settings.value("layout/window_geometry") if layout_version == 4 else None
        if geometry:
            self.restoreGeometry(geometry)
        if body_state:
            self.body_splitter.restoreState(body_state)
        else:
            self.body_splitter.setSizes([180, 1320])
        if horizontal_state:
            self.horizontal_splitter.restoreState(horizontal_state)
        else:
            self.horizontal_splitter.setSizes([240, 1120, 360])
        self._ensure_review_splitter_priority()
        section = str(settings.value("layout/main_section", "review")) if layout_version == 4 else "review"
        self._activate_workflow_section(section, persist=False)
        review_subsection = str(settings.value("layout/review_subsection", "photo")) if layout_version == 4 else "photo"
        if self.current_main_section == "review":
            self._set_review_subsection(review_subsection, persist=False)
        self.body_splitter.splitterMoved.connect(lambda _pos, _index: self._save_layout_state())
        self.horizontal_splitter.splitterMoved.connect(lambda _pos, _index: self._save_layout_state())

    def _ensure_review_splitter_priority(self) -> None:
        """Keep the image review area dominant when restoring old or cramped layouts."""
        sizes = self.horizontal_splitter.sizes()
        if len(sizes) < 3:
            self.horizontal_splitter.setSizes([240, 1120, 360])
            return
        total = sum(sizes)
        if total <= 0:
            self.horizontal_splitter.setSizes([240, 1120, 360])
            return
        left, center, right = sizes[:3]
        center_is_cramped = center < total * 0.45 or center <= max(left, right)
        if not center_is_cramped:
            return
        left_target = max(220, min(280, int(total * 0.18)))
        right_target = max(320, min(400, int(total * 0.26)))
        center_target = max(420, total - left_target - right_target)
        if center_target <= max(left_target, right_target):
            center_target = int(total * 0.58)
            remaining = max(0, total - center_target)
            left_target = min(260, max(180, int(remaining * 0.42)))
            right_target = max(260, remaining - left_target)
        self.horizontal_splitter.setSizes([left_target, center_target, right_target])

    def _build_workflow_panel(self) -> QWidget:
        tabs = QTabWidget()
        tabs.setObjectName("WorkflowTabs")
        self.workflow_tabs = tabs

        self.folder_label = QLabel("尚未选择文件夹")
        self.folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.select_folder_button = self._make_button("选择照片文件夹", "第一步：选择一个本地照片文件夹，后台导入并生成缩略图缓存。原图不会被删除。")
        self.project_wizard_button = self._make_button("新建项目向导", "按摄影工作流引导创建项目：选择文件夹、确认业务类型、进入 AI 分析。")
        self.save_button = self._make_button("保存项目", "保存当前筛片结果、配置和项目状态。快捷键 Ctrl+S。")
        self.restore_project_button = self._make_button("恢复上次项目", "恢复上次筛片项目，包括筛选条件、当前照片和缩略图设置。")
        self.workflow_guide_button = self._make_button("工作流向导", "查看推荐流程、当前完成度、样本数量和下一步建议。")
        self.cancel_import_button = self._make_button("取消导入", "安全取消正在进行的导入任务，已完成部分会保留。")
        project_tab = self._tab_with_rows(
            [
                [self.project_wizard_button, self.select_folder_button, self.save_button, self.restore_project_button, self.workflow_guide_button, self.cancel_import_button],
                [QLabel("当前项目"), self.folder_label],
            ]
        )
        tabs.addTab(project_tab, "① 导入照片")

        self.current_ai_button = self._make_button("分析当前照片", "只分析当前照片，适合复核单张疑难片。不会覆盖人工分类。")
        self.ai_analyze_button = self._make_button("批量 AI 分析", "第二步：批量生成 AI 推荐、评分、相似组和修图建议。快捷键 Ctrl+Shift+A。")
        self.model_auto_group_button = self._make_button("生成 AI 相似候选组", "使用模型 embedding 生成相似候选组选片建议；只写 auto_group_* 字段，不覆盖人工结果。")
        self.reanalyze_unconfirmed_button = self._make_button("重新分析未人工确认", "使用最新 AI 规则重新分析未审、AI已审和待复核照片；人工已确认的照片不会被覆盖。")
        self.clear_stale_ai_button = self._make_button("清理旧版 AI 建议", "清空旧版或串图的 AI 建议，不清空人工筛片结果，也不会删除原图。")
        self.rescore_button = self._make_button("重新评分", "基于已有 AI 结果、当前策略和阈值重新计算最终建议，不重新跑模型。")
        self.analysis_stats_button = self._make_button("查看分析统计", "查看强烈推荐、可交付、备选、重复和废片数量。")
        self.cancel_ai_button = self._make_button("取消 AI", "取消正在进行的 AI 分析，已完成结果会保留。")
        self.analysis_stats_label = QLabel("分析统计：尚未分析")
        self.analysis_stats_label.setWordWrap(True)
        self.ai_version_label = QLabel(f"当前 AI 版本：{CURRENT_AI_SUGGESTION_VERSION}")
        self.ai_version_label.setObjectName("ImportantConclusion")
        ai_tab = self._tab_with_rows(
            [
                [self.ai_version_label],
                [self.current_ai_button, self.ai_analyze_button, self.model_auto_group_button, self.reanalyze_unconfirmed_button, self.clear_stale_ai_button],
                [self.rescore_button, self.analysis_stats_button, self.cancel_ai_button],
                [self.analysis_stats_label],
            ]
        )
        tabs.addTab(ai_tab, "② 自动分析")

        self.single_view_button = self._make_button("单图审片", "切换到单张大图审片模式。")
        self.compare_view_button = self._make_button("多图对比", "切换到 2-4 张照片对比模式。")
        self.grid_view_button = self._make_button("缩略图网格", "切换到大缩略图网格，适合快速浏览和批量选择。")
        self.workflow_show_group_button = self._make_button("相似组查看", "查看当前照片所在的相似组并进入对比模式。")
        self.shortcut_help_button = self._make_button("快捷键说明", "查看快速审片键盘操作说明。")
        self.workflow_selected_button = self._make_button("精修候选", "把当前照片标记为后续精修候选。手动判断优先于 AI。")
        self.workflow_retouch_button = self._make_button("可交付", "照片可以进入客户交付或基础修图队列。")
        self.workflow_pending_button = self._make_button("备选", "人工保留为备选片。")
        self.workflow_crop_button = self._make_button("重复", "标记为相似重复或连拍待选，不会删除原图。")
        self.workflow_rejected_button = self._make_button("废片", "人工确认废片。导出时归入废片文件夹，但不会删除原图。")
        self.workflow_portfolio_button = self._make_button("作品集候选", "标记为适合进一步精修、发布或作品集复核的照片。")
        self.workflow_practice_button = self._make_button("练习片", "保留为练习、调色、样片测试用途。")
        review_tab = self._tab_with_rows(
            [
                [self.single_view_button, self.compare_view_button, self.grid_view_button, self.workflow_show_group_button, self.shortcut_help_button],
                [
                    self.workflow_selected_button,
                    self.workflow_retouch_button,
                    self.workflow_pending_button,
                    self.workflow_crop_button,
                    self.workflow_rejected_button,
                    self.workflow_portfolio_button,
                    self.workflow_practice_button,
                ],
            ]
        )
        tabs.addTab(review_tab, "③ 人工复核")

        self.similar_group_panel = SimilarGroupPanel()

        self.export_path_edit = QLineEdit()
        self.export_path_edit.setReadOnly(True)
        self.export_path_edit.setPlaceholderText("默认导出到照片文件夹旁边的 PhotoSelect_Output")
        self.export_path_edit.setMinimumWidth(420)
        self.choose_export_path_button = self._make_button("选择导出路径", "可选择输出文件夹。默认只复制原图，不移动、不覆盖。")
        self.export_mode_combo = QComboBox()
        self.export_mode_combo.addItems(["复制原图 + 生成报告", "复制原图", "复制 JPG / 预览图", "只生成报告"])
        self.export_mode_combo.setCurrentText("复制原图 + 生成报告")
        self.export_use_dirs_checkbox = QCheckBox("生成按用途整理目录")
        self.export_use_dirs_checkbox.setChecked(True)
        self.export_type_dirs_checkbox = QCheckBox("生成按类型整理目录")
        self.export_type_dirs_checkbox.setChecked(True)
        self.export_include_rejects_checkbox = QCheckBox("包含 X 废片")
        self.export_include_no_export_checkbox = QCheckBox("包含“不导出”照片")
        self.export_csv_checkbox = QCheckBox("生成 CSV 报告")
        self.export_csv_checkbox.setChecked(True)
        self.export_md_checkbox = QCheckBox("生成 Markdown 复盘报告")
        self.export_md_checkbox.setChecked(True)
        self.export_button = self._make_button("开始导出", "先显示分类统计预览，确认后按中文分类复制。不会删除原图。")
        self.cancel_export_button = self._make_button("取消导出", "安全取消正在进行的导出，已复制文件会保留并写入报告。")
        self.open_export_folder_button = self._make_button("打开导出文件夹", "打开当前导出目录。目录不存在时会提示。")
        self.open_report_button = self._make_button("打开报告文件", "打开最近一次导出的 CSV 或 Markdown 报告。")
        self.export_csv_button = self._make_button("生成 CSV 报告", "只生成 CSV 报告，不复制或移动照片。")
        self.export_stats_label = QLabel("导出统计：导入并分析后会显示各分类数量。")
        self.export_stats_label.setWordWrap(True)
        self.export_page = self._tab_with_rows(
            [
                [QLabel("导出路径"), self.export_path_edit, self.choose_export_path_button],
                [QLabel("导出方式"), self.export_mode_combo, self.export_button, self.cancel_export_button, self.open_export_folder_button, self.open_report_button, self.export_csv_button],
                [self.export_use_dirs_checkbox, self.export_type_dirs_checkbox, self.export_include_rejects_checkbox, self.export_include_no_export_checkbox],
                [self.export_csv_checkbox, self.export_md_checkbox],
                [self.export_stats_label],
            ]
        )

        self.record_pref_button = self._make_button("记录当前人工选择", "把当前照片的人工分类加入偏好样本。")
        self.train_pref_button = self._make_button("训练我的偏好模型", "至少需要 50 张人工标记样本。只训练本地轻量模型。")
        self.apply_pref_button = self._make_button("用我的风格重新评分", "使用偏好模型更新偏好分和建议，不覆盖人工分类。")
        self.style_report_button = self._make_button("查看我的风格报告", "查看样本数量、类别分布、风险提示和模型状态。")
        self.clear_pref_button = self._make_button("清空偏好模型", "删除本地偏好模型，保留样本文件。")
        self.export_pref_button = self._make_button("导出偏好样本", "显示偏好样本 JSONL 路径，便于备份或外部分析。")
        self.add_aesthetic_folder_button = self._make_button("添加审美素材库", "添加你拥有使用权的本地素材文件夹。不会爬取网络图片。")
        self.scan_aesthetic_button = self._make_button("扫描素材库", "扫描已添加的审美素材库，并建立本地索引。")
        self.aesthetic_report_button = self._make_button("审美素材库报告", "查看素材数量、标签分布和数据来源提醒。")
        style_tab = self._tab_with_rows(
            [
                [self.record_pref_button, self.train_pref_button, self.apply_pref_button, self.style_report_button],
                [self.clear_pref_button, self.export_pref_button],
                [self.add_aesthetic_folder_button, self.scan_aesthetic_button, self.aesthetic_report_button],
            ]
        )
        tabs.addTab(style_tab, "⑤ 训练我的风格")

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["保守", "严格", "交付", "作品集", "学习我的风格"])
        strategy_map = {
            "conservative": "保守",
            "strict": "严格",
            "delivery": "交付",
            "portfolio": "作品集",
            "learned": "学习我的风格",
        }
        self.strategy_combo.setCurrentText(strategy_map.get(self.config.screening_strategy, "交付"))
        self.strategy_combo.setToolTip("设置 AI 的筛片策略。交付模式会减少误杀，作品集模式更严格。")
        self.pick_ratio_spin = QDoubleSpinBox()
        self.pick_ratio_spin.setRange(0.01, 1.0)
        self.pick_ratio_spin.setSingleStep(0.01)
        self.pick_ratio_spin.setValue(self.config.target_pick_ratio)
        self.pick_ratio_spin.setDecimals(2)
        self.pick_ratio_spin.setToolTip("设置目标精选比例，例如 0.20 表示约推荐 20%。")
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.10, 0.95)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setValue(self.config.confidence_threshold)
        self.confidence_spin.setToolTip("低于阈值的 AI 结果会进入人工复核。")
        self.ai_enable_combo = QComboBox()
        self.ai_enable_combo.addItems(["AI开启", "AI关闭"])
        self.ai_enable_combo.setCurrentText("AI开启" if self.config.enable_ai else "AI关闭")
        self.ai_enable_combo.setToolTip("关闭后保留手动选片功能，AI 按钮会停用。")
        self.autosave_spin = QSpinBox()
        self.autosave_spin.setRange(10, 600)
        self.autosave_spin.setValue(self.config.autosave_interval_seconds)
        self.autosave_spin.setSuffix("秒")
        self.autosave_spin.setToolTip("自动保存间隔。")

        self.perf_button = self._make_button("性能统计", "查看耗时、缓存命中、fallback 数量和 GPU 占用率波动解释。")
        self.model_profile_combo = QComboBox()
        self.model_profile_combo.addItems(["fast", "balanced", "accurate", "ultra"])
        self.model_profile_combo.setCurrentText(self.config.model_profile)
        self.model_profile_combo.setToolTip("切换模型档位。RTX 4070 Ti 日常推荐 balanced，精筛推荐 accurate。")
        self.model_status_button = self._make_button("模型状态", "查看 OpenCLIP / YOLO 下载、加载、设备和缓存状态。")
        self.download_model_button = self._make_button("下载推荐模型", "确认后下载当前档位推荐的开源预训练模型。不会自动下载。")
        self.cancel_model_download_button = self._make_button("取消模型下载", "请求取消正在进行的模型下载。")
        self.test_model_button = self._make_button("测试模型", "快速测试当前 AI 管线能否初始化。")
        self.detect_gpu_button = self._make_button("测试GPU", "检测 torch、CUDA、GPU 名称和显存。")
        self.model_self_test_button = self._make_button("模型自测", "抽样运行环境、模型、小样本推理和瓶颈评估。")
        self.auto_tune_button = self._make_button("自动调优", "测试 batch_size 等配置并写入推荐配置。")
        self.clear_model_cache_button = self._make_button("清理模型缓存", "清理 models/cache，不删除已下载模型权重。")
        self.clear_embedding_cache_button = self._make_button("清理embedding缓存", "清空 cache/embeddings，后续分析会重新计算。")
        settings_tab = self._tab_with_rows(
            [
                [QLabel("筛片策略"), self.strategy_combo, QLabel("目标精选比例"), self.pick_ratio_spin, QLabel("置信度阈值"), self.confidence_spin],
                [self.ai_enable_combo, QLabel("自动保存"), self.autosave_spin],
                [QLabel("模型档位"), self.model_profile_combo, self.model_status_button, self.download_model_button, self.cancel_model_download_button],
                [self.detect_gpu_button, self.test_model_button, self.model_self_test_button, self.auto_tune_button, self.perf_button],
                [self.clear_model_cache_button, self.clear_embedding_cache_button],
            ]
        )
        background_panel = self._build_background_task_panel()
        settings_layout = settings_tab.layout()
        if isinstance(settings_layout, QVBoxLayout):
            settings_layout.addWidget(background_panel)
        tabs.addTab(settings_tab, "设置/高级/调试")
        return tabs

    def _build_background_task_panel(self) -> QWidget:
        group = QGroupBox("后台任务")
        layout = QVBoxLayout(group)

        status_grid = QGridLayout()
        self.background_state_label = QLabel("当前状态：空闲")
        self.background_task_label = QLabel("当前任务：-")
        self.background_file_label = QLabel("当前文件：-")
        self.background_count_label = QLabel("进度：0 / 0")
        self.background_result_label = QLabel("成功：0  跳过：0  失败：0")
        self.background_elapsed_label = QLabel("用时：-")
        for label in [
            self.background_state_label,
            self.background_task_label,
            self.background_file_label,
            self.background_count_label,
            self.background_result_label,
            self.background_elapsed_label,
        ]:
            label.setWordWrap(True)
        status_grid.addWidget(self.background_state_label, 0, 0)
        status_grid.addWidget(self.background_task_label, 0, 1)
        status_grid.addWidget(self.background_file_label, 1, 0, 1, 2)
        status_grid.addWidget(self.background_count_label, 2, 0)
        status_grid.addWidget(self.background_result_label, 2, 1)
        status_grid.addWidget(self.background_elapsed_label, 3, 0)
        layout.addLayout(status_grid)

        self.background_progress = QProgressBar()
        self.background_progress.setRange(0, 100)
        layout.addWidget(self.background_progress)

        button_row = QHBoxLayout()
        self.background_run_button = self._make_button("立即运行", "按勾选任务生成后台队列并开始运行。默认不会自动占用 CPU/GPU。")
        self.background_pause_button = self._make_button("暂停", "当前文件处理完成后暂停后台任务。")
        self.background_resume_button = self._make_button("继续", "从暂停处继续后台任务。")
        self.background_stop_button = self._make_button("停止", "当前文件处理完成后停止本轮任务，保留已完成结果。")
        self.background_clear_button = self._make_button("清空队列", "清空已完成或待运行的后台任务记录。运行中需要先停止。")
        for button in [
            self.background_run_button,
            self.background_pause_button,
            self.background_resume_button,
            self.background_stop_button,
            self.background_clear_button,
        ]:
            button_row.addWidget(button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        mode_row = QHBoxLayout()
        self.background_start_mode_combo = QComboBox()
        self.background_start_mode_combo.addItems(["手动启动", "软件空闲时询问我", "自动运行，低优先级"])
        self.background_start_mode_combo.setToolTip("默认手动启动。只有明确选择自动低优先级后，才允许后台任务自动运行。")
        mode_row.addWidget(QLabel("启动方式"))
        mode_row.addWidget(self.background_start_mode_combo)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        task_grid = QGridLayout()
        self.background_task_checkboxes: dict[str, QCheckBox] = {}
        default_checked = {TASK_THUMBNAIL_CACHE, TASK_SIMILAR_PHOTOS, TASK_EXPORT_PREVIEW}
        task_types = [
            TASK_THUMBNAIL_CACHE,
            TASK_SIMILAR_PHOTOS,
            TASK_MODEL_AUTO_GROUP,
            TASK_AI_PRECLASSIFY,
            TASK_FACE_QUALITY,
            TASK_POSE_ANALYSIS,
            TASK_EXPRESSION_ANALYSIS,
            TASK_PREFERENCE_STATS,
            TASK_EXPORT_PREVIEW,
        ]
        for index, task_type in enumerate(task_types):
            checkbox = QCheckBox(TASK_NAMES[task_type])
            checkbox.setChecked(task_type in default_checked)
            if task_type == TASK_MODEL_AUTO_GROUP:
                checkbox.setToolTip(
                    "使用视觉 embedding 生成 AI 相似候选组，只写 auto_group_* 建议字段；"
                    "不会删除、移动原图，也不会覆盖人工相似组状态。"
                )
            else:
                checkbox.setToolTip("模型类任务当前未接入时会安全跳过，不会崩溃或删除原图。")
            self.background_task_checkboxes[task_type] = checkbox
            task_grid.addWidget(checkbox, index // 2, index % 2)
        layout.addLayout(task_grid)

        self.background_queue_edit = QTextEdit()
        self.background_queue_edit.setReadOnly(True)
        self.background_queue_edit.setMaximumHeight(130)
        self.background_queue_edit.setPlaceholderText("任务队列会显示在这里。")
        layout.addWidget(self.background_queue_edit)

        log_row = QHBoxLayout()
        self.background_recent_log_label = QLabel("最近日志：-")
        self.background_recent_log_label.setWordWrap(True)
        self.open_background_log_button = self._make_button("打开任务日志", "打开 logs/background_tasks.log 查看详细任务记录。")
        log_row.addWidget(self.background_recent_log_label, 1)
        log_row.addWidget(self.open_background_log_button)
        layout.addLayout(log_row)
        return group

    def _make_button(self, text: str, tooltip: str) -> QPushButton:
        button = QPushButton(text)
        button.setToolTip(tooltip)
        return button

    def _tab_with_rows(self, rows: list[list[QWidget]]) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        for widgets in rows:
            row = QHBoxLayout()
            for widget in widgets:
                row.addWidget(widget)
            row.addStretch(1)
            layout.addLayout(row)
        layout.addStretch(1)
        return panel

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("LeftPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        title = QLabel("照片")
        title.setObjectName("PanelTitle")
        self.photo_view = QListView()
        self.photo_view.setModel(self.model)
        self.photo_view.setItemDelegate(self.left_delegate)
        self.photo_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.photo_view.setViewMode(QListView.ListMode)
        self.photo_view.setUniformItemSizes(True)
        self.photo_view.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.photo_view.setMinimumWidth(240)
        layout.addWidget(title)
        layout.addWidget(self.photo_view, 1)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("CenterPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.center_stack = QStackedWidget()
        self.center_stack.addWidget(self._build_single_preview())
        self.center_stack.addWidget(self._build_compare_preview())
        self.center_stack.addWidget(self._build_grid_preview())
        layout.addWidget(self.center_stack, 1)
        return panel

    def _build_single_preview(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        controls = QHBoxLayout()
        self.fit_button = QPushButton("适应窗口")
        self.actual_button = QPushButton("100%")
        self.zoom_in_button = QPushButton("放大")
        self.zoom_out_button = QPushButton("缩小")
        controls.addWidget(self.fit_button)
        controls.addWidget(self.actual_button)
        controls.addWidget(self.zoom_in_button)
        controls.addWidget(self.zoom_out_button)
        controls.addStretch(1)
        self.preview_label = QLabel("请选择照片")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setObjectName("PreviewImage")
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setWidget(self.preview_label)
        self.preview_scroll.setObjectName("PreviewScroll")
        self.preview_info_label = QLabel("-")
        self.preview_info_label.setObjectName("PreviewInfo")
        quick_bar = QHBoxLayout()
        self.quick_selected_button = QPushButton("[S] 强烈推荐")
        self.quick_retouch_button = QPushButton("[A] 可交付")
        self.quick_pending_button = QPushButton("[B] 备选")
        self.quick_rejected_button = QPushButton("[X] 废片")
        self.quick_duplicate_button = QPushButton("[R] 重复")
        self.quick_prev_button = QPushButton("← 上一张")
        self.quick_next_button = QPushButton("下一张 →")
        self.quick_fit_button = QPushButton("[Space] 适应/100%")
        for button in [
            self.quick_selected_button,
            self.quick_retouch_button,
            self.quick_pending_button,
            self.quick_rejected_button,
            self.quick_duplicate_button,
            self.quick_prev_button,
            self.quick_next_button,
            self.quick_fit_button,
        ]:
            button.setObjectName("QuickAction")
            quick_bar.addWidget(button)
        quick_bar.addStretch(1)
        layout.addLayout(controls)
        layout.addWidget(self.preview_scroll, 1)
        layout.addWidget(self.preview_info_label)
        layout.addLayout(quick_bar)
        self.shortcut_hint_label = QLabel(SHORTCUT_HINT_TEXT)
        self.shortcut_hint_label.setObjectName("ShortcutHint")
        self.shortcut_hint_label.setWordWrap(True)
        layout.addWidget(self.shortcut_hint_label)
        return panel

    def _build_compare_preview(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        self.compare_tiles = [CompareTile(index) for index in range(4)]
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for tile, (row, column) in zip(self.compare_tiles, positions):
            layout.addWidget(tile, row, column)
        return panel

    def _build_grid_preview(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.grid_view = QListView()
        self.grid_view.setModel(self.model)
        self.grid_view.setItemDelegate(self.grid_delegate)
        self.grid_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.grid_view.setViewMode(QListView.IconMode)
        self.grid_view.setResizeMode(QListView.Adjust)
        self.grid_view.setWrapping(True)
        self.grid_view.setUniformItemSizes(True)
        self.grid_view.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        layout.addWidget(self.grid_view, 1)
        return panel

    def _build_right_scroll(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setObjectName("RightScroll")
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340)
        scroll.setMaximumWidth(520)
        scroll.setWidget(self._build_right_panel())
        return scroll

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("RightPanel")
        panel.setMinimumWidth(340)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)

        self.file_name_label = QLabel("-")
        self.file_name_label.setObjectName("FileTitle")
        self.file_name_label.setWordWrap(True)
        self.basic_info_label = QLabel("-")
        self.basic_info_label.setWordWrap(True)
        self.manual_label = QLabel("人工确认：-")
        self.final_label = QLabel("系统导出分类（人工优先）：-")
        self.review_sequence_label = QLabel("当前序号：-")
        self.review_status_label = QLabel(f"审片状态：{REVIEW_STATUS_UNREVIEWED}")

        info_group = QGroupBox("当前照片信息")
        info_layout = QVBoxLayout(info_group)
        info_layout.addWidget(self.file_name_label)
        info_layout.addWidget(self.basic_info_label)
        info_layout.addWidget(self.review_sequence_label)
        info_layout.addWidget(self.review_status_label)
        info_layout.addWidget(self.manual_label)
        info_layout.addWidget(self.final_label)
        layout.addWidget(info_group)

        self.review_ai_suggestion_label = QLabel("暂无 AI 建议，请先运行 AI 预分析或手动筛片。")
        self.review_ai_suggestion_label.setWordWrap(True)
        ai_review_group = QGroupBox("AI建议，仅供参考")
        ai_review_layout = QVBoxLayout(ai_review_group)
        ai_review_layout.addWidget(self.review_ai_suggestion_label)
        self.reanalyze_current_suggestion_button = self._make_button("重新分析当前照片", "使用最新业务分类规则重新分析当前照片；人工已确认字段不会被覆盖。")
        ai_review_layout.addWidget(self.reanalyze_current_suggestion_button)
        layout.addWidget(ai_review_group)

        type_group = QGroupBox("人工决定：照片类型")
        type_layout = QVBoxLayout(type_group)
        self.photo_type_combo = QComboBox()
        self.photo_type_combo.addItem("")
        self.photo_type_combo.addItems(get_photo_types())
        type_layout.addWidget(self.photo_type_combo)
        layout.addWidget(type_group)

        subtype_group = QGroupBox("人工决定：细分类别")
        subtype_layout = QVBoxLayout(subtype_group)
        self.subtype_combo = QComboBox()
        subtype_layout.addWidget(self.subtype_combo)
        layout.addWidget(subtype_group)

        quality_group = QGroupBox("人工决定：品质评级")
        quality_layout = QGridLayout(quality_group)
        self.quality_buttons: dict[str, QPushButton] = {}
        for index, rating in enumerate(get_quality_ratings()):
            button = QPushButton(rating["display"])
            button.setCheckable(True)
            button.setToolTip(rating.get("description", ""))
            self.quality_buttons[rating["code"]] = button
            quality_layout.addWidget(button, index // 2, index % 2)
        layout.addWidget(quality_group)

        scores_group = QGroupBox("商业与作品评分")
        scores_layout = QGridLayout(scores_group)
        self.commercial_score_spin = QSpinBox()
        self.commercial_score_spin.setRange(0, 100)
        self.commercial_score_spin.setSuffix(" 分")
        self.commercial_score_spin.setToolTip("这张照片是否适合客户交付。")
        self.portfolio_score_spin = QSpinBox()
        self.portfolio_score_spin.setRange(0, 100)
        self.portfolio_score_spin.setSuffix(" 分")
        self.portfolio_score_spin.setToolTip("这张照片是否适合进入摄影师个人作品集。")
        scores_layout.addWidget(QLabel("商业交付"), 0, 0)
        scores_layout.addWidget(self.commercial_score_spin, 0, 1)
        scores_layout.addWidget(QLabel("作品集"), 1, 0)
        scores_layout.addWidget(self.portfolio_score_spin, 1, 1)
        layout.addWidget(scores_group)

        delivery_group = QGroupBox("人工决定：交付用途")
        delivery_layout = QGridLayout(delivery_group)
        self.delivery_checkboxes: dict[str, QCheckBox] = {}
        for index, name in enumerate(get_delivery_uses()):
            checkbox = QCheckBox(name)
            self.delivery_checkboxes[name] = checkbox
            delivery_layout.addWidget(checkbox, index // 2, index % 2)
        layout.addWidget(delivery_group)

        issues_group = QGroupBox("人工决定：问题标签")
        issues_layout = QGridLayout(issues_group)
        self.issue_checkboxes: dict[str, QCheckBox] = {}
        for index, name in enumerate(get_issue_tags()):
            checkbox = QCheckBox(name)
            self.issue_checkboxes[name] = checkbox
            issues_layout.addWidget(checkbox, index // 2, index % 2)
        layout.addWidget(issues_group)

        note_group = QGroupBox("人工确认：备注")
        note_layout = QVBoxLayout(note_group)
        self.review_note_edit = QTextEdit()
        self.review_note_edit.setMinimumHeight(110)
        self.review_note_edit.setPlaceholderText("例如：表情不错，但背景略乱，可修。")
        note_layout.addWidget(self.review_note_edit)
        layout.addWidget(note_group)

        actions_group = QGroupBox("操作")
        actions_layout = QGridLayout(actions_group)
        self.save_review_button = QPushButton("保存人工判断")
        self.confirm_next_button = QPushButton("确认并下一张")
        self.mark_review_button = QPushButton("标记待复核")
        self.reset_review_button = QPushButton("重置人工判断")
        actions_layout.addWidget(self.save_review_button, 0, 0)
        actions_layout.addWidget(self.confirm_next_button, 0, 1)
        actions_layout.addWidget(self.mark_review_button, 1, 0)
        actions_layout.addWidget(self.reset_review_button, 1, 1)
        layout.addWidget(actions_group)

        similar_actions_group = QGroupBox("人工确认：AI相似候选组处理")
        similar_actions_layout = QGridLayout(similar_actions_group)
        self.show_group_button = QPushButton("查看本组")
        self.recommended_keep_button = QPushButton("设为组内最佳")
        self.group_backup_quick_button = QPushButton("标记备选")
        self.group_duplicate_quick_button = QPushButton("标记重复")
        similar_actions_layout.addWidget(self.show_group_button, 0, 0)
        similar_actions_layout.addWidget(self.recommended_keep_button, 0, 1)
        similar_actions_layout.addWidget(self.group_backup_quick_button, 1, 0)
        similar_actions_layout.addWidget(self.group_duplicate_quick_button, 1, 1)
        layout.addWidget(similar_actions_group)

        learning_group = QGroupBox("学习我的选片逻辑（只记录样本，不训练大模型）")
        learning_layout = QGridLayout(learning_group)
        self.preference_record_checkbox = QCheckBox("保存人工修正为偏好样本")
        self.preference_record_checkbox.setChecked(True)
        self.preference_reason_combo = QComboBox()
        self.preference_reason_combo.addItems(
            [
                "表情问题",
                "动作问题",
                "光线问题",
                "构图问题",
                "重复片",
                "客户可能喜欢",
                "作品集价值",
                "只留档",
            ]
        )
        self.record_pref_current_button = QPushButton("记录当前人工决定")
        self.preference_hint_label = QLabel("当你保存人工决定时，会记录 AI 原建议、用户最终选择、是否采纳 AI 和修改原因标签。")
        self.preference_hint_label.setWordWrap(True)
        learning_layout.addWidget(self.preference_record_checkbox, 0, 0, 1, 2)
        learning_layout.addWidget(QLabel("修改原因标签"), 1, 0)
        learning_layout.addWidget(self.preference_reason_combo, 1, 1)
        learning_layout.addWidget(self.record_pref_current_button, 2, 0, 1, 2)
        learning_layout.addWidget(self.preference_hint_label, 3, 0, 1, 2)
        layout.addWidget(learning_group)

        mark_group = QGroupBox("旧分类/兼容")
        mark_group.setCheckable(True)
        mark_group.setChecked(False)
        mark_layout = QGridLayout(mark_group)
        self.selected_button = QPushButton("精修候选")
        self.retouch_button = QPushButton("可交付")
        self.pending_button = QPushButton("备选")
        self.crop_button = QPushButton("重复")
        self.rejected_button = QPushButton("废片")
        self.portfolio_button = QPushButton("作品集候选")
        self.practice_button = QPushButton("练习片")
        mark_buttons = [
            self.selected_button,
            self.retouch_button,
            self.pending_button,
            self.crop_button,
            self.rejected_button,
            self.portfolio_button,
            self.practice_button,
        ]
        for index, button in enumerate(mark_buttons):
            mark_layout.addWidget(button, index // 2, index % 2)
        mark_layout.addWidget(QLabel("用户备注"), 4, 0, 1, 2)
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(120)
        mark_layout.addWidget(self.notes_edit, 5, 0, 1, 2)
        layout.addWidget(mark_group)

        legacy_children = mark_group.findChildren(QWidget)

        def toggle_legacy(checked: bool, children: list[QWidget] = legacy_children) -> None:
            for child in children:
                child.setVisible(checked)

        mark_group.toggled.connect(toggle_legacy)
        toggle_legacy(False)

        self.ai_labels: dict[str, QLabel] = {}

        def add_ai_group(title: str, rows: list[tuple[str, str]], checked: bool = True) -> QGroupBox:
            group = QGroupBox(title)
            group.setCheckable(True)
            group.setChecked(checked)
            group_layout = QVBoxLayout(group)
            group_labels: list[QLabel] = []
            for label_title, key in rows:
                label = QLabel(f"{label_title}：-")
                label.setWordWrap(True)
                self.ai_labels[key] = label
                group_labels.append(label)
                group_layout.addWidget(label)

            def toggle_labels(checked: bool, labels: list[QLabel] = group_labels) -> None:
                for child_label in labels:
                    child_label.setVisible(checked)

            group.toggled.connect(toggle_labels)
            toggle_labels(checked)
            layout.addWidget(group)
            return group

        add_ai_group(
            "AI建议：推荐结论",
            [
                ("AI业务类型建议", "ai_path"),
                ("AI一级建议", "primary"),
                ("AI二级建议", "secondary"),
                ("置信度", "confidence"),
                ("AI评分建议", "final_recommendation"),
                ("AI理由", "screening_reason"),
            ],
        )
        self.ai_labels["final_recommendation"].setObjectName("ImportantConclusion")
        add_ai_group(
            "AI建议：多模型分析结果",
            [
                ("基础质量", "module_basic"),
                ("人脸状态", "module_face"),
                ("肢体姿态", "module_pose"),
                ("表情状态", "module_expression"),
                ("构图分析", "module_composition"),
                ("业务类型", "module_business"),
                ("综合建议", "module_decision"),
            ],
        )
        add_ai_group(
            "AI建议：技术评分",
            [
                ("质量标签", "quality"),
                ("绝对质量分", "absolute_score"),
                ("相对质量分", "relative_score"),
                ("可用性分", "final_pick_score"),
                ("相似组排名", "group_rank"),
                ("本组推荐保留", "recommended"),
                ("相似组ID", "similar"),
            ],
        )
        add_ai_group(
            "AI建议：审美与商业评分",
            [
                ("用户偏好模型分数", "preference_score"),
                ("用户偏好判断原因", "preference_reason"),
                ("相似历史样本", "similar_history"),
                ("筛片策略", "strategy"),
                ("作品建议", "portfolio_suggestion"),
                ("交付建议", "delivery_suggestion"),
            ],
        )
        add_ai_group(
            "AI建议：修图建议",
            [
                ("是否可修", "retouchable"),
                ("是否可裁切", "croppable"),
                ("修图建议", "retouch_suggestion"),
                ("裁切建议", "crop_suggestion"),
                ("风格判断", "style_label"),
                ("AI判断原因", "ai_reason"),
                ("最终建议原因", "final_reason"),
            ],
        )
        debug_group = add_ai_group(
            "系统状态：模型/调试信息",
            [
                ("识别来源", "source"),
                ("语义Top3", "top3"),
                ("语义分", "semantic_score"),
                ("检测人数", "detected_people"),
                ("YOLO置信度", "yolo_confidence"),
                ("主体完整性", "subject_integrity"),
            ],
            checked=False,
        )

        self.ai_dependency_label = QLabel("-")
        self.ai_dependency_label.setWordWrap(True)
        self.ai_dependency_label.setObjectName("DependencyLabel")
        self.model_status_label = QLabel("-")
        self.model_status_label.setWordWrap(True)
        self.model_status_label.setObjectName("DependencyLabel")
        debug_group.layout().addWidget(self.model_status_label)
        debug_group.layout().addWidget(self.ai_dependency_label)
        self.model_status_label.setVisible(False)
        self.ai_dependency_label.setVisible(False)
        debug_group.toggled.connect(self.model_status_label.setVisible)
        debug_group.toggled.connect(self.ai_dependency_label.setVisible)
        layout.addStretch(1)
        return panel

    def _connect_signals(self) -> None:
        self.select_folder_button.clicked.connect(self.select_folder)
        self.project_wizard_button.clicked.connect(self.show_project_setup_wizard)
        self.hero_new_project_button.clicked.connect(self.select_folder)
        self.hero_open_project_button.clicked.connect(self.select_folder)
        self.hero_continue_project_button.clicked.connect(self.restore_last_project_ui)
        self.next_step_button.clicked.connect(self.run_project_next_step)
        self.save_button.clicked.connect(lambda: self.save_all(auto=False))
        self.restore_project_button.clicked.connect(self.restore_last_project_ui)
        self.workflow_guide_button.clicked.connect(self.show_workflow_guide)
        self.cancel_import_button.clicked.connect(self.cancel_import)
        self.current_ai_button.clicked.connect(self.start_current_ai)
        self.reanalyze_current_suggestion_button.clicked.connect(self.start_current_ai)
        self.ai_analyze_button.clicked.connect(self.start_batch_ai)
        self.model_auto_group_button.clicked.connect(self.start_model_auto_group_ui)
        self.ai_empty_start_button.clicked.connect(self.start_batch_ai)
        self.ai_empty_group_button.clicked.connect(self.start_model_auto_group_ui)
        self.reanalyze_unconfirmed_button.clicked.connect(self.reanalyze_unconfirmed_photos)
        self.clear_stale_ai_button.clicked.connect(self.clear_stale_ai_suggestions)
        self.cancel_ai_button.clicked.connect(self.cancel_ai)
        self.rescore_button.clicked.connect(self.rescore_existing_ai)
        self.analysis_stats_button.clicked.connect(self.show_analysis_stats)
        self.export_button.clicked.connect(self.confirm_and_export)
        self.export_csv_button.clicked.connect(self.export_csv_report_only)
        self.cancel_export_button.clicked.connect(self.cancel_export)
        self.choose_export_path_button.clicked.connect(self.choose_export_path)
        self.open_export_folder_button.clicked.connect(self.open_export_folder)
        self.open_report_button.clicked.connect(self.open_report_file)
        self.export_empty_review_button.clicked.connect(lambda: self._activate_workflow_section("review"))
        self.export_mode_combo.currentTextChanged.connect(lambda _text: self.update_export_stats())
        for checkbox in [
            self.export_use_dirs_checkbox,
            self.export_type_dirs_checkbox,
            self.export_include_rejects_checkbox,
            self.export_include_no_export_checkbox,
            self.export_csv_checkbox,
            self.export_md_checkbox,
        ]:
            checkbox.toggled.connect(lambda _checked=False: self.update_export_stats())
        self.photo_view.selectionModel().selectionChanged.connect(self.on_selection_changed)
        self.grid_view.selectionModel().selectionChanged.connect(self.on_grid_selection_changed)
        self.filter_combo.currentTextChanged.connect(self.apply_filter_sort)
        self.sort_combo.currentTextChanged.connect(self.apply_filter_sort)
        self.view_combo.currentTextChanged.connect(self.change_view_mode)
        self.strategy_combo.currentTextChanged.connect(self.change_strategy)
        self.pick_ratio_spin.valueChanged.connect(self.change_pick_ratio)
        self.confidence_spin.valueChanged.connect(self.change_confidence_threshold)
        self.ai_enable_combo.currentTextChanged.connect(self.change_ai_enabled)
        self.autosave_spin.valueChanged.connect(self.change_autosave_interval)
        self.thumb_size_combo.currentTextChanged.connect(lambda _text: self.change_thumbnail_options())
        self.thumb_mode_combo.currentTextChanged.connect(lambda _text: self.change_thumbnail_options())
        self.photo_type_combo.currentTextChanged.connect(self.on_photo_type_changed)
        self.subtype_combo.currentTextChanged.connect(lambda _text: self.on_review_field_changed())
        self.commercial_score_spin.valueChanged.connect(lambda _value: self.on_review_field_changed())
        self.portfolio_score_spin.valueChanged.connect(lambda _value: self.on_review_field_changed())
        self.review_note_edit.textChanged.connect(self.on_review_field_changed)
        for code, button in self.quality_buttons.items():
            button.clicked.connect(lambda _checked=False, rating_code=code: self.set_quality_rating(rating_code))
        for checkbox in self.delivery_checkboxes.values():
            checkbox.toggled.connect(lambda _checked=False: self.on_review_field_changed())
        for checkbox in self.issue_checkboxes.values():
            checkbox.toggled.connect(lambda _checked=False: self.on_review_field_changed())
        self.save_review_button.clicked.connect(lambda: self.save_current_review(quiet=False))
        self.confirm_next_button.clicked.connect(self.confirm_review_and_next)
        self.mark_review_button.clicked.connect(self.mark_current_needs_review)
        self.reset_review_button.clicked.connect(self.reset_current_review)
        self.selected_button.clicked.connect(lambda: self.mark_focused(MANUAL_SELECTED))
        self.pending_button.clicked.connect(lambda: self.mark_focused(MANUAL_PENDING))
        self.rejected_button.clicked.connect(lambda: self.mark_focused(MANUAL_REJECTED))
        self.retouch_button.clicked.connect(lambda: self.mark_focused(MANUAL_RETOUCH))
        self.crop_button.clicked.connect(lambda: self.mark_focused(MANUAL_CROP))
        self.portfolio_button.clicked.connect(lambda: self.mark_focused(MANUAL_PORTFOLIO))
        self.practice_button.clicked.connect(lambda: self.mark_focused(MANUAL_PRACTICE))
        self.workflow_selected_button.clicked.connect(lambda: self.mark_focused(MANUAL_SELECTED))
        self.workflow_pending_button.clicked.connect(lambda: self.mark_focused(MANUAL_PENDING))
        self.workflow_rejected_button.clicked.connect(lambda: self.mark_focused(MANUAL_REJECTED))
        self.workflow_retouch_button.clicked.connect(lambda: self.mark_focused(MANUAL_RETOUCH))
        self.workflow_crop_button.clicked.connect(lambda: self.mark_focused(MANUAL_CROP))
        self.workflow_portfolio_button.clicked.connect(lambda: self.mark_focused(MANUAL_PORTFOLIO))
        self.workflow_practice_button.clicked.connect(lambda: self.mark_focused(MANUAL_PRACTICE))
        self.quick_selected_button.clicked.connect(lambda: self.shortcut_set_quality_rating("S"))
        self.quick_retouch_button.clicked.connect(lambda: self.shortcut_set_quality_rating("A"))
        self.quick_pending_button.clicked.connect(lambda: self.shortcut_set_quality_rating("B"))
        self.quick_rejected_button.clicked.connect(lambda: self.shortcut_set_quality_rating("X"))
        self.quick_duplicate_button.clicked.connect(lambda: self.toggle_issue_tag("重复照片"))
        self.quick_prev_button.clicked.connect(self.previous_photo)
        self.quick_next_button.clicked.connect(self.next_photo)
        self.quick_fit_button.clicked.connect(self.toggle_fit_actual)
        self.shortcut_help_button.clicked.connect(self.show_shortcut_help)
        self.single_view_button.clicked.connect(lambda: self.view_combo.setCurrentText("单图审片"))
        self.compare_view_button.clicked.connect(lambda: self.view_combo.setCurrentText("多图对比"))
        self.grid_view_button.clicked.connect(lambda: self.view_combo.setCurrentText("缩略图网格"))
        self.notes_edit.textChanged.connect(self.on_notes_changed)
        self.show_group_button.clicked.connect(self.show_current_group)
        self.workflow_show_group_button.clicked.connect(self.show_current_group)
        self.similar_group_panel.request_run_similarity_task.connect(self.calculate_similarity_groups_ui)
        self.similar_group_panel.group_status_changed.connect(self.on_similar_group_status_changed)
        self.similar_group_panel.request_open_photo.connect(self.select_source_index)
        self.review_empty_project_button.clicked.connect(lambda: self._activate_workflow_section("project"))
        self.review_empty_import_button.clicked.connect(self.select_folder)
        self.recommended_keep_button.clicked.connect(self.set_recommended_keep)
        self.group_backup_quick_button.clicked.connect(lambda: self.set_current_group_status("backup"))
        self.group_duplicate_quick_button.clicked.connect(lambda: self.set_current_group_status("duplicate"))
        self.record_pref_current_button.clicked.connect(self.record_current_preference)
        self.record_pref_button.clicked.connect(self.record_current_preference)
        self.train_pref_button.clicked.connect(self.train_preference_model_ui)
        self.apply_pref_button.clicked.connect(self.apply_preference_model)
        self.style_report_button.clicked.connect(self.show_preference_report)
        self.clear_pref_button.clicked.connect(self.clear_preference_model_ui)
        self.export_pref_button.clicked.connect(self.export_preference_data_ui)
        self.add_aesthetic_folder_button.clicked.connect(self.add_aesthetic_folder_ui)
        self.scan_aesthetic_button.clicked.connect(self.scan_aesthetic_library_ui)
        self.aesthetic_report_button.clicked.connect(self.show_aesthetic_report_ui)
        self.perf_button.clicked.connect(self.show_ai_performance_stats)
        self.model_profile_combo.currentTextChanged.connect(self.change_model_profile)
        self.model_status_button.clicked.connect(self.show_model_status)
        self.download_model_button.clicked.connect(self.download_recommended_models)
        self.cancel_model_download_button.clicked.connect(self.cancel_model_download)
        self.test_model_button.clicked.connect(self.test_models_ui)
        self.detect_gpu_button.clicked.connect(self.detect_gpu_ui)
        self.model_self_test_button.clicked.connect(self.start_model_self_test)
        self.auto_tune_button.clicked.connect(self.start_auto_tune)
        self.clear_model_cache_button.clicked.connect(self.clear_model_cache_ui)
        self.clear_embedding_cache_button.clicked.connect(self.clear_embedding_cache_ui)
        self.background_run_button.clicked.connect(self.start_background_tasks)
        self.background_pause_button.clicked.connect(self.pause_background_tasks)
        self.background_resume_button.clicked.connect(self.resume_background_tasks)
        self.background_stop_button.clicked.connect(self.stop_background_tasks)
        self.background_clear_button.clicked.connect(self.clear_background_task_queue)
        self.background_start_mode_combo.currentTextChanged.connect(self.change_background_task_options)
        self.open_background_log_button.clicked.connect(self.open_background_task_log)
        for checkbox in self.background_task_checkboxes.values():
            checkbox.toggled.connect(self.change_background_task_options)
        self.background_task_manager.task_started.connect(self.on_background_task_started)
        self.background_task_manager.task_progress.connect(self.on_background_task_progress)
        self.background_task_manager.task_paused.connect(self.on_background_task_paused)
        self.background_task_manager.task_resumed.connect(self.on_background_task_resumed)
        self.background_task_manager.task_stopping.connect(self.on_background_task_stopping)
        self.background_task_manager.task_stopped.connect(self.on_background_task_stopped)
        self.background_task_manager.task_completed.connect(self.on_background_task_completed)
        self.background_task_manager.task_error.connect(self.on_background_task_error)
        self.background_task_manager.queue_changed.connect(self.refresh_background_queue_view)
        self.background_task_manager.all_finished.connect(self.on_background_all_finished)
        self.workflow_tabs.currentChanged.connect(self.on_workflow_tab_changed)
        self.fit_button.clicked.connect(self.fit_preview)
        self.actual_button.clicked.connect(self.actual_preview)
        self.zoom_in_button.clicked.connect(lambda: self.zoom_preview(1.25))
        self.zoom_out_button.clicked.connect(lambda: self.zoom_preview(0.8))
        for tile in self.compare_tiles:
            tile.clicked.connect(self.set_focus_source_index)
            tile.mark_requested.connect(self.mark_source_index)

    def _install_shortcuts(self) -> None:
        manager = ShortcutManager(self)
        self.shortcut_manager = manager
        manager.register("Left", self.previous_photo)
        manager.register("Right", self.next_photo)
        manager.register("Return", self.confirm_review_and_next)
        manager.register("Enter", self.confirm_review_and_next)
        manager.register("Space", self.toggle_fit_actual)
        for rating in ["S", "A", "B", "C", "X"]:
            manager.register(rating, lambda code=rating: self.shortcut_set_quality_rating(code))
        for number, photo_type in PHOTO_TYPE_SHORTCUTS.items():
            manager.register(str(number), lambda selected_type=photo_type: self.set_photo_type_by_shortcut(selected_type))
        for key, delivery_use in DELIVERY_USE_SHORTCUTS.items():
            manager.register(key, lambda name=delivery_use: self.toggle_delivery_use(name))
        for key, issue_tag in ISSUE_TAG_SHORTCUTS.items():
            manager.register(key, lambda name=issue_tag: self.toggle_issue_tag(name))
        manager.register("M", self.mark_current_needs_review)
        manager.register("U", self.reset_current_review)
        manager.register("Ctrl+Alt+A", self.start_current_ai)
        manager.register("Ctrl+Shift+A", self.start_batch_ai)
        manager.register("Ctrl+A", self.select_all_visible)
        manager.register("Ctrl+S", self.save_current_item)
        manager.register("Ctrl+E", self.confirm_and_export)
        manager.register("Esc", self.escape_action)
        manager.register("Tab", self.focus_next_compare_tile)

    def _run_shortcut(self, callback) -> None:
        if isinstance(QApplication.focusWidget(), (QTextEdit, QLineEdit)):
            return
        callback()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { color: #e6eaf0; font-size: 13px; }
            QMainWindow { background: #191d23; }
            QFrame#TopStatusBar {
                background: #20262f;
                border: 1px solid #303744;
                border-radius: 8px;
            }
            QLabel#TopProjectLabel {
                color: #f8fafc;
                font-weight: 700;
            }
            QLabel#WorkflowPageTitle {
                color: #ffffff;
                font-size: 20px;
                font-weight: 800;
            }
            QLabel#WorkflowPageHint, QLabel#WorkflowContext {
                background: #20262f;
                border: 1px solid #303744;
                border-radius: 8px;
                color: #dbeafe;
                padding: 8px;
            }
            QFrame#DashboardCard, QFrame#MetricCard, QFrame#HeroCard, QFrame#ActionCard,
            QFrame#EmptyStateCard, QFrame#SafetyNoticeCard,
            QGroupBox#WorkflowCard {
                background: #20262f;
                border: 1px solid #303744;
                border-radius: 10px;
            }
            QFrame#HeroCard {
                background: #172033;
                border-color: #3b82f6;
            }
            QFrame#ActionCard {
                background: #202936;
            }
            QFrame#SafetyNoticeCard {
                background: #2a2417;
                border-color: #a16207;
            }
            QFrame#EmptyStateCard {
                background: #1e2632;
                border-color: #475569;
            }
            QLabel#HeroTitle {
                color: #ffffff;
                font-size: 30px;
                font-weight: 900;
            }
            QLabel#HeroSubtitle {
                color: #dbeafe;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#CardTitle {
                color: #f8fafc;
                font-weight: 800;
                font-size: 15px;
            }
            QLabel#MetricValue {
                color: #ffffff;
                font-size: 26px;
                font-weight: 900;
            }
            QLabel#MetricNote {
                color: #b6c2d2;
                font-size: 12px;
            }
            QLabel#EmptyStateBody {
                color: #dbeafe;
                font-size: 14px;
                line-height: 1.35;
            }
            QLabel#EmptyStateAction {
                color: #facc15;
                font-weight: 700;
            }
            QLabel#SafetyNoticeText {
                color: #fde68a;
                font-weight: 700;
            }
            QFrame#SidebarNavigation {
                background: #20262f;
                border: 1px solid #303744;
                border-radius: 8px;
            }
            QPushButton#SidebarNavButton {
                background: #242b36;
                border: 1px solid #374151;
                min-height: 34px;
                padding: 7px 10px;
                text-align: left;
            }
            QPushButton#SidebarNavButton:checked {
                background: #2563eb;
                border-color: #60a5fa;
                color: #ffffff;
                font-weight: 700;
            }
            QPushButton#WorkflowSubNavButton {
                background: #242b36;
                border: 1px solid #374151;
                min-height: 30px;
                padding: 6px 14px;
                font-weight: 700;
            }
            QPushButton#WorkflowSubNavButton:checked {
                background: #0f766e;
                border-color: #5eead4;
                color: #ffffff;
            }
            QFrame#FlowNavBar {
                background: #20262f;
                border: 1px solid #303744;
                border-radius: 8px;
            }
            QPushButton#FlowNavButton {
                background: #242b36;
                border: 1px solid #374151;
                min-height: 30px;
                padding: 5px 10px;
            }
            QPushButton#FlowNavButton:checked {
                background: #2563eb;
                border-color: #60a5fa;
                color: #ffffff;
                font-weight: 700;
            }
            QFrame#TopPanelContainer {
                background: #20262f;
                border: 1px solid #303744;
                border-radius: 8px;
            }
            QLabel#CompactTaskStatus {
                color: #dbeafe;
                font-weight: 700;
            }
            QFrame#LeftPanel { background: #242a33; border: 1px solid #303744; border-radius: 8px; }
            QFrame#RightPanel { background: #242a33; border: 1px solid #303744; border-radius: 8px; }
            QFrame#CenterPanel { background: #1f2329; border-radius: 8px; }
            QLabel#PanelTitle, QLabel#FileTitle { font-weight: 700; color: #ffffff; font-size: 15px; }
            QLabel#ImportantConclusion { font-weight: 800; color: #ffffff; font-size: 16px; }
            QLabel#PreviewInfo { background: #191d23; color: #cdd5e1; padding: 8px; border-radius: 6px; }
            QLabel#ShortcutHint { background: #20262f; color: #cbd5e1; padding: 7px; border-radius: 6px; font-size: 13px; }
            QLabel#PreviewImage, QLabel#CompareImage { background: #1f2329; color: #aeb8c6; }
            QLabel#DependencyLabel { color: #fbbf24; }
            QListView { background: #20262f; border: 1px solid #303744; border-radius: 8px; outline: none; }
            QScrollArea#PreviewScroll { background: #1f2329; border: 1px solid #303744; border-radius: 8px; }
            QScrollArea#RightScroll { background: transparent; border: none; }
            QScrollArea#TopDetailScroll { background: transparent; border: none; }
            QSplitter::handle { background: #303744; }
            QSplitter::handle:hover { background: #475569; }
            QTabWidget#WorkflowTabs::pane { border: 1px solid #303744; border-radius: 8px; background: #20262f; }
            QTabWidget#WorkflowTabs QTabBar::tab {
                background: #252b34;
                color: #cbd5e1;
                padding: 8px 14px;
                min-height: 28px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabWidget#WorkflowTabs QTabBar::tab:selected { background: #334155; color: #ffffff; font-weight: 700; }
            QGroupBox {
                background: #20262f;
                border: 1px solid #343d4b;
                border-radius: 8px;
                margin-top: 10px;
                padding: 8px;
                font-weight: 700;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #f8fafc; }
            QTextEdit, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
                background: #151922;
                border: 1px solid #3a4351;
                border-radius: 6px;
                color: #e6eaf0;
                padding: 5px;
            }
            QPushButton {
                background: #2e3643;
                border: 1px solid #465164;
                border-radius: 6px;
                padding: 7px 10px;
                color: #f8fafc;
                min-height: 32px;
            }
            QPushButton:hover { background: #3b4656; }
            QPushButton:disabled { background: #252b34; color: #707987; border-color: #333b47; }
            QPushButton#PrimaryButton {
                background: #2563eb;
                border-color: #60a5fa;
                color: #ffffff;
                font-weight: 800;
            }
            QPushButton#PrimaryButton:hover { background: #1d4ed8; }
            QPushButton#SecondaryButton {
                background: #263241;
                border-color: #4b5563;
                color: #e5e7eb;
                font-weight: 700;
            }
            QPushButton#SecondaryButton:hover { background: #334155; }
            QPushButton#DangerButton {
                background: #3b2227;
                border-color: #b45309;
                color: #fed7aa;
                font-weight: 700;
            }
            QPushButton#ActiveMark { background: #2563eb; border-color: #60a5fa; font-weight: 700; }
            QPushButton#QuickAction { background: #263241; border-color: #4b5563; font-weight: 700; }
            QProgressBar { background: #151922; border: 1px solid #3a4351; border-radius: 6px; height: 18px; text-align: center; }
            QProgressBar::chunk { background: #3b82f6; border-radius: 5px; }
            QFrame#CompareTile { background: #242a33; border: 1px solid #303744; border-radius: 8px; }
            QFrame#CompareTile[focused="true"] { border: 2px solid #60a5fa; }
            """
        )

    def _apply_thumbnail_options(self) -> None:
        size = THUMBNAIL_SIZES.get(self.thumb_size_combo.currentText(), THUMBNAIL_SIZES["中"])
        mode = self.thumb_mode_combo.currentText() or "contain"
        self.left_delegate.set_thumbnail_options(size, mode)
        self.grid_delegate.set_thumbnail_options(size, mode)
        self.photo_view.setIconSize(QSize(*size))
        self.grid_view.setIconSize(QSize(*size))
        self.photo_view.doItemsLayout()
        self.grid_view.doItemsLayout()

    def _sync_config_controls(self) -> None:
        controls = [
            self.confidence_spin,
            self.pick_ratio_spin,
            self.autosave_spin,
            self.ai_enable_combo,
            self.model_profile_combo,
            self.background_start_mode_combo,
        ]
        controls.extend(self.background_task_checkboxes.values())
        for control in controls:
            control.blockSignals(True)
        self.confidence_spin.setValue(self.config.confidence_threshold)
        self.pick_ratio_spin.setValue(self.config.target_pick_ratio)
        self.autosave_spin.setValue(self.config.autosave_interval_seconds)
        self.ai_enable_combo.setCurrentText("AI开启" if self.config.enable_ai else "AI关闭")
        self.model_profile_combo.setCurrentText(self.config.model_profile)
        mode_text = {
            "manual": "手动启动",
            "ask_when_idle": "软件空闲时询问我",
            "auto_low_priority": "自动运行，低优先级",
        }.get(self.config.background_task_start_mode, "手动启动")
        self.background_start_mode_combo.setCurrentText(mode_text)
        enabled_types = set(self.config.background_task_enabled_types or [TASK_THUMBNAIL_CACHE, TASK_SIMILAR_PHOTOS, TASK_EXPORT_PREVIEW])
        for task_type, checkbox in self.background_task_checkboxes.items():
            checkbox.setChecked(task_type in enabled_types)
        for control in controls:
            control.blockSignals(False)

    def _refresh_ai_status(self) -> None:
        self.ai_env = detect_ai_environment(self.config.use_gpu)
        self.model_manager = ModelManager(self.config)
        model_status = self.model_manager.status()
        detail = self.ai_env.message
        if self.ai_env.cuda_available:
            detail += f" · batch_size={self.config.batch_size} · 显存 {gpu_memory_text()}"
        self.ai_dependency_label.setText(detail)
        self.model_status_label.setText(
            f"语义模型：{model_status.semantic_model}（{model_status.semantic_status}）\n"
            f"检测模型：{model_status.detector_model}（{model_status.detector_status}）\n"
            f"档位：{model_status.model_profile} · embedding缓存：{model_status.embedding_cache_count} · 模型占用：{model_status.model_size_text}"
        )
        if hasattr(self, "gpu_status_label"):
            mode = "可用" if self.ai_env.ai_ready else "未就绪"
            self.gpu_status_label.setText(f"AI状态：{mode}")
        if hasattr(self, "model_default_label"):
            self.model_default_label.setText(self._model_default_text())
        self.ai_analyze_button.setEnabled(bool(self.items))
        self.model_auto_group_button.setEnabled(bool(self.items))
        self.current_ai_button.setEnabled(bool(self.items))
        self.reanalyze_unconfirmed_button.setEnabled(bool(self.items))
        self.clear_stale_ai_button.setEnabled(bool(self.items))
        self.reanalyze_current_suggestion_button.setEnabled(bool(self.items))
        summary = preference_summary()
        can_train = bool(summary.get("can_train"))
        missing = max(0, 50 - int(summary.get("total", 0)))
        self.train_pref_button.setEnabled(can_train)
        self.train_pref_button.setToolTip(
            "样本已达到 50 张，可以训练本地轻量偏好模型。"
            if can_train
            else f"至少需要 50 张人工标记样本，还差 {missing} 张。"
        )

    def mark_dirty(self, reason: str = "") -> None:
        self.dirty = True
        self.saved_status_label.setText("未保存")
        if reason:
            self.logger.info("状态变更：%s", reason)
        if not self.delayed_save_timer.isActive():
            self.delayed_save_timer.start(1500)

    def save_all(self, auto: bool = False, clean_shutdown: bool = False) -> bool:
        try:
            if self.repository and self.items:
                for item in self.items:
                    item.compute_final_category(self.config.confidence_threshold)
                self.repository.save_items(self.items)
            save_project_state(self._current_project_state(clean_shutdown=clean_shutdown))
            self.dirty = False
            now = datetime.now().strftime("%H:%M:%S")
            self.last_saved_text = f"已自动保存：{now}" if auto else f"已保存：{now}"
            self.saved_status_label.setText(self.last_saved_text)
            self.logger.info("%s保存时间：%s", "自动" if auto else "手动", now)
            if not auto:
                self.log("已保存。")
            return True
        except Exception:
            self.logger.exception("保存失败")
            if not auto:
                QMessageBox.critical(self, "保存失败", "保存失败，请查看 logs/app.log。")
            return False

    def _current_project_state(self, clean_shutdown: bool = False) -> ProjectState:
        selected_path = ""
        if 0 <= self.current_source_index < len(self.items):
            selected_path = str(self.items[self.current_source_index].path)
        return ProjectState(
            project_folder=str(self.selected_folder or ""),
            selected_path=selected_path,
            filter_name=self.filter_combo.currentText(),
            sort_name=self.sort_combo.currentText(),
            view_mode=self.view_combo.currentText(),
            thumbnail_size_label=self.thumb_size_combo.currentText(),
            thumbnail_mode=self.thumb_mode_combo.currentText(),
            dirty=self.dirty,
            clean_shutdown=clean_shutdown,
            extra={"current_source_index": self.current_source_index},
        )

    def mark_project_running(self) -> None:
        state = load_project_state()
        state.clean_shutdown = False
        save_project_state(state)

    def maybe_restore_project(self) -> None:
        state = load_project_state()
        if not self.config.restore_last_project or state.clean_shutdown or not state.project_folder:
            return
        reply = QMessageBox.question(
            self,
            "恢复筛片项目",
            "检测到上次未正常关闭的筛片项目，是否恢复？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.filter_combo.setCurrentText(state.filter_name)
            self.sort_combo.setCurrentText(state.sort_name)
            self.view_combo.setCurrentText(state.view_mode)
            self.thumb_size_combo.setCurrentText(state.thumbnail_size_label)
            self.thumb_mode_combo.setCurrentText(state.thumbnail_mode)
            self.start_import_folder(Path(state.project_folder), restore_selected_path=state.selected_path)

    def restore_last_project_ui(self) -> None:
        state = load_project_state()
        if not state.project_folder:
            QMessageBox.information(self, "恢复项目", "没有找到可恢复的上次项目。")
            return
        folder = Path(state.project_folder)
        if not folder.exists():
            QMessageBox.warning(self, "恢复项目", f"上次项目文件夹不存在：\n{folder}")
            return
        self.filter_combo.setCurrentText(state.filter_name)
        self.sort_combo.setCurrentText(state.sort_name)
        self.view_combo.setCurrentText(state.view_mode)
        self.thumb_size_combo.setCurrentText(state.thumbnail_size_label)
        self.thumb_mode_combo.setCurrentText(state.thumbnail_mode)
        self.start_import_folder(folder, restore_selected_path=state.selected_path)
        self.log(f"恢复上次项目：{folder}")

    def show_workflow_guide(self) -> None:
        sample_info = preference_summary()
        imported = len(self.items)
        analyzed = sum(1 for item in self.items if item.ai_primary_category or item.ai_confidence)
        manual_count = sum(1 for item in self.items if item.manual_override)
        pending_count = sum(1 for item in self.items if item.manual_category == MANUAL_PENDING)
        similar_groups = len({item.similar_group_id for item in self.items if item.similar_group_id})
        missing = max(0, 50 - sample_info.get("total", 0))
        pending_ratio = pending_count / imported if imported else 0
        advice = model_profile_advice(self.config, self.last_ai_performance.get("avg_ms", 0))
        lines = [
            "推荐流程",
            "",
            f"1. 导入照片：{'已完成' if imported else '未完成'}（{imported} 张）",
            f"2. 选择模型档位：当前 {self.config.model_profile}。{advice}",
            f"3. 运行 AI 粗筛：已分析 {analyzed}/{imported} 张",
            f"4. 人工修正样本：已人工标记 {manual_count} 张",
            f"5. 训练我的筛片风格：偏好样本 {sample_info.get('total', 0)} 张，还差 {missing} 张" if missing else f"5. 训练我的筛片风格：偏好样本 {sample_info.get('total', 0)} 张，已达到训练条件",
            f"6. 用我的风格重新评分：{'可执行' if sample_info.get('can_train') else '先积累样本'}",
            f"7. 多图对比复核：相似组 {similar_groups} 组",
            "8. 导出分类结果：导出前会显示统计预览，原图只复制不删除",
            "",
            "风险提示",
            f"- 当前备选比例：{pending_ratio:.0%}。AI 不会把不确定照片直接甩给备选，会优先给出人工复核或可修复建议。",
            f"- 当前 GPU / AI 状态：{self.ai_env.message}",
            "- 不会自动爬取网络照片；素材库只扫描你手动选择的本地文件夹。",
        ]
        QMessageBox.information(self, "工作流向导", "\n".join(lines))

    def show_project_setup_wizard(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("项目建立向导")
        dialog.setMinimumWidth(560)
        layout = QVBoxLayout(dialog)
        title = QLabel("建立一个新的摄影筛片项目")
        title.setObjectName("WorkflowPageTitle")
        title.setWordWrap(True)
        body = QLabel(
            "1. 选择本地照片文件夹，软件只读取和复制，不删除原图。\n"
            "2. 选择基础业务风格和处理目标，帮助 AI 给出更稳妥的建议。\n"
            "3. 运行 AI 分析和 AI 相似候选组，再由你人工确认最终结果。"
        )
        body.setWordWrap(True)
        body.setObjectName("WorkflowPageHint")
        layout.addWidget(title)
        layout.addWidget(body)
        row = QHBoxLayout()
        choose_button = self._make_button("选择照片文件夹", "关闭向导并选择项目照片文件夹。")
        restore_button = self._make_button("恢复上次项目", "关闭向导并尝试恢复上次项目。")
        close_button = self._make_button("稍后再说", "关闭向导。")
        row.addWidget(choose_button)
        row.addWidget(restore_button)
        row.addStretch(1)
        row.addWidget(close_button)
        layout.addLayout(row)
        choose_button.clicked.connect(lambda: (dialog.accept(), self.select_folder()))
        restore_button.clicked.connect(lambda: (dialog.accept(), self.restore_last_project_ui()))
        close_button.clicked.connect(dialog.reject)
        dialog.exec()

    def run_project_next_step(self) -> None:
        if not self.items:
            self.select_folder()
            return
        analyzed = sum(1 for item in self.items if item.ai_suggestion or item.ai_primary_category or item.ai_confidence)
        human_confirmed = sum(1 for item in self.items if item.review_status == REVIEW_STATUS_HUMAN_CONFIRMED)
        if analyzed <= 0:
            self._activate_workflow_section("ai")
            return
        if human_confirmed <= 0:
            self._activate_workflow_section("review")
            return
        self._activate_workflow_section("export")

    @Slot()
    def select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择照片文件夹")
        if not folder:
            return
        self.start_import_folder(Path(folder))

    def start_import_folder(self, folder: Path, restore_selected_path: str = "") -> None:
        self.selected_folder = folder
        self.export_output_dir = None
        self.folder_label.setText(str(self.selected_folder))
        if hasattr(self, "export_path_edit"):
            self.export_path_edit.setText(str(self.selected_folder.parent / "PhotoSelect_Output"))
        self.project_status_label.setText(self._project_status_text())
        self.items = []
        self.model.set_items([])
        self.current_source_index = -1
        self.focus_source_index = -1
        self.update_current_status()
        self.preview_label.setText("正在导入...")
        self.log(f"开始导入：{self.selected_folder}")
        self.logger.info("导入文件夹路径：%s", self.selected_folder)
        self.import_started_at = datetime.now()
        self._restore_selected_path = restore_selected_path
        self._set_import_running(True)

        self.import_thread = QThread(self)
        self.import_worker = ImportWorker(self.selected_folder, self.config)
        self.import_worker.moveToThread(self.import_thread)
        self.import_thread.started.connect(self.import_worker.run)
        self.import_worker.progress.connect(self.on_import_progress)
        self.import_worker.finished.connect(self.on_import_finished)
        self.import_worker.failed.connect(self.on_import_failed)
        self.import_worker.finished.connect(lambda *_args: self.import_thread.quit())
        self.import_worker.failed.connect(lambda *_args: self.import_thread.quit())
        self.import_thread.finished.connect(self.import_worker.deleteLater)
        self.import_thread.finished.connect(self.import_thread.deleteLater)
        self.import_thread.finished.connect(self._clear_import_refs)
        self.import_thread.start()

    @Slot()
    def cancel_import(self) -> None:
        if self.import_worker:
            self.import_worker.cancel()
            self.cancel_import_button.setEnabled(False)
            self.stage_label.setText("阶段：正在取消导入")

    @Slot(str, int, int, str)
    def on_import_progress(self, stage: str, done: int, total: int, filename: str) -> None:
        self.stage_label.setText(f"阶段：{stage}")
        self.current_file_label.setText(f"当前文件：{filename or '-'}")
        self.count_label.setText(f"已处理：{max(0, done)} / {max(0, total)}")
        self.work_progress.setValue(int(done / total * 100) if total else 0)
        if self.import_started_at:
            seconds = int((datetime.now() - self.import_started_at).total_seconds())
            self.elapsed_label.setText(f"已用时间：{seconds // 60:02d}:{seconds % 60:02d}")

    @Slot(object, object, object)
    def on_import_finished(self, items: list[PhotoItem], skipped: list[str], cancelled: bool) -> None:
        self.items = items
        self.repository = PhotoRepository(self.selected_folder / "cache" / "photoselect.db") if self.selected_folder else None
        self.model.set_items(self.items)
        self._set_import_running(False)
        self.export_button.setEnabled(bool(self.items))
        self.export_csv_button.setEnabled(bool(self.items))
        self._refresh_ai_status()
        self.total_status_label.setText(f"总照片数：{len(self.items)}")
        self.project_status_label.setText(self._project_status_text())
        self.log(f"导入完成：已导入 {len(items)} 张照片")
        self.count_label.setText(f"已完成：{len(items)} / {len(items)}")
        self.work_progress.setValue(100 if items else 0)
        self.update_export_stats()
        self.update_ai_result_overview()
        self.refresh_similarity_panel()
        self.background_ask_prompted = False
        if skipped:
            self.log(f"跳过 {len(skipped)} 张无法读取的图片，详情见 logs/app.log")
        if cancelled:
            self.log("导入已取消。")
        if self.model.rowCount() > 0:
            restore_path = getattr(self, "_restore_selected_path", "")
            restore_index = next((idx for idx, item in enumerate(self.items) if str(item.path) == restore_path), -1)
            self.select_source_index(restore_index if restore_index >= 0 else self.model.source_index_at_row(0))
        self.mark_dirty("导入新照片")
        QMessageBox.information(self, "导入完成", f"已导入 {len(items)} 张照片")

    @Slot(str)
    def on_import_failed(self, message: str) -> None:
        self._set_import_running(False)
        QMessageBox.critical(self, "导入失败", f"{message}\n详情见 logs/app.log")
        self.log(f"导入失败：{message}")

    @Slot()
    def _clear_import_refs(self) -> None:
        self.import_thread = None
        self.import_worker = None

    def _set_import_running(self, running: bool) -> None:
        self.select_folder_button.setEnabled(not running)
        self.cancel_import_button.setEnabled(running)
        self.current_ai_button.setEnabled(False if running else bool(self.items))
        self.ai_analyze_button.setEnabled(False if running else bool(self.items))
        self.model_auto_group_button.setEnabled(False if running else bool(self.items))
        self.reanalyze_unconfirmed_button.setEnabled(False if running else bool(self.items))
        self.clear_stale_ai_button.setEnabled(False if running else bool(self.items))
        self.reanalyze_current_suggestion_button.setEnabled(False if running else bool(self.items))
        self.export_button.setEnabled(False if running else bool(self.items))
        self.export_csv_button.setEnabled(False if running else bool(self.items))

    def apply_filter_sort(self) -> None:
        self.model.filter_name = self.filter_combo.currentText()
        self.model.sort_name = self.sort_combo.currentText()
        self.model.refresh()
        self.update_current_status()
        self.log(f"筛选：{self.model.filter_name}，排序：{self.model.sort_name}")

    def change_thumbnail_options(self) -> None:
        self.config = update_config(
            {
                "thumbnail_mode": self.thumb_mode_combo.currentText(),
                "thumbnail_size": THUMBNAIL_SIZES.get(self.thumb_size_combo.currentText(), THUMBNAIL_SIZES["中"])[0],
            }
        )
        self._apply_thumbnail_options()
        self.model.refresh()
        self.mark_dirty("修改缩略图配置")

    def change_similarity_mode(self, text: str) -> None:
        mode = {"严格": "strict", "标准": "standard", "宽松": "loose"}.get(text, "standard")
        if mode == self.config.similarity_mode:
            return
        self.config = update_config({"similarity_mode": mode})
        self.mark_dirty("修改相似度模式")
        self.log(f"相似度模式：{text}")

    def change_strategy(self, text: str) -> None:
        mapping = {
            "保守": "conservative",
            "严格": "strict",
            "交付": "delivery",
            "作品集": "portfolio",
            "学习我的风格": "learned",
        }
        self.config = update_config({"screening_strategy": mapping.get(text, "delivery")})
        self.mark_dirty("修改筛片策略")

    def change_pick_ratio(self, value: float) -> None:
        self.config = update_config({"target_pick_ratio": float(value)})
        self.mark_dirty("修改目标精选比例")

    def change_confidence_threshold(self, value: float) -> None:
        self.config = update_config({"confidence_threshold": float(value)})
        for item in self.items:
            item.compute_final_category(self.config.confidence_threshold)
        self.model.refresh()
        self.mark_dirty("修改置信度阈值")

    def change_ai_enabled(self, text: str) -> None:
        self.config = update_config({"enable_ai": text == "AI开启"})
        self._refresh_ai_status()
        self.mark_dirty("修改AI启用状态")

    def change_autosave_interval(self, value: int) -> None:
        self.config = update_config({"autosave_interval_seconds": int(value)})
        self.autosave_timer.setInterval(self.config.autosave_interval_seconds * 1000)
        self.mark_dirty("修改自动保存间隔")

    def change_view_mode(self, mode: str) -> None:
        if mode == "单图审片":
            self.center_stack.setCurrentIndex(0)
        elif mode == "多图对比":
            self.center_stack.setCurrentIndex(1)
            self.update_compare_view()
        else:
            self.center_stack.setCurrentIndex(2)

    @Slot()
    def on_selection_changed(self) -> None:
        source_indexes = self._selected_source_indexes(self.photo_view)
        if not source_indexes:
            self.update_current_status()
            return
        self.current_source_index = source_indexes[-1]
        self.focus_source_index = self.current_source_index
        self.show_item(self.items[self.current_source_index])
        if len(source_indexes) >= 2:
            self.view_combo.setCurrentText("多图对比")
            self.update_compare_view(source_indexes)

    @Slot()
    def on_grid_selection_changed(self) -> None:
        source_indexes = self._selected_source_indexes(self.grid_view)
        if source_indexes:
            self.select_source_index(source_indexes[-1])

    def _selected_source_indexes(self, view: QListView) -> list[int]:
        rows = sorted({index.row() for index in view.selectionModel().selectedIndexes() if index.isValid()})
        source_indexes: list[int] = []
        for row in rows:
            if 0 <= row < self.model.rowCount():
                source_index = self.model.source_index_at_row(row)
                if 0 <= source_index < len(self.items):
                    source_indexes.append(source_index)
        return source_indexes

    def select_source_index(self, source_index: int) -> None:
        if source_index < 0 or source_index >= len(self.items):
            return
        row = self.model.row_for_source_index(source_index)
        if row >= 0:
            index = self.model.index(row, 0)
            self.photo_view.selectionModel().select(index, QItemSelectionModel.ClearAndSelect)
            self.photo_view.scrollTo(index)
        self.current_source_index = source_index
        self.focus_source_index = source_index
        self.show_item(self.items[source_index])

    def _block_review_controls(self, blocked: bool) -> None:
        widgets: list[QWidget] = [
            self.photo_type_combo,
            self.subtype_combo,
            self.commercial_score_spin,
            self.portfolio_score_spin,
            self.review_note_edit,
        ]
        widgets.extend(self.quality_buttons.values())
        widgets.extend(self.delivery_checkboxes.values())
        widgets.extend(self.issue_checkboxes.values())
        for widget in widgets:
            widget.blockSignals(blocked)

    def _refresh_subtype_options(self, photo_type: str, selected: str = "") -> None:
        self.subtype_combo.blockSignals(True)
        self.subtype_combo.clear()
        subtypes = get_subtypes(photo_type)
        self.subtype_combo.addItems(subtypes)
        if selected and selected not in subtypes:
            self.subtype_combo.addItem(selected)
        if selected:
            self.subtype_combo.setCurrentText(selected)
        elif subtypes:
            self.subtype_combo.setCurrentIndex(0)
        self.subtype_combo.blockSignals(False)

    def _load_review_controls(self, item: PhotoItem) -> None:
        self._block_review_controls(True)
        try:
            if item.photo_type and self.photo_type_combo.findText(item.photo_type) < 0:
                self.photo_type_combo.addItem(item.photo_type)
            self.photo_type_combo.setCurrentText(item.photo_type or "")
            self._refresh_subtype_options(item.photo_type, item.subtype)
            for code, button in self.quality_buttons.items():
                button.setChecked(item.quality_rating == code)
                button.setObjectName("ActiveMark" if item.quality_rating == code else "")
                button.style().unpolish(button)
                button.style().polish(button)
            self.commercial_score_spin.setValue(_safe_int_score(item.commercial_score))
            self.portfolio_score_spin.setValue(_safe_int_score(item.portfolio_score))
            for name, checkbox in self.delivery_checkboxes.items():
                checkbox.setChecked(name in item.delivery_use)
            for name, checkbox in self.issue_checkboxes.items():
                checkbox.setChecked(name in item.issue_tags)
            self.review_note_edit.setPlainText(item.review_note or item.user_note)
        finally:
            self._block_review_controls(False)

    def _selected_quality_rating(self) -> str:
        for code, button in self.quality_buttons.items():
            if button.isChecked():
                return code
        return ""

    def set_quality_rating(self, rating_code: str) -> None:
        if self._loading_details:
            return
        for code, button in self.quality_buttons.items():
            button.blockSignals(True)
            button.setChecked(code == rating_code)
            button.blockSignals(False)
            button.setObjectName("ActiveMark" if code == rating_code else "")
            button.style().unpolish(button)
            button.style().polish(button)
        self.on_review_field_changed()

    def _collect_review_data(self, status: str = REVIEW_STATUS_HUMAN_CONFIRMED) -> dict:
        photo_type = self.photo_type_combo.currentText().strip()
        data = {
            "photo_type": photo_type,
            "subtype": self.subtype_combo.currentText().strip() if photo_type else "",
            "quality_rating": self._selected_quality_rating(),
            "delivery_use": [name for name, checkbox in self.delivery_checkboxes.items() if checkbox.isChecked()],
            "issue_tags": [name for name, checkbox in self.issue_checkboxes.items() if checkbox.isChecked()],
            "commercial_score": self.commercial_score_spin.value(),
            "portfolio_score": self.portfolio_score_spin.value(),
            "review_note": self.review_note_edit.toPlainText(),
            "review_status": status,
        }
        data["human_decision"] = json.dumps(
            {
                "photo_type": data["photo_type"],
                "subtype": data["subtype"],
                "quality_rating": data["quality_rating"],
                "delivery_use": data["delivery_use"],
                "issue_tags": data["issue_tags"],
                "commercial_score": data["commercial_score"],
                "portfolio_score": data["portfolio_score"],
                "review_note": data["review_note"],
                "review_status": status,
            },
            ensure_ascii=False,
        )
        return data

    def _apply_review_data_to_item(self, item: PhotoItem, review_data: dict) -> None:
        item.photo_type = str(review_data.get("photo_type") or "")
        item.subtype = str(review_data.get("subtype") or "")
        item.quality_rating = str(review_data.get("quality_rating") or "")
        item.delivery_use = list(review_data.get("delivery_use") or [])
        item.issue_tags = list(review_data.get("issue_tags") or [])
        item.commercial_score = float(review_data.get("commercial_score") or 0)
        item.portfolio_score = float(review_data.get("portfolio_score") or 0)
        item.review_note = str(review_data.get("review_note") or "")
        item.review_status = str(review_data.get("review_status") or REVIEW_STATUS_UNREVIEWED)
        item.human_decision = str(review_data.get("human_decision") or "")

    def save_manual_review(self, image_path: Path | str, review_data: dict) -> bool:
        errors = validate_review_fields(review_data)
        if errors:
            QMessageBox.warning(self, "人工判断未保存", "\n".join(errors))
            return False

        target = str(image_path)
        item = next((candidate for candidate in self.items if str(candidate.path) == target), None)
        if item is None:
            QMessageBox.warning(self, "人工判断未保存", "没有找到当前照片记录。")
            return False

        self._apply_review_data_to_item(item, review_data)
        try:
            self.save_item(item)
        except Exception as exc:
            self.logger.exception("保存人工筛片结果失败")
            QMessageBox.warning(self, "保存失败", f"人工判断保存失败：{exc}\n详情见 logs/app.log")
            return False

        self.model.refresh()
        self.update_export_stats()
        self.mark_dirty("保存人工筛片字段")
        return True

    def save_current_review(self, status: str = REVIEW_STATUS_HUMAN_CONFIRMED, quiet: bool = True) -> bool:
        if self.current_source_index < 0 or self.current_source_index >= len(self.items):
            return False
        item = self.items[self.current_source_index]
        review_data = self._collect_review_data(status)
        saved = self.save_manual_review(item.path, review_data)
        if saved:
            self._review_controls_dirty = False
            self.review_status_label.setText(f"系统状态：审片状态：{review_data['review_status']}")
            self.manual_label.setText(f"人工确认：{_review_type_text(item)} / {_quality_rating_text(item.quality_rating)}")
            self.preview_info_label.setText(self._preview_info_text(item))
            self._maybe_record_preference_sample(item, review_data)
            self._refresh_workflow_contexts()
            if not quiet:
                self.log(f"已保存人工判断：{item.filename}")
        return saved

    def _maybe_record_preference_sample(self, item: PhotoItem, review_data: dict) -> None:
        if not hasattr(self, "preference_record_checkbox") or not self.preference_record_checkbox.isChecked():
            return
        if review_data.get("review_status") != REVIEW_STATUS_HUMAN_CONFIRMED:
            return
        user_label = (
            review_data.get("quality_rating")
            or "、".join(review_data.get("delivery_use") or [])
            or review_data.get("photo_type")
            or "人工确认"
        )
        ai_suggestion = item.final_recommendation or item.ai_category_path or item.ai_primary_category or ""
        adopted = bool(ai_suggestion and (ai_suggestion in str(user_label) or str(user_label) in ai_suggestion))
        reason_tag = self.preference_reason_combo.currentText() if hasattr(self, "preference_reason_combo") else ""
        decision_key = json.dumps(
            {
                "path": str(item.path),
                "label": user_label,
                "reason": reason_tag,
                "ai": ai_suggestion,
                "delivery": review_data.get("delivery_use") or [],
                "issues": review_data.get("issue_tags") or [],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if decision_key == self._last_preference_record_key:
            return
        self._last_preference_record_key = decision_key
        record_preference_sample(
            item,
            str(user_label),
            f"人工修正；AI原建议：{ai_suggestion or '-'}；是否采纳AI：{'是' if adopted else '否'}；修改原因标签：{reason_tag or '-'}",
        )

    @Slot(str)
    def on_photo_type_changed(self, photo_type: str) -> None:
        selected = ""
        if not self._loading_details:
            subtypes = get_subtypes(photo_type)
            selected = subtypes[0] if subtypes else ""
        self._refresh_subtype_options(photo_type, selected)
        self.on_review_field_changed()

    @Slot()
    def on_review_field_changed(self) -> None:
        if self._loading_details or self.current_source_index < 0:
            return
        self._review_controls_dirty = True
        self.save_current_review(REVIEW_STATUS_HUMAN_CONFIRMED, quiet=True)

    @Slot()
    def confirm_review_and_next(self) -> None:
        if self.save_current_review(REVIEW_STATUS_HUMAN_CONFIRMED, quiet=False):
            row = self.model.row_for_source_index(self.current_source_index)
            if 0 <= row < self.model.rowCount() - 1:
                self.next_photo()
            else:
                self.log("已经是最后一张。")
                QMessageBox.information(self, "审片", "已经是最后一张。")

    @Slot()
    def mark_current_needs_review(self) -> None:
        if self.save_current_review(REVIEW_STATUS_NEEDS_REVIEW, quiet=False):
            self.log("已标记：待复核")

    @Slot()
    def reset_current_review(self) -> None:
        if self.current_source_index < 0 or self.current_source_index >= len(self.items):
            return
        item = self.items[self.current_source_index]
        answer = QMessageBox.question(
            self,
            "重置人工判断",
            "将清空当前照片的人工筛片字段，但不会删除 AI 建议或原图。是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        review_data = {
            "photo_type": "",
            "subtype": "",
            "quality_rating": "",
            "delivery_use": [],
            "issue_tags": [],
            "commercial_score": 0,
            "portfolio_score": 0,
            "review_note": "",
            "review_status": REVIEW_STATUS_UNREVIEWED,
            "human_decision": "",
        }
        if self.save_manual_review(item.path, review_data):
            self._load_review_controls(item)
            self.show_item(item)
            self.log(f"已重置人工判断：{item.filename}")

    def _has_current_photo(self) -> bool:
        return 0 <= self.current_source_index < len(self.items)

    def shortcut_set_quality_rating(self, rating_code: str) -> None:
        if not self._has_current_photo():
            return
        self.set_quality_rating(rating_code)
        if rating_code == "X" and "不导出" in self.delivery_checkboxes and not self.delivery_checkboxes["不导出"].isChecked():
            self.delivery_checkboxes["不导出"].setChecked(True)
        self.log(f"已设置：{_quality_rating_text(rating_code)}")

    def set_photo_type_by_shortcut(self, photo_type: str) -> None:
        if not self._has_current_photo():
            return
        if self.photo_type_combo.findText(photo_type) < 0:
            return
        self.photo_type_combo.setCurrentText(photo_type)
        self.save_current_review(REVIEW_STATUS_HUMAN_CONFIRMED, quiet=True)
        self.log(f"已设置照片类型：{photo_type}")

    def toggle_delivery_use(self, name: str) -> None:
        if not self._has_current_photo():
            return
        checkbox = self.delivery_checkboxes.get(name)
        if checkbox is None:
            return
        new_state = not checkbox.isChecked()
        checkbox.setChecked(new_state)
        self.log(f"已{'添加' if new_state else '移除'}交付用途：{name}")

    def toggle_issue_tag(self, name: str) -> None:
        if not self._has_current_photo():
            return
        checkbox = self.issue_checkboxes.get(name)
        if checkbox is None:
            return
        new_state = not checkbox.isChecked()
        checkbox.setChecked(new_state)
        self.log(f"已{'添加' if new_state else '移除'}问题标签：{name}")

    def show_shortcut_help(self) -> None:
        QMessageBox.information(self, "快捷键说明", SHORTCUT_HELP_TEXT)

    def _ai_suggestion_text(self, item: PhotoItem) -> str:
        status, loaded, message = self._ai_suggestion_state(item)
        if status == "current" and loaded:
            if "decision" in loaded:
                return compact_ai_suggestion_text(loaded)
            return "\n".join(f"{key}：{value}" for key, value in loaded.items()) or item.ai_suggestion
        if status in {"legacy", "stale", "mismatch", "invalid"}:
            version = loaded.get("version") or "legacy"
            analyzed_at = loaded.get("analyzed_at") or "-"
            return f"{message}\n\n当前记录版本：{version}\n分析时间：{analyzed_at}\n请点击“重新分析当前照片”刷新。"
        return message

    def _ai_suggestion_state(self, item: PhotoItem) -> tuple[str, dict, str]:
        status, loaded, message = ai_suggestion_status(item.ai_suggestion, item.path)
        if status == "mismatch":
            self.logger.warning(
                "AI建议绑定异常：当前主图=%s 当前路径=%s AI建议路径=%s",
                item.filename,
                item.path,
                loaded.get("image_path") if loaded else "",
            )
        return status, loaded, message

    def _current_ai_payload(self, item: PhotoItem) -> dict:
        status, loaded, _message = self._ai_suggestion_state(item)
        return loaded if status == "current" else {}

    def _log_ai_binding(self, item: PhotoItem) -> None:
        status, loaded, _message = self._ai_suggestion_state(item)
        suggestion_path = str(loaded.get("image_path") or item.path if loaded else item.path)
        suggestion_name = Path(suggestion_path).name if suggestion_path else "-"
        self.logger.info("AI建议绑定检查：主图=%s AI建议=%s status=%s path=%s", item.filename, suggestion_name, status, item.path)

    def _preview_info_text(self, item: PhotoItem) -> str:
        return (
            f"{item.filename} | AI建议：{item.ai_category_path or item.final_recommendation or '-'} | "
            f"人工确认：{_review_type_text(item)} / {_quality_rating_text(item.quality_rating)} | "
            f"人工用途：{_join_values(item.delivery_use)} | 系统状态：{_review_status_text(item)}"
        )

    def show_item(self, item: PhotoItem) -> None:
        self._loading_details = True
        self.file_name_label.setText(item.filename)
        self.basic_info_label.setText(
            f"{item.resolution_text} · {item.file_size_mb:.2f} MB · 拍摄时间：{item.taken_at or '-'}"
        )
        total = len(self.items)
        current = self.current_source_index + 1 if 0 <= self.current_source_index < total else 0
        self.review_sequence_label.setText(f"当前序号：第 {current} 张 / 共 {total} 张" if current else f"当前序号：- / 共 {total} 张")
        self.review_status_label.setText(f"系统状态：审片状态：{_review_status_text(item)}")
        self.manual_label.setText(f"人工确认：{_review_type_text(item)} / {_quality_rating_text(item.quality_rating)}")
        self.final_label.setText(f"系统导出分类（人工优先）：{item.export_category}")
        self.notes_edit.setPlainText(item.user_note)
        self.review_ai_suggestion_label.setText(self._ai_suggestion_text(item))
        self.reanalyze_current_suggestion_button.setEnabled(True)
        self._log_ai_binding(item)
        self._load_review_controls(item)
        self._show_ai_details(item)
        self._update_mark_buttons(item.manual_category)
        self.load_single_preview(item)
        self.preview_info_label.setText(self._preview_info_text(item))
        self.update_current_status()
        self._refresh_workflow_contexts()
        self.update_ai_result_overview()
        self._loading_details = False

    def update_current_status(self) -> None:
        total = len(self.items)
        if total <= 0:
            self.current_status_label.setText("当前：第 0 张 / 共 0 张")
            self._refresh_workflow_contexts()
            return
        if not (0 <= self.current_source_index < total):
            self.current_status_label.setText(f"当前：未选择 / 共 {total} 张")
            self._refresh_workflow_contexts()
            return
        current = max(1, min(self.current_source_index + 1, total))
        self.current_status_label.setText(f"当前：第 {current} 张 / 共 {total} 张")
        self._refresh_workflow_contexts()

    def _show_ai_details(self, item: PhotoItem) -> None:
        ai_payload = self._current_ai_payload(item)
        analysis_for_modules = ai_payload if ai_payload else None
        if ai_payload:
            self.ai_labels["ai_path"].setText(f"AI业务类型建议：{item.ai_category_path or '-'}")
            self.ai_labels["primary"].setText(f"AI一级建议：{item.ai_primary_category or '-'}")
            self.ai_labels["secondary"].setText(f"AI二级建议：{item.ai_secondary_category or '-'}")
            self.ai_labels["quality"].setText(f"AI质量标签：{', '.join(item.ai_quality_tags) or '-'}")
            self.ai_labels["confidence"].setText(f"置信度：{item.ai_confidence_text}")
            self.ai_labels["source"].setText(f"识别来源：{item.ai_source or '-'}")
            self.ai_labels["top3"].setText(f"语义Top3：{', '.join(item.top3_semantic_matches) or '-'}")
            self.ai_labels["semantic_score"].setText(f"语义分：{item.semantic_score:.1f} / 置信度 {item.semantic_confidence:.2f}" if item.semantic_score or item.semantic_confidence else "语义分：-")
            self.ai_labels["final_pick_score"].setText(f"可用性分：{item.final_pick_score:.1f}" if item.final_pick_score else "可用性分：-")
            self.ai_labels["detected_people"].setText(f"检测人数：{item.detected_person_count or item.people_count or '-'}")
            self.ai_labels["yolo_confidence"].setText(f"YOLO置信度：{item.detection_confidence:.2f}" if item.detection_confidence else "YOLO置信度：-")
            self.ai_labels["subject_integrity"].setText(
                f"主体面积：{item.subject_area_ratio:.2f} · 居中：{item.subject_center_score:.2f} · 边缘风险：{item.edge_cutoff_risk:.2f}"
                if item.subject_area_ratio or item.subject_center_score or item.edge_cutoff_risk
                else "主体完整性：-"
            )
            self.ai_labels["retouchable"].setText(f"是否可修：{'是' if item.retouch_suggestion else '-'}")
            self.ai_labels["croppable"].setText(f"是否可裁切：{'是' if item.crop_suggestion else '-'}")
        else:
            self.ai_labels["ai_path"].setText("AI业务类型建议：-")
            self.ai_labels["primary"].setText("AI一级建议：-")
            self.ai_labels["secondary"].setText("AI二级建议：-")
            self.ai_labels["quality"].setText("AI质量标签：-")
            self.ai_labels["confidence"].setText("置信度：-")
            self.ai_labels["source"].setText("识别来源：-")
            self.ai_labels["top3"].setText("语义Top3：-")
            self.ai_labels["semantic_score"].setText("语义分：-")
            self.ai_labels["final_pick_score"].setText("可用性分：-")
            self.ai_labels["detected_people"].setText("检测人数：-")
            self.ai_labels["yolo_confidence"].setText("YOLO置信度：-")
            self.ai_labels["subject_integrity"].setText("主体完整性：-")
            self.ai_labels["retouchable"].setText("是否可修：-")
            self.ai_labels["croppable"].setText("是否可裁切：-")
        status_text = {
            "best": "组内最佳",
            "backup": "组内备选",
            "duplicate": "重复淘汰",
            "review": "待复核",
            "rejected": "组内淘汰",
        }.get(item.similar_group_status, item.similar_group_status or "-")
        human_group = (
            f"人工相似组 {item.similar_group_id}（本组共 {item.similar_group_size} 张，人工状态：{status_text}，"
            f"AI组内最佳建议：{'是' if item.ai_recommended_best else '否'}）"
            if item.similar_group_id
            else "人工相似组：-"
        )
        auto_group = (
            f"AI相似候选组 {item.auto_group_id}（共 {item.auto_group_size} 张，置信度 {item.auto_group_confidence:.2f}，"
            f"方法：{item.grouping_method or '-'}）"
            if item.auto_group_id
            else "AI相似候选组：-"
        )
        self.ai_labels["similar"].setText(f"{auto_group}\n{human_group}")
        self.ai_labels["absolute_score"].setText(f"绝对质量分：{item.absolute_quality_score:.1f}" if item.absolute_quality_score else "绝对质量分：-")
        self.ai_labels["relative_score"].setText(f"相对质量分：{item.relative_quality_score:.1f}" if item.relative_quality_score else "相对质量分：-")
        rank = f"{item.group_rank}/{item.group_size}" if item.group_rank and item.group_size else "-"
        self.ai_labels["group_rank"].setText(f"相似组排名：{rank}")
        self.ai_labels["recommended"].setText(f"本组推荐保留：{'是' if item.recommended_keep or item.recommended_in_group else '否'}")
        self.ai_labels["ai_reason"].setText(f"AI判断原因：{item.ai_reason or '-'}" if ai_payload else "AI判断原因：-")
        self.ai_labels["preference_score"].setText(f"用户偏好模型分数：{item.user_preference_score:.2f}" if item.user_preference_score else "用户偏好模型分数：-")
        self.ai_labels["preference_reason"].setText(f"用户偏好判断原因：{item.preference_reason or item.user_preference_reason or '-'}")
        self.ai_labels["strategy"].setText(f"筛片策略：{self.strategy_combo.currentText()}")
        self.ai_labels["final_reason"].setText(f"最终建议原因：{item.final_reason or item.ai_reason or '-'}" if ai_payload else "最终建议原因：-")
        self.ai_labels["screening_reason"].setText(f"筛选原因：{item.screening_reason or item.final_reason or item.ai_reason or '-'}" if ai_payload else "筛选原因：-")
        self.ai_labels["style_label"].setText(f"风格判断：{item.style_label or '-'}" if ai_payload else "风格判断：-")
        self.ai_labels["retouch_suggestion"].setText(f"修图建议：{item.retouch_suggestion or '-'}" if ai_payload else "修图建议：-")
        self.ai_labels["crop_suggestion"].setText(f"裁切建议：{item.crop_suggestion or '-'}" if ai_payload else "裁切建议：-")
        self.ai_labels["portfolio_suggestion"].setText(f"作品建议：{item.portfolio_suggestion or '-'}" if ai_payload else "作品建议：-")
        self.ai_labels["delivery_suggestion"].setText(f"交付建议：{item.delivery_suggestion or '-'}" if ai_payload else "交付建议：-")
        self.ai_labels["final_recommendation"].setText(f"AI评分建议：{item.final_recommendation or '-'}" if ai_payload else "AI评分建议：-")
        self.ai_labels["module_basic"].setText("基础质量：\n" + module_summary_text(analysis_for_modules, "basic_quality"))
        self.ai_labels["module_face"].setText("人脸状态：\n" + module_summary_text(analysis_for_modules, "face"))
        self.ai_labels["module_pose"].setText("肢体姿态：\n" + module_summary_text(analysis_for_modules, "pose"))
        self.ai_labels["module_expression"].setText("表情状态：\n" + module_summary_text(analysis_for_modules, "expression"))
        self.ai_labels["module_composition"].setText("构图分析：\n" + module_summary_text(analysis_for_modules, "composition"))
        self.ai_labels["module_business"].setText("业务类型：\n" + module_summary_text(analysis_for_modules, "business_type"))
        self.ai_labels["module_decision"].setText("综合建议：\n" + module_summary_text(analysis_for_modules, "decision"))
        self.ai_labels["similar_history"].setText(
            f"相似历史样本：喜欢 {item.aesthetic_like_similarity:.2f} / 不喜欢 {item.aesthetic_dislike_similarity:.2f} / 参考 {item.similar_reference_count} 张"
            if item.similar_reference_count or item.aesthetic_like_similarity or item.aesthetic_dislike_similarity
            else "相似历史样本：-"
        )
        self.show_group_button.setEnabled(bool(item.similar_group_id))
        self.workflow_show_group_button.setEnabled(bool(item.similar_group_id))
        self.recommended_keep_button.setEnabled(bool(item.similar_group_id))
        self.group_backup_quick_button.setEnabled(bool(item.similar_group_id))
        self.group_duplicate_quick_button.setEnabled(bool(item.similar_group_id))

    def load_single_preview(self, item: PhotoItem) -> None:
        reader = QImageReader(str(item.path))
        reader.setAutoTransform(True)
        width, height = preview_pixmap_source_size(item.path, self.config.preview_max_size)
        reader.setScaledSize(QSize(width, height))
        image = reader.read()
        if image.isNull():
            self.preview_label.setText("无法预览该图片")
            self.current_preview_pixmap = None
            return
        self.current_preview_pixmap = QPixmap.fromImage(image)
        self.zoom_factor = 0.0
        self.fit_preview()

    def fit_preview(self) -> None:
        if self.center_stack.currentIndex() == 1:
            self.compare_zoom_factor = 1.0
            self.update_compare_view()
            return
        if not self.current_preview_pixmap:
            return
        self.zoom_factor = 0.0
        viewport = self.preview_scroll.viewport().size()
        pixmap = self.current_preview_pixmap.scaled(viewport, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(pixmap)

    def actual_preview(self) -> None:
        if self.center_stack.currentIndex() == 1:
            self.compare_zoom_factor = 1.35
            self.update_compare_view()
            return
        if self.current_preview_pixmap:
            self.zoom_factor = 1.0
            self.preview_label.setPixmap(self.current_preview_pixmap)

    def zoom_preview(self, factor: float) -> None:
        if self.center_stack.currentIndex() == 1:
            self.compare_zoom_factor = max(0.5, min(2.5, self.compare_zoom_factor * factor))
            self.update_compare_view()
            return
        if not self.current_preview_pixmap:
            return
        current = self.preview_label.pixmap()
        if current is None or current.isNull():
            current = self.current_preview_pixmap
        self.zoom_factor = max(0.1, self.zoom_factor * factor) if self.zoom_factor else factor
        new_size = QSize(max(1, int(current.width() * factor)), max(1, int(current.height() * factor)))
        self.preview_label.setPixmap(self.current_preview_pixmap.scaled(new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_compare_view(self, source_indexes: list[int] | None = None) -> None:
        if source_indexes is None:
            source_indexes = self._selected_source_indexes(self.photo_view)
        source_indexes = [index for index in source_indexes if 0 <= index < len(self.items)]
        if len(source_indexes) > self.config.max_compare_images:
            self.log(f"最多支持同时对比 {self.config.max_compare_images} 张，请减少选择数量。已显示前 {self.config.max_compare_images} 张。")
            QMessageBox.information(self, "多图对比", f"最多支持同时对比 {self.config.max_compare_images} 张，请减少选择数量。")
            source_indexes = source_indexes[: self.config.max_compare_images]
        if source_indexes and self.focus_source_index not in source_indexes:
            self.focus_source_index = source_indexes[0]

        tile_size = QSize(int(620 * self.compare_zoom_factor), int(420 * self.compare_zoom_factor))
        display_count = len(source_indexes)
        for tile_index, (tile, source_index) in enumerate(zip(self.compare_tiles, source_indexes + [-1] * 4)):
            tile.setVisible(tile_index < max(1, display_count))
            if source_index < 0 or source_index >= len(self.items):
                tile.set_item(None, -1, None, False)
                continue
            item = self.items[source_index]
            pixmap = self._preview_pixmap_for_item(item, tile_size)
            tile.set_item(item, source_index, pixmap, source_index == self.focus_source_index)

    def _preview_pixmap_for_item(self, item: PhotoItem, size: QSize) -> QPixmap:
        reader = QImageReader(str(item.path))
        reader.setAutoTransform(True)
        width, height = preview_pixmap_source_size(item.path, self.config.preview_max_size)
        reader.setScaledSize(QSize(width, height))
        image = reader.read()
        if image.isNull():
            return QPixmap()
        return QPixmap.fromImage(image).scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _update_mark_buttons(self, category: str) -> None:
        for button, value in [
            (self.selected_button, MANUAL_SELECTED),
            (self.pending_button, MANUAL_PENDING),
            (self.rejected_button, MANUAL_REJECTED),
            (self.retouch_button, MANUAL_RETOUCH),
            (self.crop_button, MANUAL_CROP),
            (self.portfolio_button, MANUAL_PORTFOLIO),
            (self.practice_button, MANUAL_PRACTICE),
            (self.workflow_selected_button, MANUAL_SELECTED),
            (self.workflow_pending_button, MANUAL_PENDING),
            (self.workflow_rejected_button, MANUAL_REJECTED),
            (self.workflow_retouch_button, MANUAL_RETOUCH),
            (self.workflow_crop_button, MANUAL_CROP),
            (self.workflow_portfolio_button, MANUAL_PORTFOLIO),
            (self.workflow_practice_button, MANUAL_PRACTICE),
            (self.quick_selected_button, MANUAL_SELECTED),
            (self.quick_retouch_button, MANUAL_RETOUCH),
            (self.quick_pending_button, MANUAL_PENDING),
            (self.quick_rejected_button, MANUAL_REJECTED),
            (self.quick_duplicate_button, MANUAL_CROP),
        ]:
            button.setObjectName("ActiveMark" if category == value else "")
            button.style().unpolish(button)
            button.style().polish(button)

    def mark_focused(self, category: str) -> None:
        if self.center_stack.currentIndex() == 2:
            source_indexes = self._selected_source_indexes(self.grid_view)
            if source_indexes:
                self.mark_source_indexes(source_indexes, category)
                return
        source_indexes = self._selected_source_indexes(self.photo_view)
        if len(source_indexes) > 1 and self.center_stack.currentIndex() != 1:
            self.mark_source_indexes(source_indexes, category)
            return
        self.mark_source_index(self.focus_source_index if self.focus_source_index >= 0 else self.current_source_index, category)

    def _add_unique_value(self, values: list[str], value: str) -> list[str]:
        result = list(values or [])
        if value and value not in result:
            result.append(value)
        return result

    def _apply_legacy_category_to_review(self, item: PhotoItem, category: str) -> None:
        item.review_status = REVIEW_STATUS_HUMAN_CONFIRMED
        if category == MANUAL_SELECTED:
            item.quality_rating = "S"
            item.delivery_use = self._add_unique_value(item.delivery_use, "精修候选")
        elif category == MANUAL_RETOUCH:
            item.quality_rating = "A"
            item.delivery_use = self._add_unique_value(item.delivery_use, "客户可选")
        elif category == MANUAL_PENDING:
            item.quality_rating = item.quality_rating or "B"
            item.review_status = REVIEW_STATUS_NEEDS_REVIEW
        elif category == MANUAL_CROP:
            item.issue_tags = self._add_unique_value(item.issue_tags, "重复照片")
            item.review_status = REVIEW_STATUS_NEEDS_REVIEW
        elif category == MANUAL_REJECTED:
            item.quality_rating = "X"
            item.delivery_use = self._add_unique_value(item.delivery_use, "不导出")
        elif category == MANUAL_PORTFOLIO:
            item.quality_rating = item.quality_rating or "S"
            item.delivery_use = self._add_unique_value(item.delivery_use, "作品集候选")
        elif category == MANUAL_PRACTICE:
            item.quality_rating = item.quality_rating or "C"
            item.delivery_use = self._add_unique_value(item.delivery_use, "修图练习")
        item.human_decision = json.dumps(
            {
                "legacy_category": category,
                "photo_type": item.photo_type,
                "subtype": item.subtype,
                "quality_rating": item.quality_rating,
                "delivery_use": item.delivery_use,
                "issue_tags": item.issue_tags,
                "commercial_score": item.commercial_score,
                "portfolio_score": item.portfolio_score,
                "review_note": item.review_note,
                "review_status": item.review_status,
            },
            ensure_ascii=False,
        )

    def mark_source_indexes(self, source_indexes: list[int], category: str) -> None:
        changed: list[PhotoItem] = []
        for source_index in source_indexes:
            if source_index < 0 or source_index >= len(self.items):
                continue
            item = self.items[source_index]
            item.manual_category = category
            item.manual_override = True
            item.user_label = category
            item.user_label_time = datetime.now().isoformat(timespec="seconds")
            self._apply_legacy_category_to_review(item, category)
            item.compute_final_category(self.config.confidence_threshold)
            record_preference_sample(item, category, "用户批量标记")
            changed.append(item)
        if self.repository and changed:
            self.repository.save_items(changed)
        self.model.refresh()
        if self.current_source_index >= 0:
            self.show_item(self.items[self.current_source_index])
        self.update_compare_view()
        self.update_export_stats()
        self.mark_dirty(f"批量修改手动分类：{len(changed)} 张")

    @Slot(int, str)
    def mark_source_index(self, source_index: int, category: str) -> None:
        if source_index < 0 or source_index >= len(self.items):
            return
        item = self.items[source_index]
        item.manual_category = category
        item.manual_override = True
        item.user_label = category
        item.user_label_time = datetime.now().isoformat(timespec="seconds")
        self._apply_legacy_category_to_review(item, category)
        item.compute_final_category(self.config.confidence_threshold)
        record_preference_sample(item, category, "用户手动标记")
        self.save_item(item)
        self.model.refresh()
        self.current_source_index = source_index
        self.focus_source_index = source_index
        self.show_item(item)
        self.update_compare_view()
        self.update_export_stats()
        self.mark_dirty("修改手动分类")

    def save_item(self, item: PhotoItem) -> None:
        if self.repository:
            self.repository.save_item(item)

    def save_current_item(self) -> None:
        self.save_all(auto=False)

    @Slot()
    def on_notes_changed(self) -> None:
        if self._loading_details or self.current_source_index < 0:
            return
        item = self.items[self.current_source_index]
        item.user_note = self.notes_edit.toPlainText()
        item.review_note = item.review_note or item.user_note
        self.mark_dirty("修改备注")

    def start_batch_ai(self) -> None:
        self.reanalyze_unconfirmed_photos()

    def reanalyze_unconfirmed_photos(self) -> None:
        if not self.items:
            QMessageBox.information(self, "重新分析", "请先导入照片。")
            return
        if self.background_task_manager.is_running():
            QMessageBox.information(self, "重新分析", "后台任务正在运行，请等待完成或先停止当前任务。")
            return
        human_confirmed = sum(1 for item in self.items if item.review_status == REVIEW_STATUS_HUMAN_CONFIRMED)
        target_count = len(self.items) - human_confirmed
        if target_count <= 0:
            QMessageBox.information(self, "重新分析", "当前项目没有需要重新分析的未人工确认照片。")
            return
        self.background_task_manager.set_tasks(build_background_tasks([TASK_AI_PRECLASSIFY], self.items))
        started = self.background_task_manager.start(self.items, self.config, self.selected_folder)
        if started:
            self.stage_label.setText("阶段：重新分析未人工确认照片")
            self.log(
                f"已开始重新分析未人工确认照片：{target_count} 张；人工已确认 {human_confirmed} 张会跳过字段覆盖。"
            )
            self._refresh_background_buttons(TASK_STATUS_RUNNING)

    def start_model_auto_group_ui(self) -> None:
        if not self.items:
            QMessageBox.information(self, "AI相似候选组", "请先导入照片。")
            return
        if self.background_task_manager.is_running():
            QMessageBox.information(self, "AI相似候选组", "后台任务正在运行，请等待完成或先停止当前任务。")
            return
        reply = QMessageBox.question(
            self,
            "生成 AI 相似候选组",
            "模型自动分组结果只是选片建议，不会覆盖人工相似组状态，也不会移动或删除原图。\n\n是否开始生成 AI 相似候选组？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.background_task_manager.set_tasks(build_background_tasks([TASK_MODEL_AUTO_GROUP], self.items))
        started = self.background_task_manager.start(self.items, self.config, self.selected_folder)
        if started:
            self._activate_workflow_section("ai", persist=False)
            self.stage_label.setText("阶段：生成 AI 相似候选组")
            self.log("已开始生成 AI 相似候选组；结果仅写入 auto_group_* 建议字段。")
            self._refresh_background_buttons(TASK_STATUS_RUNNING)

    def start_current_ai(self) -> None:
        if self.current_source_index < 0:
            return
        item = self.items[self.current_source_index]
        try:
            self.stage_label.setText("阶段：多模型分析")
            self.current_file_label.setText(f"当前文件：{item.filename}")
            self.work_progress.setValue(0)
            analysis = run_analyzer_pipeline(item.path, item)
            apply_analysis_to_item(item, analysis)
            self.save_item(item)
            self.model.refresh()
            self.show_item(item)
            self.update_export_stats()
            self.work_progress.setValue(100)
            self.count_label.setText("已完成：1 / 1")
            self.mark_dirty("执行单张多模型分析")
            self.log(f"多模型分析完成：{item.filename}")
        except Exception as exc:
            self.logger.exception("单张多模型分析失败")
            QMessageBox.warning(self, "分析失败", f"{exc}\n详情见 logs/analyzer.log。")

    def _clear_stale_ai_suggestion_fields(self, item: PhotoItem) -> bool:
        status, _loaded, _message = ai_suggestion_status(item.ai_suggestion, item.path)
        if status not in {"legacy", "stale", "mismatch", "invalid"}:
            return False
        item.ai_suggestion = ""
        item.ai_primary_category = ""
        item.ai_secondary_category = ""
        item.ai_quality_tags = []
        item.ai_confidence = 0.0
        item.ai_source = ""
        item.ai_reason = ""
        return True

    def clear_stale_ai_suggestions(self) -> None:
        if not self.items:
            QMessageBox.information(self, "清理旧版 AI 建议", "请先导入照片。")
            return
        stale_items = [item for item in self.items if ai_suggestion_status(item.ai_suggestion, item.path)[0] in {"legacy", "stale", "mismatch", "invalid"}]
        if not stale_items:
            QMessageBox.information(self, "清理旧版 AI 建议", f"当前没有旧版 AI 建议。当前版本：{CURRENT_AI_SUGGESTION_VERSION}")
            return
        reply = QMessageBox.question(
            self,
            "清理旧版 AI 建议",
            "这会清空旧版 AI 建议，但不会删除原图，也不会清空人工筛片结果。是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        changed = [item for item in stale_items if self._clear_stale_ai_suggestion_fields(item)]
        if self.repository and changed:
            self.repository.save_items(changed)
        self.model.refresh()
        if self.current_source_index >= 0:
            self.show_item(self.items[self.current_source_index])
        self.update_export_stats()
        self.mark_dirty("清理旧版 AI 建议")
        self.log(f"已清理旧版 AI 建议：{len(changed)} 张")
        QMessageBox.information(self, "清理完成", f"已清理旧版 AI 建议：{len(changed)} 张。")

    def _start_ai_for_paths(self, paths: list[Path]) -> None:
        self.current_ai_total = len(paths)
        self.config = load_config()
        self._refresh_ai_status()
        if not self.config.enable_ai:
            QMessageBox.information(self, "AI 已关闭", "config.json 中 enable_ai=false。")
            return
        if not self.ai_env.ai_ready:
            QMessageBox.warning(self, "AI依赖缺失", self.ai_env.message)
            return
        self._set_ai_running(True)
        self.work_progress.setValue(0)
        self.stage_label.setText("阶段：AI分析")
        self.ai_thread = QThread(self)
        self.ai_worker = AIWorker(paths, self.config)
        self.ai_worker.moveToThread(self.ai_thread)
        self.ai_thread.started.connect(self.ai_worker.run)
        self.ai_worker.progress.connect(self.on_ai_progress)
        self.ai_worker.log.connect(self.log)
        self.ai_worker.finished.connect(self.on_ai_finished)
        self.ai_worker.failed.connect(self.on_ai_failed)
        self.ai_worker.finished.connect(lambda *_args: self.ai_thread.quit())
        self.ai_worker.failed.connect(lambda *_args: self.ai_thread.quit())
        self.ai_thread.finished.connect(self.ai_worker.deleteLater)
        self.ai_thread.finished.connect(self.ai_thread.deleteLater)
        self.ai_thread.finished.connect(self._clear_ai_refs)
        self.ai_thread.start()

    @Slot()
    def cancel_ai(self) -> None:
        if self.ai_worker:
            self.ai_worker.cancel()
            self.cancel_ai_button.setEnabled(False)

    @Slot(int, int, str)
    def on_ai_progress(self, done: int, total: int, filename: str) -> None:
        self.stage_label.setText("阶段：AI分析")
        self.current_file_label.setText(f"当前文件：{filename}")
        self.count_label.setText(f"已处理：{max(0, done)} / {max(0, total)}")
        self.work_progress.setValue(int(done / total * 100) if total else 0)

    @Slot(object, bool, object)
    def on_ai_finished(self, results: list, cancelled: bool, performance: dict) -> None:
        self.last_ai_performance = performance or {}
        by_path = {str(item.path): item for item in self.items}
        changed: list[PhotoItem] = []
        for result in results:
            item = by_path.get(str(result.path))
            if not item:
                continue
            apply_ai_result_to_item(item, result, self.config.confidence_threshold)
            changed.append(item)
        if self.repository and changed:
            self.repository.save_items(changed)
        self.model.refresh()
        self._set_ai_running(False)
        if self.current_source_index >= 0:
            self.show_item(self.items[self.current_source_index])
        self.update_compare_view()
        self._refresh_ai_status()
        self.mark_dirty("执行AI分析")
        self.log(f"AI分析完成：{len(results)} 张{'（已取消）' if cancelled else ''}")
        total = self.current_ai_total or len(results)
        self.count_label.setText(f"已完成：{len(results)} / {total}")
        self.work_progress.setValue(100 if not cancelled and total else self.work_progress.value())
        self.update_export_stats()
        self.show_analysis_completion(cancelled)
        self._warn_if_ai_pending_too_high()

    @Slot(str)
    def on_ai_failed(self, message: str) -> None:
        self._set_ai_running(False)
        QMessageBox.warning(self, "AI分析失败", f"{message}\n详情见 logs/app.log")
        self.log(f"AI分析失败：{message}")

    def rescore_existing_ai(self) -> None:
        if not self.items:
            return
        changed = []
        for item in self.items:
            old_final = item.final_category
            item.compute_final_category(self.config.confidence_threshold)
            if item.final_category != old_final or item.ai_primary_category:
                changed.append(item)
        if self.repository and changed:
            self.repository.save_items(changed)
        self.model.refresh()
        if self.current_source_index >= 0:
            self.show_item(self.items[self.current_source_index])
        self.mark_dirty("重新评分")
        self.update_export_stats()
        self.log(f"重新评分完成：{len(changed)} 张")
        self.show_analysis_stats()

    def _analysis_counter(self) -> Counter:
        counter = Counter()
        for item in self.items:
            key = item.final_recommendation or item.export_category or item.final_category or "未分类"
            counter[key] += 1
        return counter

    def _ai_score_bucket(self, item: PhotoItem) -> str:
        if not (item.ai_primary_category or item.ai_confidence or item.final_pick_score or item.absolute_quality_score):
            return "无法判断"
        score = item.final_pick_score or item.absolute_quality_score or item.relative_quality_score or 0
        if score >= 90:
            return "S 强烈推荐"
        if score >= 75:
            return "A 可交付"
        if score >= 60:
            return "B 备选"
        if score >= 40:
            return "C 留档/练习"
        return "无法判断"

    def _ai_result_overview(self) -> dict:
        score_counter = Counter({"S 强烈推荐": 0, "A 可交付": 0, "B 备选": 0, "C 留档/练习": 0, "无法判断": 0})
        business_counter = Counter()
        auto_groups: dict[str, list[PhotoItem]] = {}
        for item in self.items:
            score_counter[self._ai_score_bucket(item)] += 1
            business_counter[item.ai_primary_category or "无法判断"] += 1
            if item.auto_group_id:
                auto_groups.setdefault(item.auto_group_id, []).append(item)
        grouped_photos = sum(len(group_items) for group_items in auto_groups.values())
        singleton_count = max(0, len(self.items) - grouped_photos)
        high_risk = sum(
            1
            for group_items in auto_groups.values()
            if any("high_risk" in (item.auto_group_reason or "") or "过度合并" in (item.auto_group_reason or "") for item in group_items)
        )
        return {
            "score_counter": score_counter,
            "business_counter": business_counter,
            "auto_group_count": len(auto_groups),
            "auto_group_photo_count": grouped_photos,
            "singleton_count": singleton_count,
            "high_risk_count": high_risk,
        }

    def update_ai_result_overview(self) -> None:
        if not hasattr(self, "ai_score_overview_label"):
            return
        overview = self._ai_result_overview()
        score_counter = overview["score_counter"]
        business_counter = overview["business_counter"]
        analyzed = sum(1 for item in self.items if item.ai_primary_category or item.ai_confidence or item.ai_suggestion)
        self.ai_progress_summary_label.setText(
            f"进度：已分析 {analyzed}/{len(self.items)}；后台任务会显示成功 / 跳过 / 失败、GPU/CPU 和 cache 命中。"
        )
        self.ai_score_overview_label.setText(
            "A. AI评分建议（非人工结论）：\n"
            f"S 强烈推荐 {score_counter['S 强烈推荐']}｜A 可交付 {score_counter['A 可交付']}｜"
            f"B 备选 {score_counter['B 备选']}｜C 留档/练习 {score_counter['C 留档/练习']}｜"
            f"无法判断 {score_counter['无法判断']}"
        )
        self.ai_group_overview_label.setText(
            "B. AI相似候选组（非人工重复）：\n"
            f"候选组 {overview['auto_group_count']}｜候选组照片 {overview['auto_group_photo_count']}｜"
            f"单张照片 {overview['singleton_count']}｜高风险组 {overview['high_risk_count']}"
        )
        business_lines = "｜".join(f"{name} {count}" for name, count in business_counter.most_common(6))
        self.ai_business_overview_label.setText(f"C. 业务类型建议（非人工分类）：\n{business_lines or '暂无'}")

    def _workflow_category_counter(self) -> Counter:
        counter = Counter({"强烈推荐": 0, "可交付": 0, "备选": 0, "重复": 0, "废片": 0})
        for item in self.items:
            text = " / ".join(
                [
                    item.manual_category or "",
                    item.final_recommendation or "",
                    item.export_category or "",
                    item.final_category or "",
                    item.ai_primary_category or "",
                ]
            )
            score = item.final_pick_score or item.absolute_quality_score or item.relative_quality_score or 0
            if item.manual_category == MANUAL_REJECTED or "明确废片" in text or "废片" in text:
                counter["废片"] += 1
            elif item.manual_category == MANUAL_CROP or item.ai_primary_category == PRIMARY_DUPLICATE or item.similar_group_id or "重复" in text:
                counter["重复"] += 1
            elif item.manual_category in {MANUAL_SELECTED, MANUAL_PORTFOLIO} or score >= 90 or "推荐保留" in text or "精修" in text:
                counter["强烈推荐"] += 1
            elif item.manual_category == MANUAL_RETOUCH or score >= 75 or "可用待修" in text or "可交付" in text:
                counter["可交付"] += 1
            else:
                counter["备选"] += 1
        return counter

    def _analysis_stats_text(self, elapsed_ms: float | None = None) -> str:
        overview = self._ai_result_overview()
        score_counter = overview["score_counter"]
        elapsed_text = "-"
        if elapsed_ms is not None and elapsed_ms > 0:
            seconds = int(elapsed_ms / 1000)
            elapsed_text = f"{seconds // 60:02d}:{seconds % 60:02d}"
        lines = [
            f"总照片数：{len(self.items)}",
            f"AI S 强烈推荐建议：{score_counter['S 强烈推荐']}",
            f"AI A 可交付建议：{score_counter['A 可交付']}",
            f"AI B 备选建议：{score_counter['B 备选']}",
            f"AI C 留档/练习建议：{score_counter['C 留档/练习']}",
            f"AI 无法判断：{score_counter['无法判断']}",
            f"AI相似候选组：{overview['auto_group_count']} 组 / {overview['auto_group_photo_count']} 张",
            f"用时：{elapsed_text}",
        ]
        return "\n".join(lines)

    def show_analysis_completion(self, cancelled: bool) -> None:
        elapsed_ms = self.last_ai_performance.get("total_ms")
        text = self._analysis_stats_text(elapsed_ms)
        self.analysis_stats_label.setText("分析统计：\n" + text.replace("\n", "    "))
        self.update_ai_result_overview()
        title = "AI分析已取消" if cancelled else "AI分析完成"
        QMessageBox.information(self, title, text)

    def _warn_if_ai_pending_too_high(self) -> None:
        if not self.items:
            return
        pending_count = sum(1 for item in self.items if item.manual_category == MANUAL_PENDING)
        ratio = pending_count / len(self.items)
        if ratio > self.config.max_pending_ratio:
            QMessageBox.information(
                self,
                "AI判断偏保守",
                f"当前备选比例为 {ratio:.0%}，高于配置上限 {self.config.max_pending_ratio:.0%}。\n"
                "建议降低置信度阈值、切换交付策略，或优先使用“人工复核 / 可交付 / 重复 / 废片疑似”。",
            )

    def show_analysis_stats(self) -> None:
        if not self.items:
            QMessageBox.information(self, "分析统计", "当前还没有导入照片。")
            return
        counter = self._analysis_counter()
        total = len(self.items)
        overview = self._ai_result_overview()
        analyzed = sum(1 for item in self.items if item.ai_primary_category or item.ai_confidence)
        workflow_text = self._analysis_stats_text(self.last_ai_performance.get("total_ms"))
        lines = [
            "AI分析统计（建议，不是人工确认）：",
            workflow_text,
            "",
            f"总照片数：{total}",
            f"已 AI 分析：{analyzed}",
            f"AI相似候选组：{overview['auto_group_count']} 组 / {overview['auto_group_photo_count']} 张",
            f"人工相似组照片：{sum(1 for item in self.items if item.similar_group_id)}",
            "",
            "分类统计：",
        ]
        for name, count in counter.most_common():
            lines.append(f"- {name}：{count} 张")
        pending_count = sum(1 for item in self.items if item.manual_category == MANUAL_PENDING)
        lines.extend(
            [
                "",
                f"人工备选比例：{pending_count / total:.0%}",
                f"目标精选比例：{self.config.target_pick_ratio:.0%}",
                "提示：AI结果只是建议；人工确认后才会成为交付、备选、重复或废片等工作流结论。",
            ]
        )
        QMessageBox.information(self, "AI分析统计", "\n".join(lines))

    @Slot()
    def _clear_ai_refs(self) -> None:
        self.ai_thread = None
        self.ai_worker = None

    def _set_ai_running(self, running: bool) -> None:
        self.ai_analyze_button.setEnabled(not running and bool(self.items))
        self.current_ai_button.setEnabled(not running and bool(self.items))
        self.model_auto_group_button.setEnabled(not running and bool(self.items))
        self.reanalyze_unconfirmed_button.setEnabled(not running and bool(self.items))
        self.clear_stale_ai_button.setEnabled(not running and bool(self.items))
        self.reanalyze_current_suggestion_button.setEnabled(not running and bool(self.items))
        self.cancel_ai_button.setEnabled(running)
        self.select_folder_button.setEnabled(not running)
        self.export_button.setEnabled(not running and bool(self.items))

    def _open_similarity_tab(self) -> None:
        if hasattr(self, "main_stack") and hasattr(self, "similar_group_panel"):
            self._activate_workflow_section("similar", persist=False)
            self.refresh_similarity_panel()
            self._save_layout_state()

    def refresh_similarity_panel(self, selected_group_id: str | None = None) -> None:
        if hasattr(self, "similar_group_panel"):
            self.similar_group_panel.set_items(
                self.items,
                current_index=self.current_source_index,
                selected_group_id=selected_group_id,
            )

    def _items_for_group(self, group_id: str) -> list[PhotoItem]:
        return [item for item in self.items if item.similar_group_id == group_id]

    def rebuild_similarity_group_list(self, selected_group_id: str | None = None) -> None:
        self.refresh_similarity_panel(selected_group_id)

    def calculate_similarity_groups_ui(self) -> None:
        if not self.items:
            QMessageBox.information(self, "相似组", "请先导入照片。")
            return
        if self.background_task_manager.is_running():
            QMessageBox.information(self, "相似组", "后台任务正在运行中，请稍后再计算相似照片。")
            return
        if any(item.similar_group_id for item in self.items):
            reply = QMessageBox.question(
                self,
                "重新计算相似组",
                "重新计算会更新相似组编号、相似度和 AI 推荐结果，但不会覆盖你已经人工设置的组内最佳、备选、重复淘汰和待复核状态。是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        self.background_task_manager.set_tasks(build_background_tasks([TASK_SIMILAR_PHOTOS], self.items))
        started = self.background_task_manager.start(self.items, self.config, self.selected_folder)
        if started:
            self._open_similarity_tab()
            self.stage_label.setText("阶段：计算相似照片")
            self._refresh_background_buttons(TASK_STATUS_RUNNING)
            self.log("已开始后台计算相似照片。")

    def show_current_group(self) -> None:
        if self.current_source_index < 0:
            return
        group_id = self.items[self.current_source_index].similar_group_id
        if not group_id:
            return
        source_indexes = [index for index, item in enumerate(self.items) if item.similar_group_id == group_id]
        self._open_similarity_tab()
        if hasattr(self, "similar_group_panel"):
            self.similar_group_panel.focus_group(group_id, self.current_source_index)
        self.view_combo.setCurrentText("多图对比")
        self.update_compare_view(source_indexes)
        self.log(f"查看相似组 {group_id}，共 {len(source_indexes)} 张")

    def set_recommended_keep(self) -> None:
        if self.focus_source_index < 0 or self.focus_source_index >= len(self.items):
            return
        item = self.items[self.focus_source_index]
        if not item.similar_group_id:
            return
        changed = set_group_status(self.items, item, "best")
        if self.repository:
            self.repository.save_items(changed)
        self.model.refresh()
        self.refresh_similarity_panel(item.similar_group_id)
        self.show_item(item)
        self.update_compare_view()
        self.mark_dirty("设置推荐保留")
        self.log(f"相似组 {item.similar_group_id} 推荐保留：{item.filename}")

    def set_current_group_status(self, status: str) -> None:
        source_index = self.focus_source_index if 0 <= self.focus_source_index < len(self.items) else self.current_source_index
        if source_index < 0 or source_index >= len(self.items):
            return
        item = self.items[source_index]
        if not item.similar_group_id:
            QMessageBox.information(self, "相似组", "当前照片暂未进入相似组。")
            return
        changed = set_group_status(self.items, item, status)
        self.on_similar_group_status_changed(changed, source_index)

    @Slot(object, int)
    def on_similar_group_status_changed(self, changed: list[PhotoItem], source_index: int) -> None:
        if self.repository and changed:
            self.repository.save_items(changed)
        self.model.refresh()
        if 0 <= source_index < len(self.items):
            self.current_source_index = source_index
            self.focus_source_index = source_index
            self.show_item(self.items[source_index])
            group_id = self.items[source_index].similar_group_id
            self.refresh_similarity_panel(group_id)
        else:
            self.refresh_similarity_panel()
        self.update_compare_view()
        self.update_export_stats()
        self.mark_dirty("修改相似组人工状态")
        self.log("相似组人工状态已保存。")

    def record_current_preference(self) -> None:
        if self.current_source_index < 0:
            return
        item = self.items[self.current_source_index]
        reason_tag = self.preference_reason_combo.currentText() if hasattr(self, "preference_reason_combo") else ""
        user_label = item.quality_rating or item.manual_category or _join_values(item.delivery_use) or item.photo_type or "人工确认"
        ai_suggestion = item.final_recommendation or item.ai_category_path or item.ai_primary_category or ""
        adopted = bool(ai_suggestion and (ai_suggestion in user_label or user_label in ai_suggestion))
        record_preference_sample(
            item,
            user_label,
            f"用户点击记录偏好；AI原建议：{ai_suggestion or '-'}；是否采纳AI：{'是' if adopted else '否'}；修改原因标签：{reason_tag or '-'}",
        )
        item.user_label = user_label
        item.user_label_time = datetime.now().isoformat(timespec="seconds")
        self.save_item(item)
        self.mark_dirty("记录用户偏好标签")
        self._refresh_ai_status()
        QMessageBox.information(self, "偏好已记录", "当前照片已加入偏好学习样本。")

    def train_preference_model_ui(self) -> None:
        ok, message = train_preference_model()
        self.log(message)
        if ok:
            self.mark_dirty("训练偏好模型")
            self._refresh_ai_status()
            QMessageBox.information(self, "偏好模型", message)
        else:
            QMessageBox.warning(self, "偏好模型", message)

    def apply_preference_model(self) -> None:
        if not self.items:
            return
        changed = []
        for item in self.items:
            prediction = predict_preference(item)
            item.user_preference_score = prediction.score
            item.user_preference_reason = prediction.reason
            item.preference_reason = prediction.reason
            item.preference_model_version = "local-preference-v1" if prediction.label else ""
            if prediction.label and prediction.score >= 0.60:
                item.final_reason = f"{item.ai_reason}；偏好模型建议：{prediction.label}，偏好分 {prediction.score:.2f}。"
            elif prediction.label:
                item.final_reason = f"{item.ai_reason}；偏好模型不确定，建议人工确认。"
            changed.append(item)
        if self.repository:
            self.repository.save_items(changed)
        self.model.refresh()
        if self.current_source_index >= 0:
            self.show_item(self.items[self.current_source_index])
        self.mark_dirty("使用偏好模型重新评分")
        QMessageBox.information(self, "偏好模型", "已根据本地偏好模型更新偏好分和建议原因。")

    def clear_preference_model_ui(self) -> None:
        clear_preference_model()
        self.mark_dirty("清空偏好模型")
        QMessageBox.information(self, "偏好模型", "偏好模型已清空，偏好样本仍保留。")

    def export_preference_data_ui(self) -> None:
        from app.core.preference_learning import SAMPLES_PATH

        QMessageBox.information(self, "偏好数据", f"偏好样本文件：\n{SAMPLES_PATH}")

    def show_preference_report(self) -> None:
        summary = preference_summary()
        report = preference_report()
        missing = max(0, 50 - summary.get("total", 0))
        prefix = "我的筛片风格\n\n"
        prefix += f"当前人工偏好样本：{summary.get('total', 0)} 张\n"
        prefix += f"训练条件：{'已达到' if summary.get('can_train') else f'还差 {missing} 张'}\n\n"
        if summary.get("warnings"):
            prefix += "样本质量提示：\n" + "\n".join(f"- {item}" for item in summary["warnings"]) + "\n\n"
        QMessageBox.information(self, "我的筛片风格报告", prefix + report)

    def add_aesthetic_folder_ui(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择审美素材库文件夹")
        if not folder:
            return
        library = add_aesthetic_folder(Path(folder))
        folders = [entry.get("path", "") for entry in library.get("folders", []) if entry.get("path")]
        self.config = update_config({"aesthetic_library_folders": folders})
        self.mark_dirty("添加审美素材库")
        QMessageBox.information(self, "审美素材库", f"已添加素材文件夹：\n{folder}\n\n不会自动爬取网络照片，只扫描你选择的本地文件夹。")

    def scan_aesthetic_library_ui(self) -> None:
        library = scan_aesthetic_library()
        count = len(library.get("images", []))
        self.mark_dirty("扫描审美素材库")
        QMessageBox.information(self, "审美素材库", f"扫描完成：{count} 张素材。\n后续可用于建立本地审美参考库和偏好对比。")

    def show_aesthetic_report_ui(self) -> None:
        QMessageBox.information(self, "审美素材库报告", aesthetic_report())

    def change_model_profile(self, profile: str) -> None:
        if not profile:
            return
        self.model_manager = ModelManager(self.config)
        self.config = self.model_manager.apply_profile(profile)
        self._refresh_ai_status()
        self.mark_dirty("切换AI模型档位")
        self.log(f"AI模型档位切换为：{profile}，batch_size={self.config.batch_size}")

    def show_model_status(self) -> None:
        status = ModelManager(self.config).status()
        QMessageBox.information(
            self,
            "AI模型状态",
            f"当前档位：{status.model_profile}\n"
            f"语义模型：{status.semantic_model}\n"
            f"语义状态：{status.semantic_status}\n"
            f"检测模型：{status.detector_model}\n"
            f"检测状态：{status.detector_status}\n"
            f"运行设备：{status.device}\n"
            f"GPU：{status.gpu_name or '-'}\n"
            f"batch_size：{status.batch_size}\n"
            f"use_fp16：{status.use_fp16}\n"
            f"显存：{gpu_memory_text()}\n"
            f"模型目录占用：{status.model_size_text}\n"
            f"embedding缓存：{status.embedding_cache_count}\n"
            f"缓存目录：{status.cache_dir}",
        )

    def download_recommended_models(self) -> None:
        if self.model_download_thread:
            return
        if not self.config.allow_large_model_download:
            QMessageBox.warning(self, "模型下载已禁用", "config.json 中 allow_large_model_download=false。")
            return
        reply = QMessageBox.question(
            self,
            "下载推荐模型",
            "即将下载开源预训练模型，可能占用数 GB 空间并需要较长时间。是否继续？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._set_model_download_running(True)
        self.model_download_thread = QThread(self)
        self.model_download_worker = ModelDownloadWorker(self.config)
        self.model_download_worker.moveToThread(self.model_download_thread)
        self.model_download_thread.started.connect(self.model_download_worker.run)
        self.model_download_worker.progress.connect(self.on_model_download_progress)
        self.model_download_worker.finished.connect(self.on_model_download_finished)
        self.model_download_worker.failed.connect(self.on_model_download_failed)
        self.model_download_worker.finished.connect(lambda *_args: self.model_download_thread.quit())
        self.model_download_worker.failed.connect(lambda *_args: self.model_download_thread.quit())
        self.model_download_thread.finished.connect(self.model_download_worker.deleteLater)
        self.model_download_thread.finished.connect(self.model_download_thread.deleteLater)
        self.model_download_thread.finished.connect(self._clear_model_download_refs)
        self.model_download_thread.start()

    @Slot(str, int, str)
    def on_model_download_progress(self, stage: str, percent: int, detail: str) -> None:
        self.stage_label.setText(f"阶段：{stage}")
        self.current_file_label.setText(f"模型：{detail}")
        self.work_progress.setValue(percent)
        self.log(f"{stage}：{detail}")

    @Slot()
    def cancel_model_download(self) -> None:
        if self.model_download_worker:
            self.model_download_worker.cancel()
            self.cancel_model_download_button.setEnabled(False)
            self.log("已请求取消模型下载。正在运行的单个下载步骤可能需要等待当前网络请求结束。")

    @Slot(object)
    def on_model_download_finished(self, status) -> None:
        self._set_model_download_running(False)
        self._refresh_ai_status()
        self.mark_dirty("下载AI模型")
        try:
            classifier = LocalAIClassifier(self.config)
            test_text = f"\n模型测试通过：{classifier.backend_label}"
        except Exception as exc:
            test_text = f"\n模型测试失败：{exc}"
        QMessageBox.information(
            self,
            "模型下载完成",
            f"语义模型：{status.semantic_status}\n检测模型：{status.detector_status}\n当前设备：{status.device}{test_text}",
        )

    @Slot(str)
    def on_model_download_failed(self, message: str) -> None:
        self._set_model_download_running(False)
        self._refresh_ai_status()
        QMessageBox.warning(self, "模型下载失败", f"{message}\n请检查网络或代理，详情见 logs/app.log。")

    @Slot()
    def _clear_model_download_refs(self) -> None:
        self.model_download_thread = None
        self.model_download_worker = None

    def _set_model_download_running(self, running: bool) -> None:
        self.download_model_button.setEnabled(not running)
        self.cancel_model_download_button.setEnabled(running)
        self.select_folder_button.setEnabled(not running)
        self.ai_analyze_button.setEnabled(not running and bool(self.items))
        self.model_auto_group_button.setEnabled(not running and bool(self.items))
        self.reanalyze_unconfirmed_button.setEnabled(not running and bool(self.items))
        self.clear_stale_ai_button.setEnabled(not running and bool(self.items))

    def test_models_ui(self) -> None:
        try:
            classifier = LocalAIClassifier(self.config)
            QMessageBox.information(
                self,
                "模型测试",
                f"AI管线可用。\nbackend：{classifier.backend_label}\n设备：{classifier.device_label}\nbatch_size：{classifier.batch_size}",
            )
        except Exception as exc:
            QMessageBox.warning(self, "模型测试失败", f"{exc}\n详情见 logs/app.log")

    def detect_gpu_ui(self) -> None:
        self._refresh_ai_status()
        QMessageBox.information(self, "GPU检测", f"{self.ai_env.message}\n显存：{gpu_memory_text()}")

    def _sample_paths(self, limit: int = 10) -> list[Path]:
        selected_rows = self.photo_view.selectionModel().selectedRows() if self.photo_view.selectionModel() else []
        paths: list[Path] = []
        for index in selected_rows[:limit]:
            source_index = self.model.source_index_at_row(index.row())
            if 0 <= source_index < len(self.items):
                paths.append(self.items[source_index].path)
        if paths:
            return paths
        return [item.path for item in self.items[:limit]]

    def start_model_self_test(self) -> None:
        self._start_diagnostics("self_test")

    def start_auto_tune(self) -> None:
        self._start_diagnostics("tune")

    def _start_diagnostics(self, mode: str) -> None:
        if self.diagnostics_thread:
            QMessageBox.information(self, "模型任务", "已有模型自测或自动调优任务正在运行。")
            return
        paths = self._sample_paths(10)
        if not paths:
            QMessageBox.information(self, "模型任务", "请先导入照片。模型自测会随机/顺序抽取当前项目样本运行。")
            return
        self._set_diagnostics_running(True)
        self.work_progress.setValue(0)
        self.stage_label.setText("阶段：模型自测" if mode == "self_test" else "阶段：自动调优")
        self.diagnostics_thread = QThread(self)
        self.diagnostics_worker = DiagnosticsWorker(mode, self.config, paths)
        self.diagnostics_worker.moveToThread(self.diagnostics_thread)
        self.diagnostics_thread.started.connect(self.diagnostics_worker.run)
        self.diagnostics_worker.progress.connect(self.on_diagnostics_progress)
        self.diagnostics_worker.finished.connect(self.on_diagnostics_finished)
        self.diagnostics_worker.failed.connect(self.on_diagnostics_failed)
        self.diagnostics_worker.finished.connect(lambda *_args: self.diagnostics_thread.quit())
        self.diagnostics_worker.failed.connect(lambda *_args: self.diagnostics_thread.quit())
        self.diagnostics_thread.finished.connect(self.diagnostics_worker.deleteLater)
        self.diagnostics_thread.finished.connect(self.diagnostics_thread.deleteLater)
        self.diagnostics_thread.finished.connect(self._clear_diagnostics_refs)
        self.diagnostics_thread.start()

    @Slot(str, int, str)
    def on_diagnostics_progress(self, stage: str, percent: int, detail: str) -> None:
        self.stage_label.setText(f"阶段：{stage}")
        self.current_file_label.setText(detail)
        self.work_progress.setValue(percent)
        self.log(f"{stage}：{detail}")

    @Slot(str, object)
    def on_diagnostics_finished(self, mode: str, result: object) -> None:
        self._set_diagnostics_running(False)
        if mode == "tune":
            self.config = load_config()
            self._sync_config_controls()
            self._refresh_ai_status()
            self.mark_dirty("自动性能调优")
        report = result.get("report", str(result)) if isinstance(result, dict) else str(result)
        QMessageBox.information(self, "模型自测" if mode == "self_test" else "自动调优", report)

    @Slot(str)
    def on_diagnostics_failed(self, message: str) -> None:
        self._set_diagnostics_running(False)
        QMessageBox.warning(self, "模型任务失败", f"{message}\n详情见 logs/app.log")

    def _set_diagnostics_running(self, running: bool) -> None:
        self.model_self_test_button.setEnabled(not running)
        self.auto_tune_button.setEnabled(not running)
        self.select_folder_button.setEnabled(not running)

    @Slot()
    def _clear_diagnostics_refs(self) -> None:
        self.diagnostics_thread = None
        self.diagnostics_worker = None

    def clear_model_cache_ui(self) -> None:
        reply = QMessageBox.question(self, "清理模型缓存", "将清理 models/cache，不会删除已下载模型。是否继续？", QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        ModelManager(self.config).clear_model_cache()
        self._refresh_ai_status()
        QMessageBox.information(self, "模型缓存", "模型缓存已清理。")

    def clear_embedding_cache_ui(self) -> None:
        reply = QMessageBox.question(self, "清理embedding缓存", "将清理 cache/embeddings，后续AI分析会重新计算。是否继续？", QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        ModelManager(self.config).clear_embedding_cache()
        self._refresh_ai_status()
        QMessageBox.information(self, "embedding缓存", "embedding缓存已清理。")

    def show_ai_performance_stats(self) -> None:
        analyzed = [item for item in self.items if item.ai_runtime_ms or item.ai_primary_category]
        cached = sum(1 for item in analyzed if item.embedding_cached)
        avg_ms = sum(item.ai_runtime_ms for item in analyzed) / len(analyzed) if analyzed else 0
        perf = self.last_ai_performance or {}
        avg_text = perf.get("avg_ms", avg_ms)
        QMessageBox.information(
            self,
            "AI性能统计",
            f"模式：{self.ai_env.message}\n"
            f"batch_size：{self.config.batch_size}\n"
            f"use_fp16：{self.config.use_fp16}\n"
            f"模型档位：{self.config.model_profile}\n"
            f"显存：{gpu_memory_text()}\n"
            f"已分析：{len(analyzed)} 张\n"
            f"平均耗时：{avg_text:.1f} ms/张\n"
            f"embedding缓存命中：{perf.get('embedding_cache_hits', cached)} 张\n"
            f"新计算：{perf.get('embedding_new', max(0, len(analyzed) - cached))} 张\n"
            f"fallback：{perf.get('fallback', 0)} 张\n\n"
            f"档位建议：{model_profile_advice(self.config, avg_text)}\n\n"
            f"{gpu_low_usage_explanation()}",
        )

    def _default_export_dir(self) -> Path | None:
        if self.export_output_dir:
            return self.export_output_dir
        if not self.selected_folder:
            return None
        return self.selected_folder.parent / "PhotoSelect_Output"

    def choose_export_path(self) -> None:
        start = str(self._default_export_dir() or Path.home())
        folder = QFileDialog.getExistingDirectory(self, "选择导出文件夹", start)
        if not folder:
            return
        self.export_output_dir = Path(folder)
        self.export_path_edit.setText(str(self.export_output_dir))
        self.update_export_stats()
        self.mark_dirty("修改导出路径")

    def open_export_folder(self) -> None:
        output_dir = self._default_export_dir()
        if not output_dir:
            QMessageBox.information(self, "打开导出文件夹", "请先选择照片文件夹或导出路径。")
            return
        if not output_dir.exists():
            QMessageBox.information(self, "打开导出文件夹", f"导出文件夹尚不存在：\n{output_dir}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_dir)))

    def open_report_file(self) -> None:
        if not self.last_report_path or not self.last_report_path.exists():
            QMessageBox.information(self, "打开报告文件", "还没有生成报告。")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_report_path)))

    def _export_options_from_ui(self, mode_override: str | None = None) -> ExportOptions:
        mode = mode_override or self.export_mode_combo.currentText()
        return ExportOptions(
            mode=mode,
            organize_by_use=self.export_use_dirs_checkbox.isChecked(),
            organize_by_type=self.export_type_dirs_checkbox.isChecked(),
            include_rejects=self.export_include_rejects_checkbox.isChecked(),
            include_no_export=self.export_include_no_export_checkbox.isChecked(),
            generate_csv=self.export_csv_checkbox.isChecked() or "报告" in mode,
            generate_markdown=self.export_md_checkbox.isChecked() or "报告" in mode,
        )

    def update_export_stats(self) -> None:
        if not hasattr(self, "export_stats_label"):
            return
        output_dir = self._default_export_dir()
        output_text = str(output_dir) if output_dir else "尚未选择项目"
        if not hasattr(self, "export_path_edit") or not self.export_path_edit.text():
            if output_dir:
                self.export_path_edit.setText(output_text)
        if not self.items:
            self.export_stats_label.setText(f"导出路径：{output_text}\n导出统计：尚无照片。")
            return
        summary = build_export_summary(self.items)
        lines = [
            f"导出路径：{output_text}",
            f"导出方式：{self.export_mode_combo.currentText() if hasattr(self, 'export_mode_combo') else '复制原图 + 生成报告'}",
            "导出统计预览：",
            f"- 总照片数：{summary['total']} 张",
            f"- 客户可选：{summary['client_select_count']} 张",
            f"- 精修候选：{summary['retouch_candidate_count']} 张",
            f"- 作品集候选：{summary['portfolio_candidate_count']} 张",
            f"- 不导出：{summary['by_delivery_use'].get('不导出', 0)} 张",
            f"- X 废片：{summary['reject_count']} 张",
            f"- 待复核：{summary['needs_review_count']} 张",
            f"- 相似组：{summary.get('similar_group_count', 0)} 组，重复淘汰：{summary.get('duplicate_count', 0)} 张",
        ]
        self.export_stats_label.setText("\n".join(lines))

    def export_csv_report_only(self) -> None:
        if not self.selected_folder or not self.items:
            QMessageBox.information(self, "导出CSV报告", "请先导入照片。")
            return
        try:
            for item in self.items:
                item.compute_final_category(self.config.confidence_threshold)
            if self.repository:
                self.repository.save_items(self.items)
            report_path = write_current_report(self.selected_folder, self.items, self._default_export_dir())
            self.last_report_path = Path(report_path)
            self.open_report_button.setEnabled(True)
            self.export_output_dir = self.last_report_path.parents[1]
            self.export_path_edit.setText(str(self.export_output_dir))
            QMessageBox.information(self, "导出报告", f"报告已生成：\n{report_path}\n\n此操作没有复制或移动原图。")
            self.log(f"报告已生成：{report_path}")
            self.update_export_stats()
        except Exception as exc:
            self.logger.exception("CSV报告导出失败")
            QMessageBox.warning(self, "导出CSV报告失败", f"{exc}\n详情见 logs/app.log")

    def confirm_and_export(self) -> None:
        if not self.selected_folder or not self.items:
            return
        mode = self.export_mode_combo.currentText()
        output_dir = self._default_export_dir()
        summary = build_export_summary(self.items)
        lines = ["本次将导出：", ""]
        lines.extend(
            [
                f"- 总照片数：{summary['total']} 张",
                f"- 客户可选：{summary['client_select_count']} 张",
                f"- 精修候选：{summary['retouch_candidate_count']} 张",
                f"- 作品集候选：{summary['portfolio_candidate_count']} 张",
                f"- 不导出：{summary['by_delivery_use'].get('不导出', 0)} 张",
                f"- X 废片：{summary['reject_count']} 张",
                f"- 待复核：{summary['needs_review_count']} 张",
            ]
        )
        lines.append("")
        lines.append(f"导出路径：{output_dir}")
        lines.append(f"导出方式：{mode}")
        lines.append("原图不会被删除；默认只复制，同名文件会自动改名，避免覆盖。")
        reply = QMessageBox.question(self, "导出预览", "\n".join(lines), QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self._start_export()

    def _start_export(self) -> None:
        if self.repository:
            self.repository.save_items(self.items)
        self._set_export_running(True)
        self.active_export_task = self.background_task_manager.track_external_task(TASK_EXPORT, TASK_NAMES[TASK_EXPORT], len(self.items))
        self._refresh_background_buttons(TASK_STATUS_RUNNING)
        self.export_thread = QThread(self)
        self.export_worker = ExportWorker(
            self.selected_folder,
            list(self.items),
            self.config,
            self._default_export_dir(),
            self._export_options_from_ui(),
        )
        self.export_worker.moveToThread(self.export_thread)
        self.export_thread.started.connect(self.export_worker.run)
        self.export_worker.progress.connect(self.on_export_progress)
        self.export_worker.finished.connect(self.on_export_finished)
        self.export_worker.failed.connect(self.on_export_failed)
        self.export_worker.finished.connect(lambda *_args: self.export_thread.quit())
        self.export_worker.failed.connect(lambda *_args: self.export_thread.quit())
        self.export_thread.finished.connect(self.export_worker.deleteLater)
        self.export_thread.finished.connect(self.export_thread.deleteLater)
        self.export_thread.finished.connect(self._clear_export_refs)
        self.export_thread.start()

    @Slot(int, int, str, str)
    def on_export_progress(self, done: int, total: int, filename: str, target: str) -> None:
        self.stage_label.setText("阶段：导出")
        self.current_file_label.setText(f"当前文件：{filename}")
        self.count_label.setText(f"已处理：{max(0, done)} / {max(0, total)}")
        self.work_progress.setValue(int(done / total * 100) if total else 0)
        if self.active_export_task:
            self.active_export_task.total_count = total
            self.background_task_manager.update_external_task(self.active_export_task, done, filename)
        if target:
            self.log(f"导出：{filename} -> {target}")

    @Slot(str, str, object)
    def on_export_finished(self, output_dir: str, report_path: str, result: object) -> None:
        self._set_export_running(False)
        self.export_output_dir = Path(output_dir)
        self.export_path_edit.setText(str(self.export_output_dir))
        self.last_report_path = Path(report_path)
        self.open_report_button.setEnabled(bool(self.last_report_path.exists()))
        success = getattr(result, "success_count", 0)
        skipped = getattr(result, "skipped_count", 0)
        failed = getattr(result, "failed_count", 0)
        total = success + skipped + failed
        status = TASK_STATUS_STOPPED if getattr(result, "cancelled", False) else TASK_STATUS_COMPLETED
        self.background_task_manager.finish_external_task(
            self.active_export_task,
            status=status,
            success=success,
            skipped=skipped,
            failed=failed,
            error_message="用户取消导出" if getattr(result, "cancelled", False) else "",
        )
        self.active_export_task = None
        self._refresh_background_buttons(status)
        self.count_label.setText(f"已完成：{total} / {total}，成功 {success}，跳过 {skipped}，失败 {failed}")
        self.work_progress.setValue(100)
        self.update_export_stats()
        QMessageBox.information(self, "导出完成", f"导出完成：成功 {success} 张，跳过 {skipped} 张，失败 {failed} 张。\n\n{output_dir}\n\n报告：{report_path}")
        self.log(f"导出完成：成功 {success}，跳过 {skipped}，失败 {failed}，报告 {report_path}")

    @Slot(str)
    def on_export_failed(self, message: str) -> None:
        self._set_export_running(False)
        self.background_task_manager.finish_external_task(
            self.active_export_task,
            status=TASK_STATUS_ERROR,
            failed=len(self.items),
            error_message=message,
        )
        self.active_export_task = None
        self._refresh_background_buttons(TASK_STATUS_ERROR)
        QMessageBox.critical(self, "导出失败", f"{message}\n详情见 logs/app.log")
        self.log(f"导出失败：{message}")

    def _set_export_running(self, running: bool) -> None:
        self.export_button.setEnabled(not running and bool(self.items))
        self.export_csv_button.setEnabled(not running and bool(self.items))
        self.cancel_export_button.setEnabled(running)
        self.choose_export_path_button.setEnabled(not running)
        self.open_export_folder_button.setEnabled(not running)
        self.open_report_button.setEnabled(not running and bool(self.last_report_path and self.last_report_path.exists()))
        self.select_folder_button.setEnabled(not running)
        self.ai_analyze_button.setEnabled(not running and bool(self.items))
        self.model_auto_group_button.setEnabled(not running and bool(self.items))
        self.reanalyze_unconfirmed_button.setEnabled(not running and bool(self.items))
        self.clear_stale_ai_button.setEnabled(not running and bool(self.items))
        self.reanalyze_current_suggestion_button.setEnabled(not running and bool(self.items))

    def cancel_export(self) -> None:
        if self.export_worker:
            self.export_worker.cancel()
            self.cancel_export_button.setEnabled(False)
            self.stage_label.setText("阶段：正在取消导出")

    @Slot()
    def _clear_export_refs(self) -> None:
        self.export_thread = None
        self.export_worker = None

    def previous_photo(self) -> None:
        if self._review_controls_dirty:
            self.save_current_review(REVIEW_STATUS_HUMAN_CONFIRMED, quiet=True)
        row = self.model.row_for_source_index(self.current_source_index)
        if row > 0:
            self.select_source_index(self.model.source_index_at_row(row - 1))
        elif self.items:
            self.log("已经是第一张。")

    def next_photo(self) -> None:
        if self._review_controls_dirty:
            self.save_current_review(REVIEW_STATUS_HUMAN_CONFIRMED, quiet=True)
        row = self.model.row_for_source_index(self.current_source_index)
        if 0 <= row < self.model.rowCount() - 1:
            self.select_source_index(self.model.source_index_at_row(row + 1))
        elif self.items:
            self.log("已经是最后一张。")

    def focus_current_photo(self) -> None:
        if self.current_source_index >= 0:
            self.focus_source_index = self.current_source_index
            self.view_combo.setCurrentText("单图审片")

    def toggle_fit_actual(self) -> None:
        if self.center_stack.currentIndex() == 1:
            if self.compare_zoom_factor > 1.05:
                self.fit_preview()
            else:
                self.actual_preview()
            return
        if self.zoom_factor == 1.0:
            self.fit_preview()
        else:
            self.actual_preview()

    def select_all_visible(self) -> None:
        active_view = self.grid_view if self.center_stack.currentIndex() == 2 else self.photo_view
        active_view.selectAll()

    def focus_next_compare_tile(self) -> None:
        if self.center_stack.currentIndex() != 1:
            return
        source_indexes = [tile.source_index for tile in self.compare_tiles if tile.source_index >= 0]
        if not source_indexes:
            return
        if self.focus_source_index not in source_indexes:
            self.focus_source_index = source_indexes[0]
        else:
            self.focus_source_index = source_indexes[(source_indexes.index(self.focus_source_index) + 1) % len(source_indexes)]
        self.update_compare_view(source_indexes)

    @Slot(int)
    def set_focus_source_index(self, source_index: int) -> None:
        if 0 <= source_index < len(self.items):
            self.focus_source_index = source_index
            self.current_source_index = source_index
            self.show_item(self.items[source_index])
            self.update_compare_view()

    def escape_action(self) -> None:
        if self.center_stack.currentIndex() == 1:
            self.view_combo.setCurrentText("单图审片")
        elif self.import_worker:
            self.cancel_import()
        elif self.ai_worker:
            self.cancel_ai()

    def _background_mode_value(self) -> str:
        text = self.background_start_mode_combo.currentText()
        return {
            "手动启动": "manual",
            "软件空闲时询问我": "ask_when_idle",
            "自动运行，低优先级": "auto_low_priority",
        }.get(text, "manual")

    def _selected_background_task_types(self) -> list[str]:
        return [task_type for task_type, checkbox in self.background_task_checkboxes.items() if checkbox.isChecked()]

    def change_background_task_options(self) -> None:
        self.config = update_config(
            {
                "background_task_start_mode": self._background_mode_value(),
                "background_task_enabled_types": self._selected_background_task_types(),
                "enable_idle_tasks": self._background_mode_value() != "manual",
                "enable_idle_similarity": TASK_SIMILAR_PHOTOS in self._selected_background_task_types(),
            }
        )
        self.background_ask_prompted = False
        self.mark_dirty("修改后台任务设置")
        self.log(f"后台任务启动方式：{self.background_start_mode_combo.currentText()}")

    def start_background_tasks(self) -> None:
        if not self.items:
            QMessageBox.information(self, "后台任务", "请先导入照片。")
            return
        if self.background_task_manager.is_running():
            QMessageBox.information(self, "后台任务", "后台任务正在运行中。")
            return
        task_types = self._selected_background_task_types()
        if not task_types:
            QMessageBox.information(self, "后台任务", "请至少勾选一个任务类型。")
            return
        pending_task_types = {
            task.task_type
            for task in self.background_task_manager.tasks
            if task.status in {TASK_STATUS_PENDING, TASK_STATUS_STOPPED}
        }
        if TASK_MODEL_AUTO_GROUP in set(task_types) | pending_task_types:
            reply = QMessageBox.question(
                self,
                "确认运行 AI 相似候选组",
                "模型自动分组会生成“AI 相似候选组”建议，用于把画面相似、构图或动作接近、需要互相比较的照片放在一起。\n\n"
                "这些结果只是建议，人工确认后才可采纳。\n"
                "本任务不会删除、移动原图，也不会覆盖已有人工相似组状态、组内最佳、人工分类、交付用途、问题标签或审片状态。\n\n"
                "是否继续运行？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                self.log("已取消 AI 相似候选组任务。")
                return
        pending_exists = bool(pending_task_types)
        if not pending_exists:
            self.background_task_manager.add_tasks(build_background_tasks(task_types, self.items))
        started = self.background_task_manager.start(self.items, self.config, self.selected_folder)
        if started:
            self.background_ask_prompted = False
            self.stage_label.setText("阶段：后台任务")
            self.log("后台任务已开始。")
            self._refresh_background_buttons(TASK_STATUS_RUNNING)
        else:
            self.log("没有可运行的后台任务。")

    def pause_background_tasks(self) -> None:
        self.background_task_manager.pause()
        self.log("已请求暂停后台任务，当前文件处理完成后暂停。")

    def resume_background_tasks(self) -> None:
        self.background_task_manager.resume()
        self.log("后台任务继续运行。")

    def stop_background_tasks(self) -> None:
        if self.export_worker:
            self.cancel_export()
        self.background_task_manager.stop()
        self.log("已请求停止后台任务，当前文件处理完成后停止。")

    def clear_background_task_queue(self) -> None:
        if self.background_task_manager.is_running() or self.export_worker:
            reply = QMessageBox.question(
                self,
                "清空队列",
                "当前有后台任务正在运行。请先停止任务，是否现在请求停止？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.stop_background_tasks()
            return
        self.background_task_manager.clear()
        self.background_progress.setValue(0)
        self.background_state_label.setText("当前状态：空闲")
        self.background_task_label.setText("当前任务：-")
        self.background_file_label.setText("当前文件：-")
        self.background_count_label.setText("进度：0 / 0")
        self.background_result_label.setText("成功：0  跳过：0  失败：0")
        self.background_elapsed_label.setText("用时：-")
        self.background_recent_log_label.setText("最近日志：队列已清空")
        if hasattr(self, "background_compact_label"):
            self.background_compact_label.setText("后台：空闲｜0/0｜成功 0｜失败 0")
        self.log("后台任务队列已清空。")
        self._refresh_background_buttons()

    def open_background_task_log(self) -> None:
        BACKGROUND_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not BACKGROUND_LOG_PATH.exists():
            BACKGROUND_LOG_PATH.write_text("", encoding="utf-8")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(BACKGROUND_LOG_PATH)))

    def show_background_tasks_dialog(self) -> None:
        dialog = getattr(self, "background_tasks_dialog", None)
        if dialog and dialog.isVisible():
            dialog.raise_()
            dialog.activateWindow()
            return
        dialog = QDialog(self)
        dialog.setObjectName("BackgroundTaskDialog")
        dialog.setWindowTitle("后台任务")
        dialog.resize(760, 560)
        layout = QVBoxLayout(dialog)
        status_grid = QGridLayout()
        self.dialog_background_state_label = QLabel(self.background_state_label.text())
        self.dialog_background_task_label = QLabel(self.background_task_label.text())
        self.dialog_background_file_label = QLabel(self.background_file_label.text())
        self.dialog_background_count_label = QLabel(self.background_count_label.text())
        self.dialog_background_result_label = QLabel(self.background_result_label.text())
        for row, label in enumerate(
            [
                self.dialog_background_state_label,
                self.dialog_background_task_label,
                self.dialog_background_file_label,
                self.dialog_background_count_label,
                self.dialog_background_result_label,
            ]
        ):
            label.setWordWrap(True)
            status_grid.addWidget(label, row, 0, 1, 2)
        layout.addLayout(status_grid)
        self.dialog_background_progress = QProgressBar()
        self.dialog_background_progress.setRange(0, 100)
        self.dialog_background_progress.setValue(self.background_progress.value())
        layout.addWidget(self.dialog_background_progress)

        button_row = QHBoxLayout()
        for text, callback in [
            ("立即运行", self.start_background_tasks),
            ("暂停", self.pause_background_tasks),
            ("继续", self.resume_background_tasks),
            ("停止", self.stop_background_tasks),
            ("清空队列", self.clear_background_task_queue),
            ("打开任务日志", self.open_background_task_log),
        ]:
            button = QPushButton(text)
            button.clicked.connect(callback)
            button_row.addWidget(button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("启动方式"))
        mode_combo = QComboBox()
        mode_combo.addItems(["手动启动", "软件空闲时询问我", "自动运行，低优先级"])
        mode_combo.setCurrentText(self.background_start_mode_combo.currentText())
        mode_combo.currentTextChanged.connect(self.background_start_mode_combo.setCurrentText)
        mode_row.addWidget(mode_combo)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        task_group = QGroupBox("任务类型")
        task_grid = QGridLayout(task_group)
        self.dialog_background_checkboxes: dict[str, QCheckBox] = {}
        for index, (task_type, source_checkbox) in enumerate(self.background_task_checkboxes.items()):
            checkbox = QCheckBox(source_checkbox.text())
            checkbox.setChecked(source_checkbox.isChecked())
            checkbox.toggled.connect(source_checkbox.setChecked)
            self.dialog_background_checkboxes[task_type] = checkbox
            task_grid.addWidget(checkbox, index // 2, index % 2)
        layout.addWidget(task_group)

        self.dialog_background_queue_edit = QTextEdit()
        self.dialog_background_queue_edit.setReadOnly(True)
        self.dialog_background_queue_edit.setMinimumHeight(140)
        self.dialog_background_queue_edit.setPlainText(self.background_queue_edit.toPlainText())
        layout.addWidget(self.dialog_background_queue_edit, 1)
        dialog.finished.connect(lambda _code: setattr(self, "background_tasks_dialog", None))
        self.background_tasks_dialog = dialog
        dialog.show()

    @Slot(object)
    def on_background_task_started(self, task: BackgroundTask) -> None:
        self._update_background_status(task)
        self._refresh_background_buttons(task.status)
        self.log(f"后台任务开始：{task.task_name}")

    @Slot(object)
    def on_background_task_progress(self, task: BackgroundTask) -> None:
        self._update_background_status(task)

    @Slot(object)
    def on_background_task_paused(self, task: BackgroundTask) -> None:
        self._update_background_status(task)
        self._refresh_background_buttons(TASK_STATUS_PAUSED)
        self.log(f"后台任务已暂停：{task.task_name}")

    @Slot(object)
    def on_background_task_resumed(self, task: BackgroundTask) -> None:
        self._update_background_status(task)
        self._refresh_background_buttons(TASK_STATUS_RUNNING)
        self.log(f"后台任务继续：{task.task_name}")

    @Slot(object)
    def on_background_task_stopping(self, task: BackgroundTask) -> None:
        self._update_background_status(task)
        self._refresh_background_buttons(TASK_STATUS_STOPPING)

    @Slot(object)
    def on_background_task_stopped(self, task: BackgroundTask) -> None:
        self._update_background_status(task)
        self._refresh_background_buttons(TASK_STATUS_STOPPED)
        self.log(f"后台任务已停止：{task.task_name}")

    @Slot(object)
    def on_background_task_completed(self, task: BackgroundTask) -> None:
        self._update_background_status(task)
        if task.task_type in {TASK_THUMBNAIL_CACHE, TASK_SIMILAR_PHOTOS, TASK_MODEL_AUTO_GROUP}:
            self.model.refresh()
            self.update_ai_result_overview()
        if task.task_type == TASK_SIMILAR_PHOTOS:
            self.refresh_similarity_panel()
        self.log(f"后台任务完成：{task.task_name}")

    @Slot(object)
    def on_background_task_error(self, task: BackgroundTask) -> None:
        self._update_background_status(task)
        self._refresh_background_buttons(TASK_STATUS_ERROR)
        QMessageBox.warning(self, "后台任务出错", f"{task.task_name} 出错：{task.error_message}\n详情见 logs/background_tasks.log。")

    @Slot(object)
    def on_background_all_finished(self, tasks: list[BackgroundTask]) -> None:
        if self.repository and self.items:
            try:
                self.repository.save_items(self.items)
            except Exception:
                self.logger.exception("后台任务结果保存失败")
        self.model.refresh()
        if self.current_source_index >= 0 and self.current_source_index < len(self.items):
            self.show_item(self.items[self.current_source_index])
        self.rebuild_similarity_group_list()
        self.update_export_stats()
        self.update_ai_result_overview()
        final_status = TASK_STATUS_COMPLETED
        if any(task.status == TASK_STATUS_ERROR for task in tasks):
            final_status = TASK_STATUS_ERROR
        elif any(task.status in {TASK_STATUS_STOPPED, TASK_STATUS_CANCELLED} for task in tasks):
            final_status = TASK_STATUS_STOPPED
        self._refresh_background_buttons(final_status)
        self.stage_label.setText(f"阶段：后台任务{self._task_status_text(final_status)}")
        self.log("后台任务队列已结束。")

    @Slot(object)
    def refresh_background_queue_view(self, tasks: list[BackgroundTask]) -> None:
        if not tasks:
            self.background_queue_edit.setPlainText("队列为空。")
            if hasattr(self, "dialog_background_queue_edit"):
                self.dialog_background_queue_edit.setPlainText("队列为空。")
            return
        lines = ["状态 | 任务类型 | 进度 | 成功 | 跳过 | 失败"]
        lines.append("--- | --- | ---: | ---: | ---: | ---:")
        for task in tasks:
            lines.append(
                f"{self._task_status_text(task.status)} | {task.task_name} | {task.progress_text()} | "
                f"{task.success_count} | {task.skipped_count} | {task.failed_count}"
            )
        self.background_queue_edit.setPlainText("\n".join(lines))
        if hasattr(self, "dialog_background_queue_edit"):
            self.dialog_background_queue_edit.setPlainText("\n".join(lines))

    def _update_background_status(self, task: BackgroundTask) -> None:
        self.background_state_label.setText(f"当前状态：{self._task_status_text(task.status)}")
        self.background_task_label.setText(f"当前任务：{task.task_name}")
        self.background_file_label.setText(f"当前文件：{task.current_file or '-'}")
        self.background_count_label.setText(f"进度：{task.completed_count} / {task.total_count}")
        self.background_result_label.setText(f"成功：{task.success_count}  跳过：{task.skipped_count}  失败：{task.failed_count}")
        self.background_elapsed_label.setText(f"用时：{self._task_elapsed_text(task)}")
        if hasattr(self, "background_compact_label"):
            self.background_compact_label.setText(
                f"后台：{self._task_status_text(task.status)}｜{task.task_name}｜"
                f"进度 {task.completed_count}/{task.total_count}｜成功 {task.success_count}｜"
                f"跳过 {task.skipped_count}｜失败 {task.failed_count}"
            )
        value = int(task.completed_count / task.total_count * 100) if task.total_count else 0
        self.background_progress.setValue(max(0, min(100, value)))
        if hasattr(self, "dialog_background_state_label"):
            self.dialog_background_state_label.setText(self.background_state_label.text())
            self.dialog_background_task_label.setText(self.background_task_label.text())
            self.dialog_background_file_label.setText(self.background_file_label.text())
            self.dialog_background_count_label.setText(self.background_count_label.text())
            self.dialog_background_result_label.setText(self.background_result_label.text())
            self.dialog_background_progress.setValue(self.background_progress.value())
        if task.error_message:
            self.background_recent_log_label.setText(f"最近日志：{task.error_message}")
        self.stage_label.setText(f"阶段：后台任务 - {task.task_name}")
        self.current_file_label.setText(f"当前文件：{task.current_file or '-'}")
        self.count_label.setText(f"已处理：{task.completed_count} / {task.total_count}")
        self.work_progress.setValue(max(0, min(100, value)))

    def _task_elapsed_text(self, task: BackgroundTask) -> str:
        if not task.started_at:
            return "-"
        try:
            start = datetime.strptime(task.started_at, "%Y-%m-%d %H:%M:%S")
            end = datetime.strptime(task.finished_at, "%Y-%m-%d %H:%M:%S") if task.finished_at else datetime.now()
            seconds = max(0, int((end - start).total_seconds()))
            return f"{seconds // 60:02d}:{seconds % 60:02d}"
        except ValueError:
            return "-"

    def _task_status_text(self, status: str) -> str:
        return {
            TASK_STATUS_PENDING: "等待中",
            TASK_STATUS_RUNNING: "运行中",
            TASK_STATUS_PAUSED: "已暂停",
            TASK_STATUS_STOPPING: "正在停止",
            TASK_STATUS_STOPPED: "已停止",
            TASK_STATUS_COMPLETED: "已完成",
            TASK_STATUS_ERROR: "出错",
            TASK_STATUS_CANCELLED: "已取消",
        }.get(status, status or "空闲")

    def _refresh_background_buttons(self, status: str | None = None) -> None:
        running = self.background_task_manager.is_running() or self.export_worker is not None
        paused = status == TASK_STATUS_PAUSED or self.background_task_manager.is_paused()
        stopping = status == TASK_STATUS_STOPPING
        self.background_run_button.setEnabled(not running)
        self.background_pause_button.setEnabled(running and not paused and not stopping)
        self.background_resume_button.setEnabled(running and paused)
        self.background_stop_button.setEnabled(running)
        self.background_clear_button.setEnabled(not running)
        if hasattr(self, "compact_background_run_button"):
            self.compact_background_run_button.setEnabled(not running)
            self.compact_background_pause_button.setEnabled(running and not paused and not stopping)
            self.compact_background_resume_button.setEnabled(running and paused)
            self.compact_background_stop_button.setEnabled(running)
        if not running and status in {None, TASK_STATUS_COMPLETED, TASK_STATUS_STOPPED, TASK_STATUS_ERROR, TASK_STATUS_CANCELLED}:
            self.background_state_label.setText(f"当前状态：{self._task_status_text(status) if status else '空闲'}")
            if status is None and hasattr(self, "background_compact_label"):
                self.background_compact_label.setText(
                    f"后台：{self._task_status_text(status) if status else '空闲'}｜0/0｜成功 0｜失败 0"
                )

    def change_idle_task_options(self) -> None:
        self.change_background_task_options()

    def pause_idle_tasks(self) -> None:
        self.pause_background_tasks()

    def cancel_idle_tasks(self) -> None:
        self.stop_background_tasks()

    def run_idle_task_tick(self) -> None:
        if self._has_running_task() or not self.items:
            return
        mode = self.config.background_task_start_mode
        if mode == "manual":
            return
        if mode == "ask_when_idle":
            self.stage_label.setText("阶段：后台任务待确认")
            self.current_file_label.setText("后台任务可运行，点击“立即运行”开始。")
            if not self.background_ask_prompted:
                self.background_ask_prompted = True
                reply = QMessageBox.question(
                    self,
                    "后台任务",
                    "当前软件空闲，可以运行已勾选的后台任务。是否现在开始？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    self.start_background_tasks()
            return
        if mode == "auto_low_priority":
            self.start_background_tasks()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._has_running_task():
            box = QMessageBox(self)
            box.setWindowTitle("当前有任务正在运行")
            box.setText("当前有任务正在运行，是否停止任务并退出？")
            wait_button = box.addButton("等待完成", QMessageBox.RejectRole)
            stop_button = box.addButton("停止任务并退出", QMessageBox.AcceptRole)
            cancel_button = box.addButton("取消", QMessageBox.DestructiveRole)
            box.exec()
            clicked = box.clickedButton()
            if clicked in {wait_button, cancel_button}:
                event.ignore()
                return
            if clicked == stop_button:
                self._cancel_running_tasks()
                self.logger.info("用户选择停止任务并退出")

        if self.config.ask_save_on_exit and self.dirty:
            decision = self._ask_save_before_exit()
            if decision == "cancel":
                event.ignore()
                return
            if decision == "save" and not self._save_with_retry_for_exit():
                event.ignore()
                return
        else:
            save_project_state(self._current_project_state(clean_shutdown=True))

        self._save_layout_state()
        self.logger.info("程序正常退出")
        event.accept()

    def _has_running_task(self) -> bool:
        if self.background_task_manager.is_running():
            return True
        return any(
            worker is not None
            for worker in (
                self.import_worker,
                self.ai_worker,
                self.export_worker,
                self.model_download_worker,
                self.diagnostics_worker,
            )
        )

    def _cancel_running_tasks(self) -> None:
        self.background_task_manager.stop()
        for worker in (self.import_worker, self.ai_worker, self.export_worker, self.model_download_worker, self.diagnostics_worker):
            if worker:
                worker.cancel()

    def _ask_save_before_exit(self) -> str:
        box = QMessageBox(self)
        box.setWindowTitle("保存当前筛片结果？")
        box.setText("当前项目有未保存的筛片结果。是否保存后退出？")
        save_button = box.addButton("保存并退出", QMessageBox.AcceptRole)
        dont_save_button = box.addButton("不保存退出", QMessageBox.DestructiveRole)
        cancel_button = box.addButton("取消", QMessageBox.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked == save_button:
            return "save"
        if clicked == dont_save_button:
            save_project_state(self._current_project_state(clean_shutdown=True))
            return "dont_save"
        if clicked == cancel_button:
            return "cancel"
        return "cancel"

    def _save_with_retry_for_exit(self) -> bool:
        while True:
            if self.save_all(auto=False, clean_shutdown=True):
                return True
            box = QMessageBox(self)
            box.setWindowTitle("保存失败")
            box.setText("保存失败，请查看 logs/app.log。")
            retry_button = box.addButton("重试保存", QMessageBox.AcceptRole)
            dont_save_button = box.addButton("不保存退出", QMessageBox.DestructiveRole)
            cancel_button = box.addButton("取消", QMessageBox.RejectRole)
            box.exec()
            clicked = box.clickedButton()
            if clicked == retry_button:
                continue
            if clicked == dont_save_button:
                save_project_state(self._current_project_state(clean_shutdown=True))
                return True
            if clicked == cancel_button:
                return False
            return False

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_edit.append(f"[{timestamp}] {message}")
