from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.analyzers.similarity_analyzer import set_group_status
from app.core.mvp_models import PhotoItem


STATUS_LABELS = {
    "best": "组内最佳",
    "backup": "组内备选",
    "duplicate": "重复淘汰",
    "rejected": "组内淘汰",
    "review": "待复核",
    "": "未确认",
}


class ZoomableImageView(QScrollArea):
    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignCenter)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: #15181e;")
        self.setWidget(self.image_label)
        self.original_pixmap = QPixmap()
        self.zoom_factor = 1.0
        self._drag_start: QPoint | None = None
        self._h_start = 0
        self._v_start = 0
        self.viewport().setCursor(Qt.OpenHandCursor)

    def load_image(self, path: Path) -> None:
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        image = reader.read()
        self.original_pixmap = QPixmap.fromImage(image) if not image.isNull() else QPixmap()
        if self.original_pixmap.isNull():
            self.image_label.setText("无法预览该图片")
            self.image_label.resize(420, 320)
            return
        self.image_label.setText("")
        self.actual_size()

    def fit_to_view(self) -> None:
        if self.original_pixmap.isNull():
            return
        viewport_size = self.viewport().size()
        if viewport_size.width() <= 0 or viewport_size.height() <= 0:
            return
        width_ratio = viewport_size.width() / max(1, self.original_pixmap.width())
        height_ratio = viewport_size.height() / max(1, self.original_pixmap.height())
        self.set_zoom(max(0.05, min(width_ratio, height_ratio)), emit_signal=True)

    def actual_size(self) -> None:
        self.set_zoom(1.0, emit_signal=True)

    def set_zoom(self, zoom: float, *, emit_signal: bool = True) -> None:
        if self.original_pixmap.isNull():
            return
        zoom = max(0.05, min(6.0, float(zoom)))
        self.zoom_factor = zoom
        scaled_size = self.original_pixmap.size()
        scaled_size.setWidth(max(1, int(scaled_size.width() * zoom)))
        scaled_size.setHeight(max(1, int(scaled_size.height() * zoom)))
        pixmap = self.original_pixmap.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.size())
        if emit_signal:
            self.zoom_changed.emit(self.zoom_factor)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self.original_pixmap.isNull():
            return super().wheelEvent(event)
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.set_zoom(self.zoom_factor * factor, emit_signal=True)
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag_start = event.position().toPoint()
            self._h_start = self.horizontalScrollBar().value()
            self._v_start = self.verticalScrollBar().value()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_start is not None:
            delta = event.position().toPoint() - self._drag_start
            self.horizontalScrollBar().setValue(self._h_start - delta.x())
            self.verticalScrollBar().setValue(self._v_start - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._drag_start is not None:
            self._drag_start = None
            self.viewport().setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class DualCompareDialog(QDialog):
    group_status_changed = Signal(object, int)

    def __init__(self, items: list[PhotoItem], left_index: int, right_index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.items = items
        self.left_index = left_index
        self.right_index = right_index
        self._syncing_zoom = False
        self.setWindowTitle("双图对比")
        self.resize(1320, 860)
        self._build_ui()
        self._load_images()
        self._refresh_infos()

    @property
    def left_item(self) -> PhotoItem:
        return self.items[self.left_index]

    @property
    def right_item(self) -> PhotoItem:
        return self.items[self.right_index]

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        control_row = QHBoxLayout()
        self.fit_button = QPushButton("适应窗口")
        self.actual_button = QPushButton("100%")
        self.sync_zoom_checkbox = QCheckBox("同步缩放")
        self.sync_zoom_checkbox.setChecked(True)
        self.fit_button.clicked.connect(self.fit_to_window)
        self.actual_button.clicked.connect(self.actual_size)
        control_row.addWidget(self.fit_button)
        control_row.addWidget(self.actual_button)
        control_row.addWidget(self.sync_zoom_checkbox)
        control_row.addStretch(1)
        root.addLayout(control_row)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_side("left"))
        splitter.addWidget(self._build_side("right"))
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([650, 650])
        root.addWidget(splitter, 1)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        close_row.addWidget(close_button)
        root.addLayout(close_row)

        self.left_view.zoom_changed.connect(lambda zoom: self._sync_zoom(self.left_view, self.right_view, zoom))
        self.right_view.zoom_changed.connect(lambda zoom: self._sync_zoom(self.right_view, self.left_view, zoom))

    def _build_side(self, side: str) -> QWidget:
        panel = QFrame()
        panel.setObjectName("DualCompareSide")
        panel.setStyleSheet(
            "QFrame#DualCompareSide { background: #20242d; border: 1px solid #3a4250; border-radius: 8px; }"
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        title = QLabel("左图" if side == "left" else "右图")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #f8fafc;")
        info = QLabel("-")
        info.setWordWrap(True)
        info.setStyleSheet("color: #dbe4f0;")
        view = ZoomableImageView()
        view.setMinimumHeight(480)
        button_row = QHBoxLayout()
        best_button = QPushButton("设为组内最佳")
        backup_button = QPushButton("设为备选")
        duplicate_button = QPushButton("标记重复淘汰")
        best_button.clicked.connect(lambda _checked=False, s=side: self._apply_side_status(s, "best"))
        backup_button.clicked.connect(lambda _checked=False, s=side: self._apply_side_status(s, "backup"))
        duplicate_button.clicked.connect(lambda _checked=False, s=side: self._apply_side_status(s, "duplicate"))
        for button in (best_button, backup_button, duplicate_button):
            button.setMinimumHeight(34)
            button_row.addWidget(button)
        layout.addWidget(title)
        layout.addWidget(info)
        layout.addWidget(view, 1)
        layout.addLayout(button_row)
        if side == "left":
            self.left_info = info
            self.left_view = view
        else:
            self.right_info = info
            self.right_view = view
        return panel

    def _load_images(self) -> None:
        self.left_view.load_image(self.left_item.path)
        self.right_view.load_image(self.right_item.path)

    def fit_to_window(self) -> None:
        self.left_view.fit_to_view()
        if self.sync_zoom_checkbox.isChecked():
            self.right_view.set_zoom(self.left_view.zoom_factor, emit_signal=False)
        else:
            self.right_view.fit_to_view()

    def actual_size(self) -> None:
        self.left_view.actual_size()
        if self.sync_zoom_checkbox.isChecked():
            self.right_view.set_zoom(1.0, emit_signal=False)
        else:
            self.right_view.actual_size()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.fit_to_window()

    def _sync_zoom(self, source: ZoomableImageView, target: ZoomableImageView, zoom: float) -> None:
        if self._syncing_zoom or not self.sync_zoom_checkbox.isChecked():
            return
        self._syncing_zoom = True
        try:
            target.set_zoom(zoom, emit_signal=False)
        finally:
            self._syncing_zoom = False

    def _apply_side_status(self, side: str, status: str) -> None:
        item = self.left_item if side == "left" else self.right_item
        self._apply_status(item, status)

    def _apply_status(self, item: PhotoItem, status: str) -> None:
        changed = set_group_status(self.items, item, status)
        source_index = self.items.index(item)
        self._refresh_infos()
        self.group_status_changed.emit(changed, source_index)

    def _refresh_infos(self) -> None:
        self.left_info.setText(self._item_info_text(self.left_item))
        self.right_info.setText(self._item_info_text(self.right_item))

    def _item_info_text(self, item: PhotoItem) -> str:
        issue_tags = "、".join(item.issue_tags or []) or "-"
        return (
            f"{item.filename}\n"
            f"评级：{item.quality_rating or '-'} | 商业：{item.commercial_score or '-'} | 作品集：{item.portfolio_score or '-'}\n"
            f"问题：{issue_tags}\n"
            f"相似度：{item.similarity_score:.2f} | 组内状态：{STATUS_LABELS.get(item.similar_group_status, item.similar_group_status or '未确认')}\n"
            f"人工最佳：{'是' if item.best_in_group else '否'} | AI推荐：{'是' if item.ai_recommended_best else '否'}"
        )
