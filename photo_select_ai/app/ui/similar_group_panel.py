from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.analyzers.similarity_analyzer import build_similarity_summary, set_group_status
from app.core.mvp_models import PhotoItem
from app.ui.dual_compare_dialog import DualCompareDialog


STATUS_LABELS = {
    "best": "人工最佳",
    "backup": "组内备选",
    "duplicate": "重复淘汰",
    "rejected": "组内淘汰",
    "review": "待复核",
    "": "未确认",
}


class SimilarGroupPanel(QWidget):
    request_run_similarity_task = Signal()
    group_status_changed = Signal(object, int)
    request_open_photo = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.items: list[PhotoItem] = []
        self.current_group_id = ""
        self.highlight_source_index = -1
        self._cards: list[QFrame] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.stats_label = QLabel("相似组：0 组｜组内照片：0 张｜人工最佳：0 张｜重复淘汰：0 张｜待复核：0 张")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("font-weight: 700; color: #e5edf7; padding: 4px 2px;")
        stats_row = QHBoxLayout()
        stats_row.addWidget(self.stats_label, 1)
        self.dual_compare_button = QPushButton("双图对比")
        self.dual_compare_button.setMinimumHeight(34)
        self.dual_compare_button.setToolTip("在当前相似组内打开左右大图对比，便于判断组内最佳。")
        self.dual_compare_button.clicked.connect(self._open_dual_compare_current_group)
        stats_row.addWidget(self.dual_compare_button)
        root.addLayout(stats_row)

        self.empty_hint = QLabel("当前还没有相似组。请先运行后台任务：计算相似照片。")
        self.empty_hint.setWordWrap(True)
        self.empty_hint.setStyleSheet("color: #facc15; font-weight: 700; padding: 8px;")
        self.run_similarity_button = QPushButton("立即计算相似照片")
        self.run_similarity_button.setMinimumHeight(36)
        self.run_similarity_button.clicked.connect(self.request_run_similarity_task.emit)
        hint_row = QHBoxLayout()
        hint_row.addWidget(self.empty_hint, 1)
        hint_row.addWidget(self.run_similarity_button)
        root.addLayout(hint_row)

        content = QHBoxLayout()
        content.setSpacing(10)
        root.addLayout(content, 1)

        self.group_list = QListWidget()
        self.group_list.setMinimumWidth(330)
        self.group_list.setSpacing(4)
        self.group_list.currentItemChanged.connect(self._on_group_selected)
        content.addWidget(self.group_list, 0)

        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(10)
        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setWidget(self.grid_container)
        content.addWidget(self.grid_scroll, 1)

    def set_items(self, items: list[PhotoItem], current_index: int = -1, selected_group_id: str | None = None) -> None:
        self.items = items
        self.highlight_source_index = current_index
        if selected_group_id:
            self.current_group_id = selected_group_id
        elif 0 <= current_index < len(items) and items[current_index].similar_group_id:
            self.current_group_id = items[current_index].similar_group_id
        self.refresh()

    def focus_group(self, group_id: str, source_index: int = -1) -> None:
        self.current_group_id = group_id
        self.highlight_source_index = source_index
        self.refresh()

    def refresh(self) -> None:
        summary = build_similarity_summary(self.items)
        self.stats_label.setText(
            "相似组：{similar_group_count} 组｜组内照片：{photos_in_groups_count} 张｜"
            "人工最佳：{best_in_group_count} 张｜重复淘汰：{duplicate_count} 张｜待复核：{review_count} 张".format(**summary)
        )
        has_groups = bool(summary["groups"])
        self.empty_hint.setVisible(not has_groups)
        self.run_similarity_button.setVisible(not has_groups)
        self._refresh_group_list(summary["groups"])
        self.dual_compare_button.setEnabled(len(self._group_items(self.current_group_id)) >= 2)
        self._refresh_group_grid()

    def _refresh_group_list(self, groups: list[dict]) -> None:
        self.group_list.blockSignals(True)
        self.group_list.clear()
        for group in groups:
            group_id = group["group_id"]
            group_items = self._group_items(group_id)
            ai_best = next((item.filename for item in group_items if item.ai_recommended_best), "无")
            human_best = next((item.filename for item in group_items if item.best_in_group), "未设置")
            state = "已确认" if any(item.best_in_group for item in group_items) else "待复核" if group["review_count"] else "未确认"
            text = f"{group_id}｜{group['count']}张｜人工最佳：{human_best}｜{state}"
            detail = (
                f"{group_id}\n照片数量：{group['count']}\nAI推荐：{ai_best}\n人工最佳：{human_best}\n"
                f"重复淘汰：{group['duplicate_count']}\n待复核：{group['review_count']}\n状态：{state}"
            )
            item = QListWidgetItem(text)
            item.setToolTip(detail)
            item.setData(Qt.UserRole, group_id)
            self.group_list.addItem(item)

        if self.group_list.count():
            target_row = 0
            for row in range(self.group_list.count()):
                if self.group_list.item(row).data(Qt.UserRole) == self.current_group_id:
                    target_row = row
                    break
            self.group_list.setCurrentRow(target_row)
            self.current_group_id = self.group_list.item(target_row).data(Qt.UserRole)
        else:
            self.current_group_id = ""
        self.group_list.blockSignals(False)

    def _on_group_selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        self.current_group_id = current.data(Qt.UserRole) or ""
        self._refresh_group_grid()

    def _refresh_group_grid(self) -> None:
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self._cards.clear()

        group_items = self._group_items(self.current_group_id)
        if not group_items:
            label = QLabel("请选择一个相似组。")
            label.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(label, 0, 0)
            return

        for position, photo in enumerate(sorted(group_items, key=lambda p: (p.similar_group_rank or 999, p.filename))):
            card = self._build_photo_card(photo)
            self._cards.append(card)
            self.grid_layout.addWidget(card, position // 2, position % 2)
        self.grid_layout.setRowStretch((len(group_items) + 1) // 2, 1)
        self.grid_layout.setColumnStretch(0, 1)
        self.grid_layout.setColumnStretch(1, 1)

    def _build_photo_card(self, item: PhotoItem) -> QFrame:
        source_index = self.items.index(item)
        card = QFrame()
        card.setObjectName("SimilarPhotoCard")
        card.setFrameShape(QFrame.StyledPanel)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        is_highlight = source_index == self.highlight_source_index
        border = "#22c55e" if item.best_in_group else "#facc15" if item.similar_group_status == "review" else "#4f8cff" if is_highlight else "#3a4250"
        background = (
            "#203126"
            if item.best_in_group
            else "#30272a"
            if item.similar_group_status == "duplicate"
            else "#332f1f"
            if item.similar_group_status == "review"
            else "#252a33"
        )
        card.setStyleSheet(f"QFrame#SimilarPhotoCard {{ background: {background}; border: 2px solid {border}; border-radius: 8px; }}")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumHeight(170)
        image_label.setPixmap(self._thumbnail_pixmap(item))
        layout.addWidget(image_label)

        title = QLabel(item.filename)
        title.setWordWrap(True)
        title.setStyleSheet("font-weight: 700; color: #f8fafc;")
        layout.addWidget(title)

        badges: list[str] = []
        if item.best_in_group:
            badges.append("人工最佳")
        if item.ai_recommended_best:
            badges.append("AI推荐")
        badges.append(STATUS_LABELS.get(item.similar_group_status, item.similar_group_status or "未确认"))
        badge_label = QLabel("｜".join(badges))
        badge_label.setWordWrap(True)
        badge_label.setStyleSheet(self._badge_style(item))
        layout.addWidget(badge_label)

        issue_text = "、".join(item.issue_tags or []) or "-"
        info = QLabel(
            f"评级：{item.quality_rating or '-'}｜商业：{item.commercial_score or '-'}｜作品：{item.portfolio_score or '-'}\n"
            f"相似度：{item.similarity_score:.2f}｜问题：{issue_text}"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #dbe4f0;")
        layout.addWidget(info)

        button_row = QHBoxLayout()
        for label, status in [
            ("最佳", "best"),
            ("备选", "backup"),
            ("淘汰", "duplicate"),
            ("复核", "review"),
        ]:
            button = QPushButton(label)
            button.setMinimumHeight(32)
            button.clicked.connect(lambda _checked=False, photo=item, next_status=status: self._apply_status(photo, next_status))
            button_row.addWidget(button)
        open_button = QPushButton("大图")
        open_button.setMinimumHeight(32)
        open_button.clicked.connect(lambda _checked=False, photo=item: self._open_large_preview(photo))
        button_row.addWidget(open_button)
        compare_button = QPushButton("对比")
        compare_button.setMinimumHeight(32)
        compare_button.setToolTip("与同组另一张照片做左右大图对比。")
        compare_button.clicked.connect(lambda _checked=False, photo=item: self._open_dual_compare(photo))
        button_row.addWidget(compare_button)
        layout.addLayout(button_row)
        return card

    def _thumbnail_pixmap(self, item: PhotoItem) -> QPixmap:
        path = Path(item.thumbnail_path) if item.thumbnail_path else item.path
        pixmap = QPixmap(str(path))
        if pixmap.isNull() and path != item.path:
            pixmap = QPixmap(str(item.path))
        if pixmap.isNull():
            return QPixmap()
        return pixmap.scaled(260, 170, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _badge_style(self, item: PhotoItem) -> str:
        if item.best_in_group:
            return "color: #bbf7d0; font-weight: 700;"
        if item.similar_group_status == "duplicate":
            return "color: #fecaca; font-weight: 700;"
        if item.similar_group_status == "review":
            return "color: #fde68a; font-weight: 700;"
        if item.ai_recommended_best:
            return "color: #93c5fd; font-weight: 700;"
        return "color: #cbd5e1;"

    def _apply_status(self, item: PhotoItem, status: str) -> None:
        changed = set_group_status(self.items, item, status)
        self.highlight_source_index = self.items.index(item)
        self.group_status_changed.emit(changed, self.highlight_source_index)
        self.refresh()

    def _open_dual_compare_current_group(self) -> None:
        group_items = self._sorted_group_items(self.current_group_id)
        if len(group_items) < 2:
            return
        self._open_dual_compare(self._preferred_left_item(group_items))

    def _open_dual_compare(self, item: PhotoItem) -> None:
        partner = self._comparison_partner(item)
        if partner is None:
            return
        dialog = DualCompareDialog(self.items, self.items.index(item), self.items.index(partner), self)
        dialog.group_status_changed.connect(self._on_dual_compare_status_changed)
        dialog.exec()
        self.refresh()

    def _on_dual_compare_status_changed(self, changed: list[PhotoItem], source_index: int) -> None:
        self.highlight_source_index = source_index
        self.group_status_changed.emit(changed, source_index)
        self.refresh()

    def _preferred_left_item(self, group_items: list[PhotoItem]) -> PhotoItem:
        if 0 <= self.highlight_source_index < len(self.items):
            highlighted = self.items[self.highlight_source_index]
            if highlighted in group_items:
                return highlighted
        for candidate in group_items:
            if candidate.best_in_group:
                return candidate
        for candidate in group_items:
            if candidate.ai_recommended_best:
                return candidate
        return group_items[0]

    def _comparison_partner(self, item: PhotoItem) -> PhotoItem | None:
        group_items = self._sorted_group_items(item.similar_group_id)
        if len(group_items) < 2:
            return None
        for candidate in group_items:
            if candidate.path != item.path and (candidate.best_in_group or candidate.ai_recommended_best):
                return candidate
        position = next((index for index, candidate in enumerate(group_items) if candidate.path == item.path), 0)
        if position + 1 < len(group_items):
            return group_items[position + 1]
        return group_items[position - 1] if position > 0 else None

    def _sorted_group_items(self, group_id: str) -> list[PhotoItem]:
        return sorted(self._group_items(group_id), key=lambda p: (p.similar_group_rank or 999, p.filename))

    def _open_large_preview(self, item: PhotoItem) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"相似组大图预览 - {item.filename}")
        dialog.resize(1000, 760)
        layout = QVBoxLayout(dialog)

        reader = QImageReader(str(item.path))
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid():
            size.scale(920, 560, Qt.KeepAspectRatio)
            reader.setScaledSize(size)
        image = reader.read()
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumHeight(560)
        if image.isNull():
            image_label.setText("无法预览该图片。")
        else:
            image_label.setPixmap(QPixmap.fromImage(image))
        layout.addWidget(image_label)

        info = QLabel(
            f"{item.filename}\n"
            f"相似组：{item.similar_group_id or '-'}｜状态：{STATUS_LABELS.get(item.similar_group_status, item.similar_group_status or '未确认')}｜"
            f"AI推荐：{'是' if item.ai_recommended_best else '否'}｜人工最佳：{'是' if item.best_in_group else '否'}\n"
            f"评级：{item.quality_rating or '-'}｜商业：{item.commercial_score or '-'}｜作品：{item.portfolio_score or '-'}｜"
            f"问题：{'、'.join(item.issue_tags or []) or '-'}"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button, alignment=Qt.AlignRight)
        dialog.exec()

    def _group_items(self, group_id: str) -> list[PhotoItem]:
        if not group_id:
            return []
        return [item for item in self.items if item.similar_group_id == group_id]
