from __future__ import annotations

from collections.abc import Callable

from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QTextEdit,
    QWidget,
)


ShortcutCallback = Callable[[], None]


class ShortcutManager:
    """Registers review shortcuts and prevents typing fields from stealing intent."""

    def __init__(self, parent: QWidget) -> None:
        self.parent = parent
        self._shortcuts: list[QShortcut] = []

    def register(self, sequence: str, callback: ShortcutCallback, *, allow_in_editor: bool = False) -> None:
        shortcut = QShortcut(QKeySequence(sequence), self.parent)
        shortcut.activated.connect(
            lambda cb=callback, allow=allow_in_editor: self._dispatch(cb, allow)
        )
        self._shortcuts.append(shortcut)

    def _dispatch(self, callback: ShortcutCallback, allow_in_editor: bool) -> None:
        if not allow_in_editor and self.focus_blocks_shortcuts():
            return
        callback()

    @staticmethod
    def focus_blocks_shortcuts() -> bool:
        widget = QApplication.focusWidget()
        if widget is None:
            return False
        return isinstance(
            widget,
            (
                QLineEdit,
                QTextEdit,
                QPlainTextEdit,
                QSpinBox,
                QDoubleSpinBox,
                QComboBox,
            ),
        )


PHOTO_TYPE_SHORTCUTS: dict[int, str] = {
    1: "婚纱照",
    2: "个人写真",
    3: "证件照",
    4: "全家福",
    5: "情侣照",
    6: "会议/活动照",
    7: "商业形象照",
    8: "风景/环境",
    9: "素材/练习",
}


DELIVERY_USE_SHORTCUTS: dict[str, str] = {
    "F": "精修候选",
    "P": "作品集候选",
    "D": "不导出",
    "G": "客户可选",
    "H": "小红书/朋友圈展示",
    "L": "仅留档",
}


ISSUE_TAG_SHORTCUTS: dict[str, str] = {
    "R": "重复照片",
    "V": "虚焦",
    "E": "闭眼",
    "T": "技术可修",
    "Q": "气质不对",
}


SHORTCUT_HELP_TEXT = """基础操作：
← / →：上一张 / 下一张
Enter：保存并下一张
Space：适应窗口 / 100%

品质评级：
S：强烈推荐
A：可交付
B：备选
C：留档/练习
X：废片

照片类型：
1：婚纱照
2：个人写真
3：证件照
4：全家福
5：情侣照
6：会议/活动照
7：商业形象照
8：风景/环境
9：素材/练习

交付用途：
F：精修候选
P：作品集候选
G：客户可选
H：小红书/朋友圈展示
D：不导出
L：仅留档

问题标签：
R：重复照片
V：虚焦
E：闭眼
T：技术可修
Q：气质不对

复核：
M：标记待复核
U：重置人工判断，需要确认"""


SHORTCUT_HINT_TEXT = (
    "快捷键：←→切图 | Enter确认 | S/A/B/C/X评级 | 1-9类型 | "
    "F精修 | P作品集 | G客户可选 | D不导出 | R重复 | M待复核 | Space适应/100%"
)
