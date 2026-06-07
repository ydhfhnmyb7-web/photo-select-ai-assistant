from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QGroupBox

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.categories import MANUAL_CROP, MANUAL_RETOUCH
from app.core.mvp_models import PhotoItem
from app.ui.main_window import MainWindow
from app.ui.main_window import PhotoListModel


def _app() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def test_main_window_modular_layout() -> None:
    settings = QSettings("PhotoSelectAI", "PhotoSelectAIAssistant")
    settings.clear()
    settings.sync()
    app = _app()
    window = MainWindow()
    try:
        top_status = window.findChild(type(window.project_status_label.parent()), "TopStatusBar")
        assert top_status is not None
        assert top_status.maximumHeight() <= 50 or top_status.height() <= 50
        assert hasattr(window, "body_splitter")
        assert hasattr(window, "horizontal_splitter")
        assert window.main_stack.count() == 4
        assert window.current_main_section == "review"
        assert window.main_stack.currentWidget().objectName() == "ReviewCorrectionPage"
        assert window.review_stack.currentWidget().objectName() == "ReviewWorkspacePage"
        assert list(window.nav_buttons) == ["project", "ai", "review", "export"]

        expected_pages = {
            "project": "ProjectImportPage",
            "ai": "AIAnalysisPage",
            "review": "ReviewCorrectionPage",
            "export": "ExportReviewPage",
        }
        for section, object_name in expected_pages.items():
            window._activate_workflow_section(section, persist=False)
            assert window.current_main_section == section
            assert window.main_stack.currentWidget().objectName() == object_name

        window._activate_workflow_section("similar", persist=False)
        assert window.current_main_section == "review"
        assert window.review_stack.currentWidget().objectName() == "SimilarGroupPage"

        window._activate_workflow_section("settings", persist=False)
        assert window.current_main_section == "project"
        assert window.advanced_settings_group.isChecked()

        window._activate_workflow_section("export", persist=True)
        settings.sync()
        window.close()
        app.processEvents()

        restored = MainWindow()
        try:
            assert restored.current_main_section == "export"
            assert restored.main_stack.currentWidget().objectName() == "ExportReviewPage"
        finally:
            restored.close()
            app.processEvents()
    finally:
        window.close()
        app.processEvents()
        settings.clear()
        settings.sync()


def test_workflow_labels_separate_ai_and_human_states() -> None:
    settings = QSettings("PhotoSelectAI", "PhotoSelectAIAssistant")
    settings.clear()
    settings.sync()
    app = _app()
    window = MainWindow()
    try:
        assert "AI 建议" in window.review_ai_suggestion_label.text()
        group_titles = [group.title() for group in window.findChildren(QGroupBox)]
        assert any(title.startswith("人工决定") for title in group_titles)
        assert window.delivery_checkboxes
        assert "AI相似候选组" in window.ai_group_overview_label.text()
        assert "人工确认" in window.preview_info_label.text() or window.preview_info_label.text() == "-"
        assert not window.move_source_checkbox.isChecked()
        assert not window.move_source_checkbox.isEnabled()
    finally:
        window.close()
        app.processEvents()
        settings.clear()
        settings.sync()


def test_filter_terms_do_not_mix_ai_suggestion_and_human_workflow() -> None:
    model = PhotoListModel()
    manual_deliverable = PhotoItem(path=Path("a.jpg"), filename="a.jpg", thumbnail_path=Path("a.jpg"), width=1, height=1, file_size_mb=1)
    manual_deliverable.manual_category = MANUAL_RETOUCH
    ai_candidate = PhotoItem(path=Path("b.jpg"), filename="b.jpg", thumbnail_path=Path("b.jpg"), width=1, height=1, file_size_mb=1)
    ai_candidate.auto_group_id = "auto_001"
    human_duplicate = PhotoItem(path=Path("c.jpg"), filename="c.jpg", thumbnail_path=Path("c.jpg"), width=1, height=1, file_size_mb=1)
    human_duplicate.manual_category = MANUAL_CROP
    model.set_items([manual_deliverable, ai_candidate, human_duplicate])

    model.filter_name = "人工：可交付"
    model.refresh()
    assert model.rowCount() == 1
    assert model.source_index_at_row(0) == 0

    model.filter_name = "AI：相似候选组"
    model.refresh()
    assert model.rowCount() == 1
    assert model.source_index_at_row(0) == 1

    model.filter_name = "人工：重复"
    model.refresh()
    assert model.rowCount() == 1
    assert model.source_index_at_row(0) == 2


if __name__ == "__main__":
    test_main_window_modular_layout()
    test_workflow_labels_separate_ai_and_human_states()
    test_filter_terms_do_not_mix_ai_suggestion_and_human_workflow()
    print("main window layout tests passed")
