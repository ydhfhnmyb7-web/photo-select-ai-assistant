from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ui.main_window import MainWindow


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
        assert window.main_stack.count() == 7
        assert window.current_main_section == "review"
        assert window.main_stack.currentWidget().objectName() == "ReviewWorkspacePage"

        expected_pages = {
            "project": "ProjectImportPage",
            "ai": "AIAnalysisPage",
            "similar": "SimilarGroupPage",
            "export": "ExportReviewPage",
            "settings": "SettingsDebugPage",
            "review": "ReviewWorkspacePage",
        }
        for section, object_name in expected_pages.items():
            window._activate_workflow_section(section, persist=False)
            assert window.current_main_section == section
            assert window.main_stack.currentWidget().objectName() == object_name

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


if __name__ == "__main__":
    test_main_window_modular_layout()
    print("main window layout tests passed")
