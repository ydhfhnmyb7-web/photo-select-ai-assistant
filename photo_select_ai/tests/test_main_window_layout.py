from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QFrame, QGroupBox, QLabel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.categories import MANUAL_CROP, MANUAL_RETOUCH
from app.core.mvp_models import PhotoItem
from app.core.review_models import REVIEW_STATUS_HUMAN_CONFIRMED, REVIEW_STATUS_UNREVIEWED
from app.ui.main_window import MainWindow
from app.ui.main_window import PhotoListModel


def _app() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def _make_photo(name: str = "a.jpg") -> PhotoItem:
    return PhotoItem(
        path=Path(name),
        filename=name,
        thumbnail_path=Path(name),
        width=1200,
        height=800,
        file_size_mb=1.0,
    )


def _label_texts(widget) -> list[str]:
    return [label.text() for label in widget.findChildren(QLabel)]


def test_main_window_modular_layout() -> None:
    settings = QSettings("PhotoSelectAI", "PhotoSelectAIAssistant")
    settings.clear()
    settings.sync()
    app = _app()
    window = MainWindow()
    try:
        top_status = window.findChild(QFrame, "TopStatusBar")
        assert top_status is not None
        assert top_status.maximumHeight() <= 50 or top_status.height() <= 50
        assert hasattr(window, "body_splitter")
        assert hasattr(window, "horizontal_splitter")
        assert window.main_stack.count() == 4
        assert window.current_main_section == "review"
        assert window.main_stack.currentWidget().objectName() == "ReviewCorrectionPage"
        assert window.review_stack.currentWidget().objectName() == "ReviewWorkspacePage"
        assert list(window.nav_buttons) == ["project", "ai", "review", "export"]
        assert window.nav_buttons["project"].text() == "1. 导入与设置"
        assert window.nav_buttons["ai"].text() == "2. AI分析结果"
        assert window.nav_buttons["review"].text() == "3. 审核与修正"
        assert window.nav_buttons["export"].text() == "4. 导出与整理"
        assert window.top_settings_button.text() == "设置"

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


def test_project_home_empty_state_and_dashboard_visibility() -> None:
    settings = QSettings("PhotoSelectAI", "PhotoSelectAIAssistant")
    settings.clear()
    settings.sync()
    app = _app()
    window = MainWindow()
    try:
        window._activate_workflow_section("project", persist=False)
        window.update_project_dashboard()

        assert not window.project_landing_container.isHidden()
        assert window.hero_new_project_button.text() == "新建选片项目"
        assert window.hero_new_project_button.objectName() == "PrimaryButton"
        assert window.hero_open_project_button.text() == "打开已有项目"
        assert window.hero_continue_project_button.text() == "继续上次项目"
        assert "PhotoSelect AI Assistant" in _label_texts(window.project_landing_container)
        assert "导入照片" in _label_texts(window.project_landing_container)
        assert "AI 分析" in _label_texts(window.project_landing_container)
        assert "人工审核与导出" in _label_texts(window.project_landing_container)
        assert window.project_dashboard_card.isHidden()
        assert window.project_setup_group.isHidden()
        assert window.project_import_card.isHidden()
        assert window.advanced_settings_group.isHidden()
        assert not window.advanced_settings_widget.isVisible()
        assert all(widget.isHidden() for widget in window.workflow_header_widgets["project"])

        window.selected_folder = Path("demo_project")
        window.items = [_make_photo("001.jpg"), _make_photo("002.jpg")]
        window.model.set_items(window.items)
        window.update_project_dashboard()

        assert window.project_landing_container.isHidden()
        assert not window.project_dashboard_card.isHidden()
        assert not window.project_setup_group.isHidden()
        assert not window.project_import_card.isHidden()
        assert window.advanced_settings_group.isHidden()
        assert all(not widget.isHidden() for widget in window.workflow_header_widgets["project"])
        assert window.dashboard_total_value.text() == "2"
        assert window.next_step_button.text() == "开始 AI 分析"

        window.items[0].ai_suggestion = '{"version": "business_classifier_v2"}'
        window.update_project_dashboard()
        assert window.next_step_button.text() == "进入审核与修正"

        window.items[0].review_status = REVIEW_STATUS_HUMAN_CONFIRMED
        window.update_project_dashboard()
        assert window.next_step_button.text() == "导出与整理"
    finally:
        window.close()
        app.processEvents()
        settings.clear()
        settings.sync()


def test_empty_states_hide_inactive_controls() -> None:
    settings = QSettings("PhotoSelectAI", "PhotoSelectAIAssistant")
    settings.clear()
    settings.sync()
    app = _app()
    window = MainWindow()
    try:
        window.items = []
        window.model.set_items([])
        window.update_project_dashboard()

        assert not window.review_empty_state_wrapper.isHidden()
        assert window.review_empty_state_card.maximumWidth() <= 700
        assert window.review_stack.isHidden()
        assert window.review_mode_header_widget.isHidden()
        assert window.review_context_label.isHidden()
        assert window.review_empty_import_button.text() == "导入照片"
        assert window.review_empty_import_button.objectName() == "PrimaryButton"
        assert window.review_empty_project_button.text() == "返回项目中心"

        assert not window.ai_empty_state_wrapper.isHidden()
        assert window.ai_empty_state_card.maximumWidth() <= 700
        assert window.ai_action_group.isHidden()
        assert window.ai_overview_group.isHidden()
        assert "尚未运行 AI 分析" in _label_texts(window.ai_empty_state_card)
        assert window.ai_empty_start_button.text() == "开始 AI 分析"
        assert window.ai_empty_start_button.objectName() == "PrimaryButton"
        assert window.ai_empty_group_button.text() == "生成 AI 相似候选组"

        assert not window.export_empty_state_wrapper.isHidden()
        assert window.export_empty_state_card.maximumWidth() <= 700
        assert window.export_controls_widget.isHidden()
        assert window.export_safety_group.isHidden()
        assert window.export_empty_review_button.text() == "进入审核与修正"
        assert window.export_empty_review_button.objectName() == "PrimaryButton"

        window.items = [_make_photo("001.jpg")]
        window.items[0].review_status = REVIEW_STATUS_UNREVIEWED
        window.model.set_items(window.items)
        window.update_project_dashboard()
        assert window.review_empty_state_wrapper.isHidden()
        assert not window.review_stack.isHidden()
        assert not window.review_mode_header_widget.isHidden()
        assert not window.review_context_label.isHidden()
        assert not window.export_empty_state_wrapper.isHidden()
        assert window.export_controls_widget.isHidden()

        window.items[0].review_status = REVIEW_STATUS_HUMAN_CONFIRMED
        window.items[0].delivery_use = ["客户可选"]
        window.update_project_dashboard()
        assert window.export_empty_state_wrapper.isHidden()
        assert not window.export_controls_widget.isHidden()
        assert not window.export_safety_group.isHidden()
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
        labels = _label_texts(window)
        assert "人工状态筛选" in labels
        assert "AI建议筛选" in labels
        assert "视图与缩略图" in labels
        assert window.delivery_checkboxes
        assert "AI相似候选组" in window.ai_group_overview_label.text()
        assert "人工确认" in window.preview_info_label.text() or window.preview_info_label.text() == "-"
        assert not window.move_source_checkbox.isChecked()
        assert not window.move_source_checkbox.isEnabled()
        assert window.project_wizard_button.text() == "新建项目向导"
        assert "复制到新文件夹" in window.export_copy_safety_label.text()
        assert window.filter_combo.count() > 0
        assert window.sort_combo.count() > 0
        assert window.view_combo.count() > 0
    finally:
        window.close()
        app.processEvents()
        settings.clear()
        settings.sync()


def test_filter_terms_do_not_mix_ai_suggestion_and_human_workflow() -> None:
    model = PhotoListModel()
    manual_deliverable = _make_photo("a.jpg")
    manual_deliverable.manual_category = MANUAL_RETOUCH
    ai_candidate = _make_photo("b.jpg")
    ai_candidate.auto_group_id = "auto_001"
    human_duplicate = _make_photo("c.jpg")
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
    test_project_home_empty_state_and_dashboard_visibility()
    test_empty_states_hide_inactive_controls()
    test_workflow_labels_separate_ai_and_human_states()
    test_filter_terms_do_not_mix_ai_suggestion_and_human_workflow()
    print("main window layout tests passed")
