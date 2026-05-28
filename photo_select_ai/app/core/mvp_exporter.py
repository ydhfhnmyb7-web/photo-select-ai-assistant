from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

from app.core.app_logging import get_logger
from app.core.config import AppConfig
from app.core.mvp_models import PhotoItem
from app.export.export_manager import ExportOptions, ExportResult, run_export
from app.export.report_generator import build_export_summary


ProgressCallback = Callable[[int, int, str, str], None]
CancelCallback = Callable[[], bool]


def build_export_plan(items: list[PhotoItem]) -> Counter:
    summary = build_export_summary(items)
    return Counter(summary.get("by_delivery_use", {}))


def export_labeled_photos(
    source_folder: Path,
    items: list[PhotoItem],
    config: AppConfig,
    output_dir: Path | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
    options: ExportOptions | None = None,
) -> tuple[Path, Path, list[dict]]:
    result = run_export(
        source_folder,
        items,
        config,
        output_dir,
        options or ExportOptions(),
        progress_callback,
        cancel_callback,
    )
    get_logger().info("导出完成：%s", result.output_dir)
    rows = list(result.records.values())
    report_path = result.csv_path or result.markdown_path or result.log_path
    return result.output_dir, report_path, rows


def export_photos_with_result(
    source_folder: Path,
    items: list[PhotoItem],
    config: AppConfig,
    output_dir: Path | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
    options: ExportOptions | None = None,
) -> ExportResult:
    return run_export(
        source_folder,
        items,
        config,
        output_dir,
        options or ExportOptions(),
        progress_callback,
        cancel_callback,
    )


def write_current_report(source_folder: Path, items: list[PhotoItem], output_dir: Path | None = None) -> Path:
    result = run_export(
        source_folder,
        items,
        AppConfig(),
        output_dir,
        ExportOptions(mode="只生成报告", organize_by_use=False, organize_by_type=False),
    )
    return result.csv_path or result.markdown_path or result.log_path
