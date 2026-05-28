from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.core.config import AppConfig
from app.core.mvp_models import PhotoItem
from app.export.export_utils import (
    copy_jpg_preview,
    copy_original,
    quality_folder_name,
    safe_name,
    timestamped_output_dir,
)
from app.export.report_generator import write_csv_report, write_markdown_report


ProgressCallback = Callable[[int, int, str, str], None]
CancelCallback = Callable[[], bool]


DELIVERY_DIRS = {
    "客户可选": "01_客户可选",
    "精修候选": "02_精修候选",
    "作品集候选": "03_作品集候选",
    "直接交付": "04_直接交付",
    "小红书/朋友圈展示": "05_小红书朋友圈展示",
    "修图练习": "06_修图练习",
    "内部参考": "07_内部参考",
    "仅留档": "08_仅留档",
    "不导出": "09_不导出清单",
}


@dataclass
class ExportOptions:
    mode: str = "复制原图 + 生成报告"
    organize_by_use: bool = True
    organize_by_type: bool = True
    include_rejects: bool = False
    include_no_export: bool = False
    generate_csv: bool = True
    generate_markdown: bool = True

    @property
    def report_only(self) -> bool:
        return self.mode.strip().startswith("只生成报告")

    @property
    def copy_preview(self) -> bool:
        return "JPG" in self.mode or "预览" in self.mode

    @property
    def should_copy(self) -> bool:
        return not self.report_only


@dataclass
class ExportResult:
    output_dir: Path
    report_dir: Path
    csv_path: Path | None
    markdown_path: Path | None
    log_path: Path
    records: dict[str, dict] = field(default_factory=dict)
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    cancelled: bool = False


def run_export(
    source_folder: Path,
    items: list[PhotoItem],
    config: AppConfig,
    output_dir: Path | None = None,
    options: ExportOptions | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> ExportResult:
    options = options or ExportOptions()
    base_dir = output_dir or source_folder.parent / "PhotoSelect_Output"
    root = timestamped_output_dir(base_dir)
    report_dir = root / "00_报告"
    report_dir.mkdir(parents=True, exist_ok=True)
    (root / DELIVERY_DIRS["不导出"]).mkdir(parents=True, exist_ok=True)
    log_lines = [
        f"导出开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"导出根目录：{root}",
        f"导出模式：{options.mode}",
    ]

    records: dict[str, dict] = {}
    total = len(items)
    for index, item in enumerate(items, start=1):
        if cancel_callback and cancel_callback():
            log_lines.append(f"用户取消导出：已处理 {index - 1}/{total}")
            break
        if progress_callback:
            progress_callback(index, total, item.filename, "")
        record = _export_one(root, item, config, options, log_lines)
        records[str(item.path)] = record

    cancelled = bool(cancel_callback and cancel_callback())
    if cancelled:
        for item in items[len(records) :]:
            records[str(item.path)] = {
                "exported_paths": [],
                "export_status": "skipped_no_export",
                "export_error": "用户取消导出",
            }

    csv_path = report_dir / "review_report.csv" if options.generate_csv else None
    markdown_path = report_dir / "review_summary.md" if options.generate_markdown else None
    if csv_path:
        write_csv_report(csv_path, items, records)
        log_lines.append(f"生成 CSV 报告：{csv_path}")
    if markdown_path:
        write_markdown_report(markdown_path, items, records)
        log_lines.append(f"生成 Markdown 复盘报告：{markdown_path}")

    no_export_list = report_dir / "no_export_list.txt"
    _write_no_export_list(no_export_list, items)
    log_lines.append(f"生成不导出清单：{no_export_list}")

    success_count = sum(1 for record in records.values() if record.get("export_status") == "exported")
    failed_count = sum(1 for record in records.values() if record.get("export_status") == "failed")
    skipped_count = sum(1 for record in records.values() if record.get("export_status") in {"skipped_no_export", "report_only"})
    log_lines.extend(
        [
            f"导出结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"成功：{success_count}",
            f"跳过：{skipped_count}",
            f"失败：{failed_count}",
        ]
    )
    log_path = report_dir / "export_log.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    return ExportResult(
        output_dir=root,
        report_dir=report_dir,
        csv_path=csv_path,
        markdown_path=markdown_path,
        log_path=log_path,
        records=records,
        success_count=success_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        cancelled=cancelled,
    )


def _export_one(root: Path, item: PhotoItem, config: AppConfig, options: ExportOptions, log_lines: list[str]) -> dict:
    exported_paths: list[str] = []
    errors: list[str] = []
    if options.report_only:
        log_lines.append(f"仅报告：{item.path}")
        return {"exported_paths": [], "export_status": "report_only", "export_error": ""}

    destinations = _destinations_for_item(root, item, options)
    if not destinations:
        log_lines.append(f"跳过：{item.path}")
        return {"exported_paths": [], "export_status": "skipped_no_export", "export_error": ""}

    for destination in destinations:
        try:
            copied = (
                copy_jpg_preview(item.path, destination, config.preview_max_size)
                if options.copy_preview
                else copy_original(item.path, destination)
            )
            exported_paths.append(str(copied))
            log_lines.append(f"成功复制：{item.path} -> {copied}")
        except Exception as exc:
            errors.append(f"{destination}: {exc}")
            log_lines.append(f"失败：{item.path} -> {destination}，原因：{exc}")

    if exported_paths:
        status = "exported"
    else:
        status = "failed"
    return {"exported_paths": exported_paths, "export_status": status, "export_error": "；".join(errors)}


def _destinations_for_item(root: Path, item: PhotoItem, options: ExportOptions) -> list[Path]:
    destinations: list[Path] = []
    uses = list(item.delivery_use or [])
    quality = item.quality_rating or ""
    no_export = "不导出" in uses
    is_reject = quality == "X"
    is_duplicate = item.similar_group_status == "duplicate"

    if no_export and not options.include_no_export:
        return []
    if is_reject and not options.include_rejects:
        return []
    if is_duplicate and not options.include_no_export:
        return []

    if options.organize_by_use and not no_export:
        export_uses = [use for use in uses if use != "不导出"]
        if not export_uses and quality in {"S", "A"}:
            export_uses = ["客户可选"]
        for use in export_uses:
            folder = DELIVERY_DIRS.get(use)
            if folder:
                destinations.append(root / folder)

    if options.organize_by_type:
        destinations.append(
            root
            / "10_按类型整理"
            / safe_name(item.photo_type, "未分类")
            / safe_name(item.subtype, "未细分")
            / quality_folder_name(item.quality_rating)
        )

    seen: set[Path] = set()
    unique_destinations: list[Path] = []
    for destination in destinations:
        if destination not in seen:
            unique_destinations.append(destination)
            seen.add(destination)
    return unique_destinations


def _write_no_export_list(path: Path, items: list[PhotoItem]) -> None:
    lines = ["不导出照片清单", ""]
    for item in items:
        if "不导出" in (item.delivery_use or []) or item.quality_rating == "X" or item.similar_group_status == "duplicate":
            lines.append(f"- {item.filename} | {item.path} | {item.review_note or ''}")
    path.write_text("\n".join(lines), encoding="utf-8")
