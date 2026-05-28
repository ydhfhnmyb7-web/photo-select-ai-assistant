from __future__ import annotations

import csv
import shutil
from pathlib import Path

from app.core.models import PhotoAnalysis


REPORT_FILENAME = "PhotoSelect_Report.csv"


def export_results(source_folder: Path, analyses: list[PhotoAnalysis]) -> tuple[Path, Path]:
    """Copy photos into category folders and write a CSV report."""
    source_folder = Path(source_folder)
    output_dir = source_folder.parent / "PhotoSelect_Output"
    output_dir.mkdir(parents=True, exist_ok=True)

    for analysis in analyses:
        category_dir = output_dir / analysis.effective_category
        category_dir.mkdir(parents=True, exist_ok=True)
        target_path = _unique_target_path(category_dir / analysis.path.name)
        shutil.copy2(analysis.path, target_path)

    report_path = output_dir / REPORT_FILENAME
    _write_report(report_path, analyses)
    return output_dir, report_path


def _unique_target_path(target_path: Path) -> Path:
    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    parent = target_path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter:03d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _write_report(report_path: Path, analyses: list[PhotoAnalysis]) -> None:
    headers = [
        "文件名",
        "路径",
        "清晰度",
        "曝光",
        "对比度",
        "综合评分",
        "自动分类",
        "手动分类",
        "备注",
        "相似组",
        "相似组推荐",
        "分辨率",
        "文件大小MB",
        "拍摄时间",
    ]
    with report_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for analysis in analyses:
            writer.writerow(
                {
                    "文件名": analysis.filename,
                    "路径": str(analysis.path),
                    "清晰度": analysis.clarity_score,
                    "曝光": analysis.exposure_score,
                    "对比度": analysis.contrast_score,
                    "综合评分": analysis.total_score,
                    "自动分类": analysis.auto_category,
                    "手动分类": analysis.manual_category,
                    "备注": analysis.notes,
                    "相似组": analysis.similar_group_id or "",
                    "相似组推荐": "是" if analysis.recommended_in_group else "否",
                    "分辨率": analysis.resolution_text,
                    "文件大小MB": analysis.file_size_mb,
                    "拍摄时间": analysis.taken_at,
                }
            )
