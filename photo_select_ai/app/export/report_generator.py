from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.mvp_models import PhotoItem
from app.core.review_models import (
    REVIEW_STATUS_AI_REVIEWED,
    REVIEW_STATUS_HUMAN_CONFIRMED,
    REVIEW_STATUS_NEEDS_REVIEW,
    REVIEW_STATUS_UNREVIEWED,
)
from app.analyzers.similarity_analyzer import build_similarity_summary
from app.export.export_utils import readable_json, readable_list


CSV_HEADERS = [
    "file_name",
    "file_path",
    "exported_paths",
    "photo_type",
    "subtype",
    "quality_rating",
    "delivery_use",
    "issue_tags",
    "commercial_score",
    "portfolio_score",
    "ai_suggestion",
    "human_decision",
    "review_status",
    "similar_group_id",
    "similar_group_rank",
    "similar_group_status",
    "similarity_score",
    "best_in_group",
    "similar_group_note",
    "ai_recommended_best",
    "ai_similarity_reason",
    "human_group_decision",
    "review_note",
    "export_status",
    "export_error",
]


QUALITY_LABELS = {
    "S": "S 强烈推荐",
    "A": "A 可交付",
    "B": "B 备选",
    "C": "C 留档/练习",
    "X": "X 废片",
    "未评级": "未评级",
}


def safe_photo_type(item: PhotoItem) -> str:
    return item.photo_type or "未分类"


def safe_subtype(item: PhotoItem) -> str:
    return item.subtype or "未细分"


def safe_quality(item: PhotoItem) -> str:
    return item.quality_rating or "未评级"


def build_export_summary(photos: list[PhotoItem]) -> dict[str, Any]:
    by_photo_type: Counter[str] = Counter()
    by_subtype: Counter[str] = Counter()
    by_quality_rating: Counter[str] = Counter()
    by_delivery_use: Counter[str] = Counter()
    by_issue_tags: Counter[str] = Counter()
    review_status: Counter[str] = Counter()
    similar_groups: set[str] = set()
    best_in_group_count = 0
    duplicate_count = 0
    backup_count = 0
    review_count = 0

    for item in photos:
        by_photo_type[safe_photo_type(item)] += 1
        by_subtype[safe_subtype(item)] += 1
        quality = safe_quality(item)
        by_quality_rating[quality] += 1
        uses = item.delivery_use or []
        if not uses:
            by_delivery_use["未设置"] += 1
        for use in uses:
            by_delivery_use[use] += 1
        for tag in item.issue_tags or []:
            by_issue_tags[tag] += 1
        review_status[item.review_status or REVIEW_STATUS_UNREVIEWED] += 1
        if item.similar_group_id:
            similar_groups.add(item.similar_group_id)
        if item.best_in_group:
            best_in_group_count += 1
        if item.similar_group_status == "duplicate":
            duplicate_count += 1
        elif item.similar_group_status == "backup":
            backup_count += 1
        elif item.similar_group_status == "review":
            review_count += 1

    total = len(photos)
    similarity_summary = build_similarity_summary(photos)
    return {
        "total": total,
        "by_photo_type": dict(by_photo_type),
        "by_subtype": dict(by_subtype),
        "by_quality_rating": dict(by_quality_rating),
        "by_delivery_use": dict(by_delivery_use),
        "by_issue_tags": dict(by_issue_tags),
        "review_status": dict(review_status),
        "reject_count": by_quality_rating.get("X", 0),
        "portfolio_candidate_count": by_delivery_use.get("作品集候选", 0),
        "retouch_candidate_count": by_delivery_use.get("精修候选", 0),
        "client_select_count": by_delivery_use.get("客户可选", 0),
        "similar_group_count": len(similar_groups),
        "best_in_group_count": best_in_group_count,
        "similarity_summary": similarity_summary,
        "photos_in_groups_count": similarity_summary["photos_in_groups_count"],
        "duplicate_count": duplicate_count,
        "backup_count": backup_count,
        "similar_review_count": review_count,
        "manual_confirmed_count": review_status.get(REVIEW_STATUS_HUMAN_CONFIRMED, 0),
        "needs_review_count": review_status.get(REVIEW_STATUS_NEEDS_REVIEW, 0),
        "unreviewed_count": review_status.get(REVIEW_STATUS_UNREVIEWED, 0),
        "ai_reviewed_count": review_status.get(REVIEW_STATUS_AI_REVIEWED, 0),
    }


def write_csv_report(report_path: Path, photos: list[PhotoItem], export_records: dict[str, dict]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for item in photos:
            record = export_records.get(str(item.path), {})
            writer.writerow(csv_row(item, record))


def csv_row(item: PhotoItem, record: dict) -> dict[str, Any]:
    return {
        "file_name": item.filename,
        "file_path": str(item.path),
        "exported_paths": ";".join(record.get("exported_paths", [])),
        "photo_type": safe_photo_type(item),
        "subtype": safe_subtype(item),
        "quality_rating": safe_quality(item),
        "delivery_use": readable_list(item.delivery_use),
        "issue_tags": readable_list(item.issue_tags),
        "commercial_score": item.commercial_score,
        "portfolio_score": item.portfolio_score,
        "ai_suggestion": readable_json(item.ai_suggestion),
        "human_decision": readable_json(item.human_decision),
        "review_status": item.review_status or REVIEW_STATUS_UNREVIEWED,
        "similar_group_id": item.similar_group_id,
        "similar_group_rank": item.similar_group_rank or "",
        "similar_group_status": item.similar_group_status,
        "similarity_score": f"{item.similarity_score:.3f}" if item.similarity_score else "",
        "best_in_group": "是" if item.best_in_group else "否",
        "similar_group_note": item.similar_group_note,
        "ai_recommended_best": "是" if item.ai_recommended_best else "否",
        "ai_similarity_reason": item.ai_similarity_reason,
        "human_group_decision": readable_json(item.human_group_decision),
        "review_note": item.review_note,
        "export_status": record.get("export_status", "report_only"),
        "export_error": record.get("export_error", ""),
    }


def write_markdown_report(report_path: Path, photos: list[PhotoItem], export_records: dict[str, dict]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_export_summary(photos)
    exported_count = sum(1 for record in export_records.values() if record.get("export_status") == "exported")
    failed_count = sum(1 for record in export_records.values() if record.get("export_status") == "failed")
    skipped_count = sum(1 for record in export_records.values() if record.get("export_status") in {"skipped_no_export", "report_only"})
    total = summary["total"] or 1
    reject_rate = summary["reject_count"] / total
    sa_count = summary["by_quality_rating"].get("S", 0) + summary["by_quality_rating"].get("A", 0)
    sa_rate = sa_count / total
    portfolio_rate = summary["portfolio_candidate_count"] / total

    lines = [
        "# PhotoSelect 筛片复盘报告",
        "",
        "## 一、项目概览",
        "",
        f"- 导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 原始照片总数：{summary['total']}",
        f"- 已人工确认：{summary['manual_confirmed_count']}",
        f"- 待复核：{summary['needs_review_count']}",
        f"- 未审：{summary['unreviewed_count']}",
        f"- AI已审：{summary['ai_reviewed_count']}",
        f"- 实际导出照片数：{exported_count}",
        f"- 跳过导出照片数：{skipped_count}",
        f"- 导出失败照片数：{failed_count}",
        "",
        "## 二、照片类型统计",
        "",
        _table(summary["by_photo_type"], "照片类型"),
        "",
        "## 三、品质等级统计",
        "",
        _quality_table(summary["by_quality_rating"]),
        "",
        "## 四、交付用途统计",
        "",
        _table(summary["by_delivery_use"], "交付用途"),
        "",
        "## 五、问题标签统计",
        "",
        _table(summary["by_issue_tags"], "问题标签") if summary["by_issue_tags"] else "暂无问题标签记录。",
        "",
        "## 六、关键指标",
        "",
        f"- 废片率：{reject_rate:.1%}",
        f"- S/A 级占比：{sa_rate:.1%}",
        f"- 作品集候选率：{portfolio_rate:.1%}",
        f"- 精修候选数量：{summary['retouch_candidate_count']}",
        f"- 客户可选数量：{summary['client_select_count']}",
        f"- 待复核数量：{summary['needs_review_count']}",
        f"- 重复组数量：{summary['similar_group_count']}",
        f"- 组内最佳数量：{summary['best_in_group_count']}",
        "",
        "## 七、相似组统计",
        "",
        f"- 相似组数量：{summary['similar_group_count']}",
        f"- 相似组内照片数量：{summary['photos_in_groups_count']}",
        f"- 组内最佳数量：{summary['best_in_group_count']}",
        f"- 重复淘汰数量：{summary['duplicate_count']}",
        f"- 待复核组内照片数量：{summary['similar_review_count']}",
        "",
        "## 八、相似组明细",
        "",
        _similarity_group_table(summary["similarity_summary"]),
        "",
        "## 九、AI 建议被人工修改的记录",
        "",
        _ai_override_table(photos),
        "",
        "## 十、待复核照片",
        "",
        _file_table([item for item in photos if (item.review_status or "") == REVIEW_STATUS_NEEDS_REVIEW], "暂无待复核照片。"),
        "",
        "## 十一、不导出照片清单",
        "",
        _file_table(
            [
                item
                for item in photos
                if "不导出" in (item.delivery_use or [])
                or item.quality_rating == "X"
                or item.similar_group_status == "duplicate"
            ],
            "暂无不导出照片。",
        ),
        "",
        "## 十二、本次筛片总结",
        "",
        _summary_text(summary),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _table(counter: dict[str, int], title: str) -> str:
    lines = [f"| {title} | 数量 |", "|---|---:|"]
    for key, count in sorted(counter.items(), key=lambda pair: (-pair[1], pair[0])):
        lines.append(f"| {key} | {count} |")
    return "\n".join(lines)


def _quality_table(counter: dict[str, int]) -> str:
    lines = ["| 品质等级 | 数量 |", "|---|---:|"]
    for code in ["S", "A", "B", "C", "X", "未评级"]:
        lines.append(f"| {QUALITY_LABELS.get(code, code)} | {counter.get(code, 0)} |")
    return "\n".join(lines)


def _similarity_group_table(summary: dict[str, Any]) -> str:
    groups = summary.get("groups") or []
    if not groups:
        return "暂无相似组记录。"
    lines = ["| 组ID | 数量 | 组内最佳 | 备选数量 | 淘汰数量 | 待复核数量 |", "|---|---:|---|---:|---:|---:|"]
    for group in groups:
        lines.append(
            "| {group_id} | {count} | {best_file} | {backup_count} | {duplicate_count} | {review_count} |".format(
                group_id=group.get("group_id", "-"),
                count=group.get("count", 0),
                best_file=group.get("best_file") or "-",
                backup_count=group.get("backup_count", 0),
                duplicate_count=group.get("duplicate_count", 0),
                review_count=group.get("review_count", 0),
            )
        )
    return "\n".join(lines)


def _ai_override_table(photos: list[PhotoItem]) -> str:
    rows: list[str] = []
    for item in photos:
        ai_text = " / ".join(part for part in [item.ai_primary_category, item.ai_secondary_category] if part)
        human_text = " / ".join(part for part in [item.photo_type, item.subtype, item.quality_rating] if part)
        if ai_text and human_text and ai_text not in human_text:
            rows.append(f"| {item.filename} | {ai_text} | {human_text} | {item.review_note or '-'} |")
    if not rows:
        return "暂无明显人工覆盖记录。"
    return "\n".join(["| 文件名 | AI建议 | 人工结果 | 备注 |", "|---|---|---|---|", *rows])


def _file_table(photos: list[PhotoItem], empty_text: str) -> str:
    if not photos:
        return empty_text
    lines = ["| 文件名 | 类型 | 品质 | 备注 |", "|---|---|---|---|"]
    for item in photos:
        type_text = " / ".join(part for part in [safe_photo_type(item), safe_subtype(item)] if part)
        lines.append(f"| {item.filename} | {type_text} | {safe_quality(item)} | {item.review_note or '-'} |")
    return "\n".join(lines)


def _summary_text(summary: dict[str, Any]) -> str:
    total = summary["total"]
    sa_count = summary["by_quality_rating"].get("S", 0) + summary["by_quality_rating"].get("A", 0)
    issue_counter = Counter(summary["by_issue_tags"])
    issues = "、".join(key for key, _count in issue_counter.most_common(3)) or "暂无集中问题标签"
    return (
        f"本次共处理 {total} 张照片，其中 A 级以上照片 {sa_count} 张，废片 {summary['reject_count']} 张，"
        f"作品集候选 {summary['portfolio_candidate_count']} 张。主要问题集中在 {issues}。"
        "建议后续拍摄时结合本次问题标签，重点关注表情引导、背景控制和构图稳定性。"
    )
