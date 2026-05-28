from __future__ import annotations

import json
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps

from app.analyzers.basic_quality_analyzer import analyze_basic_quality
from app.core.config import PROJECT_ROOT
from app.core.mvp_models import PhotoItem
from app.core.review_models import REVIEW_STATUS_HUMAN_CONFIRMED, REVIEW_STATUS_NEEDS_REVIEW


SIMILARITY_LOG_PATH = PROJECT_ROOT / "logs" / "similarity.log"
THRESHOLDS = {"strict": 5, "standard": 8, "loose": 12}
HUMAN_GROUP_STATUSES = {"best", "backup", "duplicate", "rejected", "review"}

ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]
PauseCallback = Callable[[], None]


@dataclass
class SimilarityGroup:
    group_id: str
    indexes: list[int]
    representative_index: int
    recommended_index: int
    recommended_reason: str = ""


@dataclass
class SimilarityResult:
    groups: list[SimilarityGroup] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    threshold: int = 8


def calculate_similarity_groups(
    items: list[PhotoItem],
    mode: str = "standard",
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
    pause_callback: PauseCallback | None = None,
) -> SimilarityResult:
    started = time.perf_counter()
    threshold = THRESHOLDS.get(mode, THRESHOLDS["standard"])
    _write_similarity_log(f"相似分析开始：照片 {len(items)} 张，模式 {mode}，阈值 {threshold}")
    hashes: dict[int, int] = {}
    skipped: list[str] = []

    total = len(items)
    for index, item in enumerate(items):
        if cancel_callback and cancel_callback():
            break
        if pause_callback:
            pause_callback()
        if progress_callback:
            progress_callback(index + 1, max(1, total * 2), item.filename)
        try:
            file_mtime = item.path.stat().st_mtime
            if item.similarity_hash and abs(float(item.similarity_hash_mtime or 0.0) - file_mtime) < 0.0001:
                hashes[index] = int(item.similarity_hash, 16)
            else:
                value = average_hash(item.path)
                item.similarity_hash = f"{value:016x}"
                item.similarity_hash_mtime = file_mtime
                hashes[index] = value
        except Exception as exc:
            skipped.append(f"{item.filename}: {exc}")
            _write_similarity_log(f"hash失败：{item.path} {exc}\n{traceback.format_exc()}")

    groups = _group_hashes(items, hashes, threshold, progress_callback, cancel_callback, pause_callback)
    result_groups: list[SimilarityGroup] = []

    for group_number, indexes in enumerate(groups, start=1):
        group_id = f"组{group_number:03d}"
        representative = indexes[0]
        recommended = _recommend_best_index(items, indexes)
        reason = _recommend_reason(items[recommended], indexes, items)
        result_groups.append(
            SimilarityGroup(
                group_id=group_id,
                indexes=indexes,
                representative_index=representative,
                recommended_index=recommended,
                recommended_reason=reason,
            )
        )
        _apply_group(items, group_id, indexes, representative, recommended, hashes, reason)
        _write_similarity_log(
            f"{group_id}: {len(indexes)} 张，AI推荐 {items[recommended].filename}，原因：{reason}"
        )

    grouped_indexes = {idx for group in result_groups for idx in group.indexes}
    for index, item in enumerate(items):
        if index not in grouped_indexes:
            _clear_ai_similarity_group(item)

    elapsed = (time.perf_counter() - started) * 1000
    _write_similarity_log(f"相似分析结束：组 {len(result_groups)}，跳过 {len(skipped)}，耗时 {elapsed:.1f}ms")
    return SimilarityResult(groups=result_groups, skipped=skipped, elapsed_ms=elapsed, threshold=threshold)


def average_hash(path: Path | str, hash_size: int = 8) -> int:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = list(image.getdata())
    average = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | (1 if pixel >= average else 0)
    return value


def hamming_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def build_similarity_summary(items: list[PhotoItem]) -> dict:
    groups: dict[str, list[PhotoItem]] = defaultdict(list)
    for item in items:
        if item.similar_group_id:
            groups[item.similar_group_id].append(item)
    rows = []
    best_count = 0
    duplicate_count = 0
    backup_count = 0
    review_count = 0
    for group_id, group_items in sorted(groups.items()):
        best = next((item for item in group_items if item.best_in_group or item.ai_recommended_best), None)
        group_duplicate = sum(1 for item in group_items if item.similar_group_status == "duplicate")
        group_backup = sum(1 for item in group_items if item.similar_group_status == "backup")
        group_review = sum(1 for item in group_items if item.similar_group_status == "review")
        best_count += sum(1 for item in group_items if item.best_in_group)
        duplicate_count += group_duplicate
        backup_count += group_backup
        review_count += group_review
        rows.append(
            {
                "group_id": group_id,
                "count": len(group_items),
                "best_file": best.filename if best else "",
                "backup_count": group_backup,
                "duplicate_count": group_duplicate,
                "review_count": group_review,
            }
        )
    return {
        "similar_group_count": len(groups),
        "photos_in_groups_count": sum(len(group) for group in groups.values()),
        "best_in_group_count": best_count,
        "duplicate_count": duplicate_count,
        "backup_count": backup_count,
        "review_count": review_count,
        "groups": rows,
    }


def _group_hashes(
    items: list[PhotoItem],
    hashes: dict[int, int],
    threshold: int,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
    pause_callback: PauseCallback | None,
) -> list[list[int]]:
    indexes = sorted(hashes)
    parent = {index: index for index in indexes}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    compare_total = max(1, len(indexes) * (len(indexes) - 1) // 2)
    compared = 0
    bucketed: dict[str, list[int]] = defaultdict(list)
    for index in indexes:
        prefix = f"{hashes[index]:016x}"[:2]
        bucketed[prefix].append(index)

    # Also compare neighboring prefixes loosely enough for the standard/loose modes.
    candidate_pairs: set[tuple[int, int]] = set()
    for bucket in bucketed.values():
        for pos, left in enumerate(bucket):
            for right in bucket[pos + 1 :]:
                candidate_pairs.add((left, right))
    if threshold >= 8:
        for pos, left in enumerate(indexes):
            for right in indexes[pos + 1 :]:
                if abs((items[left].width * items[left].height) - (items[right].width * items[right].height)) < max(
                    items[left].width * items[left].height, 1
                ) * 0.35:
                    candidate_pairs.add((left, right))

    for left, right in sorted(candidate_pairs):
        if cancel_callback and cancel_callback():
            break
        if pause_callback:
            pause_callback()
        compared += 1
        if progress_callback and compared % 20 == 0:
            progress_callback(len(items) + min(compared, compare_total), max(1, len(items) + compare_total), items[left].filename)
        if hamming_distance(hashes[left], hashes[right]) <= threshold:
            union(left, right)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in indexes:
        clusters[find(index)].append(index)
    return [sorted(group) for group in clusters.values() if len(group) >= 2]


def _recommend_best_index(items: list[PhotoItem], indexes: list[int]) -> int:
    scored = sorted(indexes, key=lambda idx: _group_quality_score(items[idx]), reverse=True)
    return scored[0]


def _group_quality_score(item: PhotoItem) -> float:
    score = 0.0
    if item.absolute_quality_score:
        score += item.absolute_quality_score * 0.35
    if item.commercial_score:
        score += item.commercial_score * 0.35
    if item.portfolio_score:
        score += item.portfolio_score * 0.15
    score += {"S": 18, "A": 14, "B": 8, "C": 2, "X": -12}.get(item.quality_rating, 6)
    tags = set(item.issue_tags or []) | set(item.ai_quality_tags or [])
    for tag in ["虚焦", "闭眼", "表情崩", "曝光问题"]:
        if tag in tags:
            score -= 16
    if not (item.absolute_quality_score or item.commercial_score):
        try:
            basic = analyze_basic_quality(item.path)
            item.absolute_quality_score = basic.score
            score += basic.score * 0.45
        except Exception:
            score += 35
    return score


def _recommend_reason(item: PhotoItem, indexes: list[int], items: list[PhotoItem]) -> str:
    tags = set(item.issue_tags or []) | set(item.ai_quality_tags or [])
    if tags:
        tag_text = "，且主要风险较少" if not tags.intersection({"虚焦", "闭眼", "表情崩"}) else "，但仍需人工复核风险标签"
    else:
        tag_text = "，未发现明显虚焦或闭眼风险"
    return f"该照片在同组 {len(indexes)} 张中综合评分较高，清晰度/交付评分更稳定{tag_text}，推荐作为组内最佳。"


def _apply_group(
    items: list[PhotoItem],
    group_id: str,
    indexes: list[int],
    representative: int,
    recommended: int,
    hashes: dict[int, int],
    reason: str,
) -> None:
    rep_hash = hashes.get(representative)
    ordered = sorted(indexes, key=lambda idx: _group_quality_score(items[idx]), reverse=True)
    rank_by_index = {idx: rank for rank, idx in enumerate(ordered, start=1)}
    for idx in indexes:
        item = items[idx]
        item.similar_group_id = group_id
        item.similar_group_size = len(indexes)
        item.group_size = len(indexes)
        item.similar_group_rank = rank_by_index.get(idx, 0)
        item.group_rank = item.similar_group_rank
        item.ai_recommended_best = idx == recommended
        item.ai_similarity_reason = reason if idx == recommended else f"与同组照片高度相似，建议对比 {items[recommended].filename} 后决定。"
        if rep_hash is not None and idx in hashes:
            item.similarity_score = round(max(0.0, 1.0 - hamming_distance(rep_hash, hashes[idx]) / 64.0), 4)
        if not _has_human_group_decision(item) and not item.similar_group_status:
            item.similar_group_status = "review"


def _clear_ai_similarity_group(item: PhotoItem) -> None:
    protected = _has_human_group_decision(item)
    item.similar_group_id = ""
    item.similar_group_size = 0
    item.group_size = 0
    item.similar_group_rank = 0
    item.group_rank = 0
    item.similarity_score = 0.0
    item.ai_recommended_best = False
    item.ai_similarity_reason = ""
    if not protected:
        item.similar_group_status = ""


def _has_human_group_decision(item: PhotoItem) -> bool:
    if item.best_in_group:
        return True
    if item.similar_group_note:
        return True
    if item.human_group_decision:
        try:
            decision = json.loads(item.human_group_decision)
            if decision.get("decision_source") == "human":
                return True
            if decision.get("similar_group_status") in HUMAN_GROUP_STATUSES:
                return True
        except Exception:
            return True
    if item.similar_group_status in HUMAN_GROUP_STATUSES and item.review_status in {
        REVIEW_STATUS_HUMAN_CONFIRMED,
        REVIEW_STATUS_NEEDS_REVIEW,
    }:
        return True
    return False


def set_group_status(items: list[PhotoItem], target: PhotoItem, status: str) -> list[PhotoItem]:
    group_id = target.similar_group_id
    if not group_id:
        return []
    group_items = [item for item in items if item.similar_group_id == group_id]
    if status == "best":
        for item in group_items:
            item.best_in_group = item.path == target.path
            item.recommended_keep = item.path == target.path
            item.recommended_in_group = item.path == target.path
            if item.path == target.path:
                item.similar_group_status = "best"
                item.review_status = REVIEW_STATUS_HUMAN_CONFIRMED
            elif item.similar_group_status not in {"duplicate", "rejected"} and item.review_status != REVIEW_STATUS_HUMAN_CONFIRMED:
                item.similar_group_status = item.similar_group_status or "backup"
    elif status == "backup":
        target.best_in_group = False
        target.recommended_keep = False
        target.recommended_in_group = False
        target.similar_group_status = "backup"
        target.review_status = REVIEW_STATUS_HUMAN_CONFIRMED
        if not target.quality_rating:
            target.quality_rating = "B"
    elif status == "duplicate":
        target.best_in_group = False
        target.recommended_keep = False
        target.recommended_in_group = False
        target.similar_group_status = "duplicate"
        target.review_status = REVIEW_STATUS_HUMAN_CONFIRMED
        if "重复照片" not in target.issue_tags:
            target.issue_tags.append("重复照片")
        if "不导出" not in target.delivery_use:
            target.delivery_use.append("不导出")
        if not target.quality_rating:
            target.quality_rating = "C"
    elif status == "review":
        target.best_in_group = False
        target.recommended_keep = False
        target.recommended_in_group = False
        target.similar_group_status = "review"
        target.review_status = REVIEW_STATUS_NEEDS_REVIEW
    target.human_group_decision = json.dumps(
        {
            "similar_group_id": group_id,
            "similar_group_status": target.similar_group_status,
            "best_in_group": target.best_in_group,
            "decision_source": "human",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "note": target.similar_group_note or "",
        },
        ensure_ascii=False,
    )
    return group_items


def _write_similarity_log(message: str) -> None:
    SIMILARITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SIMILARITY_LOG_PATH.open("a", encoding="utf-8").write(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
    )
