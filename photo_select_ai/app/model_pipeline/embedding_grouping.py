from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from app.core.mvp_models import PhotoItem
from app.model_pipeline.embedding_extractor import normalize_vector
from app.model_pipeline.embedding_types import AutoGroupAssignment, EmbeddingGroup, EmbeddingGroupingResult, EmbeddingResult


def group_embeddings(
    items: Sequence[PhotoItem],
    embeddings: Sequence[EmbeddingResult],
    threshold: float = 0.86,
    method: str = "embedding",
) -> EmbeddingGroupingResult:
    path_to_embedding = {Path(result.image_path): result for result in embeddings}
    indexed: list[tuple[int, PhotoItem, EmbeddingResult, np.ndarray]] = []
    for index, item in enumerate(items):
        result = path_to_embedding.get(Path(item.path))
        if result and result.vector:
            indexed.append((index, item, result, normalize_vector(result.vector)))

    parent = {index: index for index, *_rest in indexed}
    nearest_similarity: dict[int, float] = defaultdict(float)

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

    for left_pos, (left_index, left_item, _left_result, left_vector) in enumerate(indexed):
        for right_index, right_item, _right_result, right_vector in indexed[left_pos + 1 :]:
            similarity = float(np.dot(left_vector, right_vector))
            adjusted = _adjust_similarity(similarity, left_item, right_item)
            nearest_similarity[left_index] = max(nearest_similarity[left_index], adjusted)
            nearest_similarity[right_index] = max(nearest_similarity[right_index], adjusted)
            if adjusted >= threshold:
                union(left_index, right_index)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index, *_rest in indexed:
        clusters[find(index)].append(index)

    assignments: dict[Path, AutoGroupAssignment] = {}
    groups: list[EmbeddingGroup] = []
    group_number = 1
    for indexes in sorted((sorted(values) for values in clusters.values()), key=lambda values: values[0]):
        if len(indexes) < 2:
            continue
        group_id = f"auto_{group_number:03d}"
        group_number += 1
        representative_index = indexes[0]
        ranked = _rank_group(indexes, items, nearest_similarity)
        average_similarity = _average_pair_similarity(indexes, items, path_to_embedding)
        group = EmbeddingGroup(
            group_id=group_id,
            indexes=indexes,
            representative_index=representative_index,
            average_similarity=average_similarity,
            method=method,
        )
        groups.append(group)
        for rank, index in enumerate(ranked, start=1):
            item = items[index]
            confidence = _confidence_for(method, max(nearest_similarity[index], average_similarity))
            assignments[Path(item.path)] = AutoGroupAssignment(
                image_path=item.path,
                auto_group_id=group_id,
                auto_group_confidence=confidence,
                auto_group_reason=_reason_for(method, group_id, len(indexes), rank, confidence),
                auto_group_rank=rank,
                auto_group_size=len(indexes),
                grouping_method=method,
                nearest_similarity=round(nearest_similarity[index], 4),
            )

    for index, item in enumerate(items):
        path = Path(item.path)
        if path not in assignments:
            assignments[path] = AutoGroupAssignment(
                image_path=path,
                auto_group_id="",
                auto_group_confidence=0.0,
                auto_group_reason="未达到自动分组阈值，未进入相似组。",
                auto_group_rank=0,
                auto_group_size=0,
                grouping_method=method,
                nearest_similarity=round(nearest_similarity.get(index, 0.0), 4),
            )

    return EmbeddingGroupingResult(assignments=assignments, groups=groups, threshold=threshold, method=method)


def apply_auto_group_assignments(items: Sequence[PhotoItem], result: EmbeddingGroupingResult) -> None:
    for item in items:
        assignment = result.assignments.get(Path(item.path))
        if not assignment:
            continue
        item.auto_group_id = assignment.auto_group_id
        item.auto_group_confidence = assignment.auto_group_confidence
        item.auto_group_reason = assignment.auto_group_reason
        item.auto_group_rank = assignment.auto_group_rank
        item.auto_group_size = assignment.auto_group_size
        item.grouping_method = assignment.grouping_method


def _adjust_similarity(base_similarity: float, left: PhotoItem, right: PhotoItem) -> float:
    adjusted = base_similarity
    if _orientation(left) != _orientation(right):
        adjusted -= 0.04
    sequence_delta = _filename_number_delta(left.filename, right.filename)
    if sequence_delta is not None:
        if sequence_delta <= 2:
            adjusted += 0.025
        elif sequence_delta <= 8:
            adjusted += 0.01
        elif sequence_delta > 80:
            adjusted -= 0.015
    if left.taken_at and right.taken_at:
        seconds = _time_delta_seconds(left.taken_at, right.taken_at)
        if seconds is not None:
            if seconds <= 8:
                adjusted += 0.02
            elif seconds > 900:
                adjusted -= 0.03
    if left.people_count and right.people_count and left.people_count != right.people_count:
        adjusted -= min(0.08, abs(left.people_count - right.people_count) * 0.03)
    return max(-1.0, min(1.0, adjusted))


def _orientation(item: PhotoItem) -> str:
    return "portrait" if item.height >= item.width else "landscape"


def _filename_number_delta(left: str, right: str) -> int | None:
    left_digits = "".join(char for char in Path(left).stem if char.isdigit())
    right_digits = "".join(char for char in Path(right).stem if char.isdigit())
    if not left_digits or not right_digits:
        return None
    try:
        return abs(int(left_digits[-8:]) - int(right_digits[-8:]))
    except ValueError:
        return None


def _time_delta_seconds(left: str, right: str) -> float | None:
    try:
        left_dt = datetime.fromisoformat(left.replace("Z", "+00:00"))
        right_dt = datetime.fromisoformat(right.replace("Z", "+00:00"))
        return abs((left_dt - right_dt).total_seconds())
    except Exception:
        return None


def _rank_group(indexes: list[int], items: Sequence[PhotoItem], nearest_similarity: dict[int, float]) -> list[int]:
    return sorted(
        indexes,
        key=lambda index: (
            float(getattr(items[index], "commercial_score", 0.0) or 0.0),
            float(getattr(items[index], "absolute_quality_score", 0.0) or 0.0),
            nearest_similarity.get(index, 0.0),
        ),
        reverse=True,
    )


def _average_pair_similarity(
    indexes: list[int],
    items: Sequence[PhotoItem],
    embeddings: dict[Path, EmbeddingResult],
) -> float:
    values: list[float] = []
    for pos, left_index in enumerate(indexes):
        left = embeddings.get(Path(items[left_index].path))
        if not left:
            continue
        left_vector = normalize_vector(left.vector)
        for right_index in indexes[pos + 1 :]:
            right = embeddings.get(Path(items[right_index].path))
            if not right:
                continue
            values.append(float(np.dot(left_vector, normalize_vector(right.vector))))
    return round(sum(values) / len(values), 4) if values else 0.0


def _confidence_for(method: str, similarity: float) -> float:
    scale = 0.72 if method == "ahash_fallback" else 0.96
    return round(max(0.0, min(scale, similarity * scale)), 4)


def _reason_for(method: str, group_id: str, size: int, rank: int, confidence: float) -> str:
    if method == "ahash_fallback":
        return (
            f"使用 aHash/颜色直方图 fallback 自动分到 {group_id}，本组 {size} 张，"
            f"组内排序 {rank}/{size}，置信度较低，仅供人工确认。"
        )
    return (
        f"基于视觉 embedding 的余弦相似度自动分到 {group_id}，本组 {size} 张，"
        f"组内排序 {rank}/{size}，自动分组置信度 {confidence:.2f}。"
    )
