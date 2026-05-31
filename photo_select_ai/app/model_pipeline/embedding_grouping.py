from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from app.core.mvp_models import PhotoItem
from app.model_pipeline.embedding_extractor import normalize_vector
from app.model_pipeline.embedding_types import AutoGroupAssignment, EmbeddingGroup, EmbeddingGroupingResult, EmbeddingResult


GROUPING_CONNECTED_COMPONENTS = "connected_components"
GROUPING_COMPLETE_LINKAGE = "complete_linkage"
GROUPING_AVERAGE_LINKAGE = "average_linkage"
GROUPING_SEQUENCE_CONSTRAINED = "sequence_constrained"
GROUPING_STRATEGIES = {
    GROUPING_CONNECTED_COMPONENTS,
    GROUPING_COMPLETE_LINKAGE,
    GROUPING_AVERAGE_LINKAGE,
    GROUPING_SEQUENCE_CONSTRAINED,
}


def group_embeddings(
    items: Sequence[PhotoItem],
    embeddings: Sequence[EmbeddingResult],
    threshold: float = 0.86,
    method: str = "embedding",
    grouping_strategy: str = GROUPING_COMPLETE_LINKAGE,
    group_min_similarity_threshold: float = 0.92,
    max_group_size: int = 25,
) -> EmbeddingGroupingResult:
    """Group visual embeddings without writing any project state.

    ``connected_components`` is kept as the legacy union-find behavior. The
    default ``complete_linkage`` avoids chaining: every member in a group must
    be pairwise similar enough before a new item can join the group.
    """
    grouping_strategy = grouping_strategy if grouping_strategy in GROUPING_STRATEGIES else GROUPING_COMPLETE_LINKAGE
    threshold = float(threshold)
    group_min_similarity_threshold = float(group_min_similarity_threshold)
    max_group_size = max(2, int(max_group_size or 25))

    path_to_embedding = {Path(result.image_path): result for result in embeddings}
    indexed: list[tuple[int, PhotoItem, EmbeddingResult, np.ndarray]] = []
    for index, item in enumerate(items):
        result = path_to_embedding.get(Path(item.path))
        if result and result.vector:
            indexed.append((index, item, result, normalize_vector(result.vector)))

    similarity_matrix, nearest_similarity = _build_similarity_matrix(indexed)
    if grouping_strategy == GROUPING_CONNECTED_COMPONENTS:
        clusters = _connected_components(indexed, similarity_matrix, threshold)
    elif grouping_strategy == GROUPING_SEQUENCE_CONSTRAINED:
        legacy_clusters = _connected_components(indexed, similarity_matrix, threshold)
        clusters = _split_high_risk_clusters(
            clusters=legacy_clusters,
            indexed=indexed,
            similarity_matrix=similarity_matrix,
            threshold=threshold,
            group_min_similarity_threshold=group_min_similarity_threshold,
            max_group_size=max_group_size,
        )
    elif grouping_strategy == GROUPING_AVERAGE_LINKAGE:
        clusters = _greedy_linkage_clusters(
            indexed=indexed,
            similarity_matrix=similarity_matrix,
            threshold=threshold,
            group_min_similarity_threshold=group_min_similarity_threshold,
            strategy=GROUPING_AVERAGE_LINKAGE,
        )
    else:
        clusters = _greedy_linkage_clusters(
            indexed=indexed,
            similarity_matrix=similarity_matrix,
            threshold=threshold,
            group_min_similarity_threshold=group_min_similarity_threshold,
            strategy=GROUPING_COMPLETE_LINKAGE,
        )

    assignments: dict[Path, AutoGroupAssignment] = {}
    groups: list[EmbeddingGroup] = []
    group_number = 1
    for indexes in sorted((sorted(values) for values in clusters), key=lambda values: values[0] if values else 0):
        if len(indexes) < 2:
            continue
        group_id = f"auto_{group_number:03d}"
        group_number += 1
        ranked = _rank_group(indexes, items, nearest_similarity)
        stats = _group_similarity_stats(indexes, similarity_matrix)
        high_risk = _is_high_risk_overmerge(
            size=len(indexes),
            min_similarity=stats["min"],
            max_group_size=max_group_size,
            group_min_similarity_threshold=group_min_similarity_threshold,
        )
        group = EmbeddingGroup(
            group_id=group_id,
            indexes=indexes,
            representative_index=indexes[0],
            average_similarity=round(stats["avg"], 4),
            method=method,
            min_similarity=round(stats["min"], 4),
            max_similarity=round(stats["max"], 4),
            high_risk_overmerge=high_risk,
            grouping_strategy=grouping_strategy,
        )
        groups.append(group)
        for rank, index in enumerate(ranked, start=1):
            item = items[index]
            confidence = _confidence_for(method, max(nearest_similarity[index], stats["avg"]))
            assignments[Path(item.path)] = AutoGroupAssignment(
                image_path=item.path,
                auto_group_id=group_id,
                auto_group_confidence=confidence,
                auto_group_reason=_reason_for(method, grouping_strategy, group_id, len(indexes), rank, confidence, high_risk),
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
                auto_group_reason="Below auto grouping threshold; not assigned to an auto group.",
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


def _build_similarity_matrix(
    indexed: list[tuple[int, PhotoItem, EmbeddingResult, np.ndarray]],
) -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    similarity_matrix: dict[tuple[int, int], float] = {}
    nearest_similarity: dict[int, float] = defaultdict(float)
    for left_pos, (left_index, left_item, _left_result, left_vector) in enumerate(indexed):
        similarity_matrix[(left_index, left_index)] = 1.0
        for right_index, right_item, _right_result, right_vector in indexed[left_pos + 1 :]:
            similarity = float(np.dot(left_vector, right_vector))
            adjusted = _adjust_similarity(similarity, left_item, right_item)
            similarity_matrix[(left_index, right_index)] = adjusted
            similarity_matrix[(right_index, left_index)] = adjusted
            nearest_similarity[left_index] = max(nearest_similarity[left_index], adjusted)
            nearest_similarity[right_index] = max(nearest_similarity[right_index], adjusted)
    return similarity_matrix, nearest_similarity


def _connected_components(
    indexed: list[tuple[int, PhotoItem, EmbeddingResult, np.ndarray]],
    similarity_matrix: dict[tuple[int, int], float],
    threshold: float,
) -> list[list[int]]:
    parent = {index: index for index, *_rest in indexed}

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

    indexes = [index for index, *_rest in indexed]
    for left_pos, left_index in enumerate(indexes):
        for right_index in indexes[left_pos + 1 :]:
            if similarity_matrix.get((left_index, right_index), -1.0) >= threshold:
                union(left_index, right_index)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in indexes:
        clusters[find(index)].append(index)
    return [sorted(values) for values in clusters.values()]


def _greedy_linkage_clusters(
    indexed: list[tuple[int, PhotoItem, EmbeddingResult, np.ndarray]],
    similarity_matrix: dict[tuple[int, int], float],
    threshold: float,
    group_min_similarity_threshold: float,
    strategy: str,
) -> list[list[int]]:
    clusters: list[list[int]] = []
    pair_threshold = max(threshold, group_min_similarity_threshold if strategy == GROUPING_COMPLETE_LINKAGE else threshold)
    for index, *_rest in indexed:
        best_cluster_index: int | None = None
        best_score = -1.0
        for cluster_index, cluster in enumerate(clusters):
            similarities = [similarity_matrix.get((index, member), -1.0) for member in cluster]
            if not similarities:
                continue
            min_similarity = min(similarities)
            avg_similarity = sum(similarities) / len(similarities)
            if strategy == GROUPING_AVERAGE_LINKAGE:
                can_join = avg_similarity >= threshold and min_similarity >= min(threshold, group_min_similarity_threshold)
                score = avg_similarity
            else:
                can_join = min_similarity >= pair_threshold
                score = min_similarity
            if can_join and score > best_score:
                best_cluster_index = cluster_index
                best_score = score
        if best_cluster_index is None:
            clusters.append([index])
        else:
            clusters[best_cluster_index].append(index)
    return [sorted(values) for values in clusters]


def _split_high_risk_clusters(
    clusters: list[list[int]],
    indexed: list[tuple[int, PhotoItem, EmbeddingResult, np.ndarray]],
    similarity_matrix: dict[tuple[int, int], float],
    threshold: float,
    group_min_similarity_threshold: float,
    max_group_size: int,
) -> list[list[int]]:
    split_clusters: list[list[int]] = []
    index_to_tuple = {index: value for value in indexed for index in [value[0]]}
    for cluster in clusters:
        stats = _group_similarity_stats(cluster, similarity_matrix)
        if not _is_high_risk_overmerge(len(cluster), stats["min"], max_group_size, group_min_similarity_threshold):
            split_clusters.append(cluster)
            continue
        subset = [index_to_tuple[index] for index in cluster if index in index_to_tuple]
        split_clusters.extend(
            _greedy_linkage_clusters(
                indexed=subset,
                similarity_matrix=similarity_matrix,
                threshold=max(threshold, group_min_similarity_threshold),
                group_min_similarity_threshold=group_min_similarity_threshold,
                strategy=GROUPING_COMPLETE_LINKAGE,
            )
        )
    return split_clusters


def _group_similarity_stats(indexes: Sequence[int], similarity_matrix: dict[tuple[int, int], float]) -> dict[str, float]:
    values: list[float] = []
    for left_pos, left_index in enumerate(indexes):
        for right_index in indexes[left_pos + 1 :]:
            values.append(similarity_matrix.get((left_index, right_index), 0.0))
    if not values:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    return {"avg": sum(values) / len(values), "min": min(values), "max": max(values)}


def _is_high_risk_overmerge(
    size: int,
    min_similarity: float,
    max_group_size: int,
    group_min_similarity_threshold: float,
) -> bool:
    return size > max_group_size or (size > 2 and min_similarity < group_min_similarity_threshold)


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
        elif sequence_delta > 200:
            adjusted -= 0.06
        elif sequence_delta > 80:
            adjusted -= 0.03
        elif sequence_delta > 30:
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


def _confidence_for(method: str, similarity: float) -> float:
    scale = 0.72 if method == "ahash_fallback" else 0.96
    return round(max(0.0, min(scale, similarity * scale)), 4)


def _reason_for(
    method: str,
    grouping_strategy: str,
    group_id: str,
    size: int,
    rank: int,
    confidence: float,
    high_risk: bool,
) -> str:
    backend = "fallback aHash/color" if method == "ahash_fallback" else "visual embedding"
    risk = " High-risk over-merge; please review manually." if high_risk else ""
    return (
        f"Auto grouped into {group_id} by {backend} with {grouping_strategy}; "
        f"group size {size}, rank {rank}/{size}, confidence {confidence:.2f}.{risk}"
    )
