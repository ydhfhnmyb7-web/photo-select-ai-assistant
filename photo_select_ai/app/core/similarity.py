from __future__ import annotations

from collections import defaultdict

from app.core.models import PhotoAnalysis


def hamming_distance(hash_a: str, hash_b: str) -> int:
    if not hash_a or not hash_b:
        return 64
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


def assign_similarity_groups(photos: list[PhotoAnalysis], threshold: int = 6) -> None:
    """Group highly similar photos with a small union-find implementation."""
    count = len(photos)
    if count == 0:
        return

    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(count):
        for right in range(left + 1, count):
            if hamming_distance(photos[left].perceptual_hash, photos[right].perceptual_hash) <= threshold:
                union(left, right)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(count):
        grouped[find(index)].append(index)

    group_id = 1
    for indexes in grouped.values():
        if len(indexes) == 1:
            photo = photos[indexes[0]]
            photo.similar_group_id = None
            photo.recommended_in_group = True
            photo.is_duplicate = False
            continue

        best_index = max(indexes, key=lambda item: _group_quality_score(photos[item]))
        for index in indexes:
            photo = photos[index]
            photo.similar_group_id = group_id
            photo.recommended_in_group = index == best_index
            photo.is_duplicate = index != best_index
        group_id += 1


def _group_quality_score(photo: PhotoAnalysis) -> float:
    return photo.clarity_score * 0.55 + photo.exposure_score * 0.30 + photo.contrast_score * 0.15
