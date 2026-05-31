from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.mvp_models import PhotoItem
from app.model_pipeline.embedding_grouping import GROUPING_COMPLETE_LINKAGE, GROUPING_CONNECTED_COMPONENTS, group_embeddings
from app.model_pipeline.embedding_types import EmbeddingResult


def _item(name: str, width: int = 320, height: int = 240) -> PhotoItem:
    return PhotoItem(
        path=Path(name),
        filename=name,
        thumbnail_path=None,
        width=width,
        height=height,
        file_size_mb=0.01,
    )


def _embedding(name: str, vector: list[float]) -> EmbeddingResult:
    return EmbeddingResult(
        image_path=Path(name),
        vector=vector,
        model_name="mock",
        model_version="v1",
        method="mock_embedding",
    )


def test_cosine_grouping_clusters_similar_vectors_only() -> None:
    items = [_item("same_1.jpg"), _item("same_2.jpg"), _item("different.jpg")]
    embeddings = [
        _embedding("same_1.jpg", [1.0, 0.0, 0.0]),
        _embedding("same_2.jpg", [0.99, 0.04, 0.0]),
        _embedding("different.jpg", [0.0, 1.0, 0.0]),
    ]

    result = group_embeddings(items, embeddings, threshold=0.92, method="mock_embedding")

    assert len(result.groups) == 1
    same_1 = result.assignments[Path("same_1.jpg")]
    same_2 = result.assignments[Path("same_2.jpg")]
    different = result.assignments[Path("different.jpg")]
    assert same_1.auto_group_id
    assert same_1.auto_group_id == same_2.auto_group_id
    assert same_1.auto_group_size == 2
    assert different.auto_group_id == ""
    assert different.auto_group_size == 0


def test_orientation_and_person_count_can_reduce_borderline_similarity() -> None:
    items = [
        _item("IMG_0001.jpg", width=300, height=500),
        _item("IMG_0002.jpg", width=500, height=300),
    ]
    items[0].people_count = 1
    items[1].people_count = 4
    embeddings = [
        _embedding("IMG_0001.jpg", [1.0, 0.0, 0.0]),
        _embedding("IMG_0002.jpg", [0.90, 0.435, 0.0]),
    ]

    result = group_embeddings(items, embeddings, threshold=0.90, method="mock_embedding")

    assert len(result.groups) == 0
    assert result.assignments[Path("IMG_0001.jpg")].auto_group_id == ""


def test_complete_linkage_prevents_chain_overmerge() -> None:
    items = [_item("chain_a.jpg"), _item("chain_b.jpg"), _item("chain_c.jpg")]
    embeddings = [
        _embedding("chain_a.jpg", [0.97, 0.2431, 0.0]),
        _embedding("chain_b.jpg", [1.0, 0.0, 0.0]),
        _embedding("chain_c.jpg", [0.97, 0.0, 0.2431]),
    ]

    legacy = group_embeddings(
        items,
        embeddings,
        threshold=0.96,
        method="mock_embedding",
        grouping_strategy=GROUPING_CONNECTED_COMPONENTS,
        group_min_similarity_threshold=0.90,
    )
    conservative = group_embeddings(
        items,
        embeddings,
        threshold=0.96,
        method="mock_embedding",
        grouping_strategy=GROUPING_COMPLETE_LINKAGE,
        group_min_similarity_threshold=0.90,
    )

    assert len(legacy.groups) == 1
    assert legacy.groups[0].indexes == [0, 1, 2]
    assert len(conservative.groups) == 1
    assert conservative.groups[0].indexes == [0, 1]
    assert conservative.assignments[Path("chain_c.jpg")].auto_group_id == ""


def test_group_min_similarity_is_calculated_for_chain_group() -> None:
    items = [_item("chain_a.jpg"), _item("chain_b.jpg"), _item("chain_c.jpg")]
    embeddings = [
        _embedding("chain_a.jpg", [0.97, 0.2431, 0.0]),
        _embedding("chain_b.jpg", [1.0, 0.0, 0.0]),
        _embedding("chain_c.jpg", [0.97, 0.0, 0.2431]),
    ]

    result = group_embeddings(
        items,
        embeddings,
        threshold=0.96,
        method="mock_embedding",
        grouping_strategy=GROUPING_CONNECTED_COMPONENTS,
        group_min_similarity_threshold=0.90,
    )

    assert len(result.groups) == 1
    assert 0.93 < result.groups[0].min_similarity < 0.95
    assert result.groups[0].max_similarity > 0.96


def test_large_group_is_marked_high_risk_overmerge() -> None:
    items = [_item(f"same_{index}.jpg") for index in range(4)]
    embeddings = [_embedding(item.filename, [1.0, 0.0, 0.0]) for item in items]

    result = group_embeddings(
        items,
        embeddings,
        threshold=0.96,
        method="mock_embedding",
        grouping_strategy=GROUPING_COMPLETE_LINKAGE,
        max_group_size=3,
    )

    assert len(result.groups) == 1
    assert result.groups[0].high_risk_overmerge is True


if __name__ == "__main__":
    test_cosine_grouping_clusters_similar_vectors_only()
    test_orientation_and_person_count_can_reduce_borderline_similarity()
    test_complete_linkage_prevents_chain_overmerge()
    test_group_min_similarity_is_calculated_for_chain_group()
    test_large_group_is_marked_high_risk_overmerge()
    print("embedding grouping tests passed")
