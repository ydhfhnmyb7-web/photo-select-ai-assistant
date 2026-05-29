from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.mvp_models import PhotoItem
from app.model_pipeline.embedding_grouping import group_embeddings
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


if __name__ == "__main__":
    test_cosine_grouping_clusters_similar_vectors_only()
    test_orientation_and_person_count_can_reduce_borderline_similarity()
    print("embedding grouping tests passed")
