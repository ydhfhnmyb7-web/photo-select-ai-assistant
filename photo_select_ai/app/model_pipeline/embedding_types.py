from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EmbeddingResult:
    image_path: Path
    vector: list[float]
    model_name: str
    model_version: str
    method: str
    cache_key: str = ""
    cached: bool = False
    cached_at: str = ""
    dim: int = 0
    source: str = ""

    def __post_init__(self) -> None:
        self.image_path = Path(self.image_path)
        if not self.dim:
            self.dim = len(self.vector)


@dataclass
class AutoGroupAssignment:
    image_path: Path
    auto_group_id: str = ""
    auto_group_confidence: float = 0.0
    auto_group_reason: str = ""
    auto_group_rank: int = 0
    auto_group_size: int = 0
    grouping_method: str = ""
    nearest_similarity: float = 0.0

    def __post_init__(self) -> None:
        self.image_path = Path(self.image_path)


@dataclass
class EmbeddingGroup:
    group_id: str
    indexes: list[int]
    representative_index: int
    average_similarity: float
    method: str
    min_similarity: float = 0.0
    max_similarity: float = 0.0
    high_risk_overmerge: bool = False
    grouping_strategy: str = "connected_components"


@dataclass
class EmbeddingGroupingResult:
    assignments: dict[Path, AutoGroupAssignment] = field(default_factory=dict)
    groups: list[EmbeddingGroup] = field(default_factory=list)
    threshold: float = 0.86
    method: str = "embedding"
