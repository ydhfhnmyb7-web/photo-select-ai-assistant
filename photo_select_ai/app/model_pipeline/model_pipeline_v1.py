from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.core.config import AppConfig, load_config
from app.core.mvp_models import PhotoItem
from app.model_pipeline.embedding_cache import EmbeddingCache
from app.model_pipeline.embedding_extractor import EmbeddingExtractor, FallbackEmbeddingExtractor, normalize_vector
from app.model_pipeline.embedding_grouping import apply_auto_group_assignments, group_embeddings
from app.model_pipeline.embedding_types import EmbeddingGroupingResult, EmbeddingResult
from app.model_pipeline.runtime_detector import RuntimeInfo, detect_model_runtime


ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


@dataclass
class ModelPipelineResult:
    grouping: EmbeddingGroupingResult
    embeddings: list[EmbeddingResult] = field(default_factory=list)
    runtime: RuntimeInfo | None = None
    elapsed_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    method: str = ""
    skipped: list[str] = field(default_factory=list)


def run_model_pipeline_v1(
    items: list[PhotoItem],
    config: AppConfig | None = None,
    extractor: EmbeddingExtractor | None = None,
    cache: EmbeddingCache | None = None,
    threshold: float | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> ModelPipelineResult:
    started = time.perf_counter()
    config = config or load_config()
    runtime = detect_model_runtime(config.use_gpu)
    extractor = extractor or FallbackEmbeddingExtractor()
    cache = cache or EmbeddingCache()
    threshold = float(threshold if threshold is not None else config.embedding_similarity_threshold)

    embeddings: list[EmbeddingResult] = []
    to_compute: list[Path] = []
    cache_hits = 0
    cache_misses = 0
    skipped: list[str] = []
    total = len(items)

    for index, item in enumerate(items, start=1):
        if cancel_callback and cancel_callback():
            break
        if progress_callback:
            progress_callback(index, max(1, total * 2), item.filename)
        try:
            cached = cache.load(item.path, extractor.model_name, extractor.model_version)
            if cached:
                item.embedding_cache_key = cached.cache_key
                item.embedding_model_name = extractor.model_name
                item.embedding_cached_at = cached.cached_at
                item.embedding_dim = cached.dim
                item.embedding_version = extractor.model_version
                embeddings.append(
                    EmbeddingResult(
                        image_path=item.path,
                        vector=normalize_vector(cached.vector).tolist(),
                        model_name=extractor.model_name,
                        model_version=extractor.model_version,
                        method=extractor.method,
                        cache_key=cached.cache_key,
                        cached=True,
                        cached_at=cached.cached_at,
                        dim=cached.dim,
                        source="cache",
                    )
                )
                cache_hits += 1
            else:
                to_compute.append(Path(item.path))
                cache_misses += 1
        except Exception as exc:
            skipped.append(f"{item.filename}: {exc}")

    if to_compute:
        computed = extractor.extract_batch(to_compute)
        for result in computed:
            result.vector = normalize_vector(result.vector).tolist()
            result = cache.save(result)
            embeddings.append(result)
            item = _item_for_path(items, result.image_path)
            if item:
                item.embedding_cache_key = result.cache_key
                item.embedding_model_name = result.model_name
                item.embedding_cached_at = result.cached_at
                item.embedding_dim = result.dim
                item.embedding_version = result.model_version
        if progress_callback:
            progress_callback(total * 2, max(1, total * 2), "embedding 分组")

    grouping = group_embeddings(items, embeddings, threshold=threshold, method=extractor.method)
    apply_auto_group_assignments(items, grouping)
    for item in items:
        if item.embedding_model_name:
            item.grouping_method = extractor.method
    return ModelPipelineResult(
        grouping=grouping,
        embeddings=embeddings,
        runtime=runtime,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        method=extractor.method,
        skipped=skipped,
    )


def _item_for_path(items: list[PhotoItem], path: Path) -> PhotoItem | None:
    target = Path(path)
    for item in items:
        if Path(item.path) == target:
            return item
    return None
