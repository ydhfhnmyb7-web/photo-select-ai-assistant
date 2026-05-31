from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.core.config import AppConfig, load_config
from app.core.mvp_models import PhotoItem
from app.model_pipeline.embedding_cache import EmbeddingCache
from app.model_pipeline.embedding_extractor import (
    EmbeddingExtractor,
    FallbackEmbeddingExtractor,
    MockEmbeddingExtractor,
    normalize_vector,
)
from app.model_pipeline.embedding_grouping import apply_auto_group_assignments, group_embeddings
from app.model_pipeline.embedding_types import EmbeddingGroupingResult, EmbeddingResult
from app.model_pipeline.openclip_extractor import OpenClipEmbeddingExtractor, OpenClipExtractorUnavailable
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
    extractor = extractor or build_default_extractor(config, runtime)
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
            preprocess_version = getattr(extractor, "preprocess_version", "v1")
            cached = cache.load(item.path, extractor.model_name, extractor.model_version, preprocess_version)
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
        computed = _safe_extract_batch(extractor, to_compute, skipped)
        for result in computed:
            result.vector = normalize_vector(result.vector).tolist()
            try:
                result = cache.save(result, getattr(extractor, "preprocess_version", "v1"))
            except Exception as exc:
                skipped.append(f"{result.image_path.name}: cache save failed: {exc}")
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

    grouping = group_embeddings(
        items,
        embeddings,
        threshold=threshold,
        method=extractor.method,
        grouping_strategy=config.grouping_strategy,
        group_min_similarity_threshold=config.group_min_similarity_threshold,
        max_group_size=config.max_embedding_group_size,
    )
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


def build_default_extractor(config: AppConfig, runtime: RuntimeInfo | None = None) -> EmbeddingExtractor:
    """Select the v0.5 default extractor without requiring large models."""
    if config.embedding_backend == "mock":
        return MockEmbeddingExtractor()
    if config.embedding_backend == "fallback":
        return FallbackEmbeddingExtractor()
    if config.embedding_backend in {"auto", "openclip"}:
        try:
            return OpenClipEmbeddingExtractor(config=config, runtime=runtime)
        except OpenClipExtractorUnavailable as exc:
            _write_model_pipeline_log(f"OpenCLIP extractor unavailable, fallback enabled: {exc}")
            return FallbackEmbeddingExtractor()
        except Exception as exc:
            _write_model_pipeline_log(f"OpenCLIP extractor failed, fallback enabled: {exc}")
            return FallbackEmbeddingExtractor()
    return FallbackEmbeddingExtractor()


def _safe_extract_batch(
    extractor: EmbeddingExtractor,
    paths: list[Path],
    skipped: list[str],
) -> list[EmbeddingResult]:
    try:
        return extractor.extract_batch(paths)
    except Exception as batch_exc:
        computed: list[EmbeddingResult] = []
        for path in paths:
            try:
                computed.extend(extractor.extract_batch([path]))
            except Exception as exc:
                skipped.append(f"{path.name}: {exc or batch_exc}")
        return computed


def _item_for_path(items: list[PhotoItem], path: Path) -> PhotoItem | None:
    target = Path(path)
    for item in items:
        if Path(item.path) == target:
            return item
    return None


def _write_model_pipeline_log(message: str) -> None:
    try:
        from app.core.config import PROJECT_ROOT
        from datetime import datetime

        log_path = PROJECT_ROOT / "logs" / "model_pipeline.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.open("a", encoding="utf-8").write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
        )
    except Exception:
        pass
