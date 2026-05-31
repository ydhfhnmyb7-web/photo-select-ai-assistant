from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import AppConfig
from app.model_pipeline.embedding_cache import EmbeddingCache
from app.model_pipeline.embedding_extractor import FallbackEmbeddingExtractor
from app.model_pipeline.model_pipeline_v1 import build_default_extractor, run_model_pipeline_v1
from app.model_pipeline.openclip_extractor import OpenClipEmbeddingExtractor, OpenClipExtractorUnavailable


class _MissingModelManager:
    def semantic_marker(self, _spec):
        return Path("_definitely_missing_openclip_marker") / "downloaded.json"

    def semantic_cache_dir(self, _spec):
        path = Path("_tmp_openclip_missing_cache")
        path.mkdir(exist_ok=True)
        return path


def _make_image(path: Path) -> None:
    Image.new("RGB", (64, 64), (180, 180, 180)).save(path)


def test_openclip_extractor_unavailable_does_not_crash() -> None:
    try:
        OpenClipEmbeddingExtractor(config=AppConfig(embedding_backend="openclip"), manager=_MissingModelManager())
    except OpenClipExtractorUnavailable:
        pass


def test_auto_backend_falls_back_when_openclip_unavailable() -> None:
    import app.model_pipeline.model_pipeline_v1 as pipeline_module

    original = pipeline_module.OpenClipEmbeddingExtractor

    class _BrokenOpenClip:
        def __init__(self, *_args, **_kwargs):
            raise OpenClipExtractorUnavailable("forced missing model")

    try:
        pipeline_module.OpenClipEmbeddingExtractor = _BrokenOpenClip
        extractor = build_default_extractor(AppConfig(embedding_backend="auto"))
        assert isinstance(extractor, FallbackEmbeddingExtractor)
    finally:
        pipeline_module.OpenClipEmbeddingExtractor = original


def test_openclip_backend_falls_back_without_overwriting_manual_fields() -> None:
    import app.model_pipeline.model_pipeline_v1 as pipeline_module
    from app.core.mvp_models import PhotoItem

    root = Path("_tmp_openclip_fallback_pipeline")
    cache_root = root / "cache"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        p1 = root / "copy_1.jpg"
        p2 = root / "copy_2.jpg"
        _make_image(p1)
        shutil.copy2(p1, p2)
        items = [
            PhotoItem(path=p1, filename=p1.name, thumbnail_path=None, width=64, height=64, file_size_mb=0.01),
            PhotoItem(path=p2, filename=p2.name, thumbnail_path=None, width=64, height=64, file_size_mb=0.01),
        ]
        items[0].similar_group_id = "manual_group"
        items[0].similar_group_status = "best"
        items[0].best_in_group = True
        items[0].human_group_decision = '{"decision_source":"human","similar_group_status":"best"}'

        original = pipeline_module.OpenClipEmbeddingExtractor

        class _BrokenOpenClip:
            def __init__(self, *_args, **_kwargs):
                raise OpenClipExtractorUnavailable("forced missing model")

        pipeline_module.OpenClipEmbeddingExtractor = _BrokenOpenClip
        try:
            result = run_model_pipeline_v1(
                items,
                config=AppConfig(embedding_backend="openclip", embedding_similarity_threshold=0.94),
                cache=EmbeddingCache(cache_root),
            )
        finally:
            pipeline_module.OpenClipEmbeddingExtractor = original

        assert result.method == "ahash_fallback"
        assert items[0].grouping_method == "ahash_fallback"
        assert items[0].similar_group_id == "manual_group"
        assert items[0].similar_group_status == "best"
        assert items[0].best_in_group
        assert items[0].human_group_decision
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_embedding_cache_key_includes_model_version_and_preprocess() -> None:
    root = Path("_tmp_openclip_cache_key")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        path = root / "sample.jpg"
        _make_image(path)
        cache = EmbeddingCache(root / "cache")
        base = cache.cache_key(path, "model_a", "version_1", "preprocess_1")
        changed_model = cache.cache_key(path, "model_b", "version_1", "preprocess_1")
        changed_version = cache.cache_key(path, "model_a", "version_2", "preprocess_1")
        changed_preprocess = cache.cache_key(path, "model_a", "version_1", "preprocess_2")
        assert base != changed_model
        assert base != changed_version
        assert base != changed_preprocess
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_openclip_extractor_unavailable_does_not_crash()
    test_auto_backend_falls_back_when_openclip_unavailable()
    test_openclip_backend_falls_back_without_overwriting_manual_fields()
    test_embedding_cache_key_includes_model_version_and_preprocess()
    print("openclip extractor tests passed")
