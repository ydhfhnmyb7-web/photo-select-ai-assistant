from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.model_pipeline.smoke_test import run_model_pipeline_smoke


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (96, 72), color).save(path)


def test_smoke_test_mock_backend_writes_reports() -> None:
    root = Path("_tmp_model_pipeline_smoke_mock")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        for name in ["same_1.jpg", "same_2.jpg", "same_3.jpg", "different.jpg"]:
            _make_image(root / name, (160, 160, 160))
        report = root / "smoke.md"

        result = run_model_pipeline_smoke(
            root,
            backend="mock",
            batch_size=2,
            similarity_threshold=0.86,
            force_refresh_cache=True,
            output_report=report,
        )

        assert result.backend_used == "mock"
        assert result.device == "mock"
        assert result.photo_count == 4
        assert len(result.pipeline.grouping.groups) == 1
        assert report.exists()
        assert report.with_suffix(".csv").exists()
        text = report.read_text(encoding="utf-8")
        assert "backend 实际：`mock`" in text
        assert "same_1.jpg" in text
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_smoke_test_fallback_backend_writes_reports_without_database() -> None:
    root = Path("_tmp_model_pipeline_smoke_fallback")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        p1 = root / "copy_1.jpg"
        p2 = root / "copy_2.jpg"
        _make_image(p1, (230, 230, 230))
        shutil.copy2(p1, p2)
        report = root / "fallback_smoke.md"

        result = run_model_pipeline_smoke(
            root,
            backend="fallback",
            batch_size=2,
            similarity_threshold=0.94,
            force_refresh_cache=True,
            output_report=report,
        )

        assert result.backend_used == "fallback"
        assert result.device == "cpu/fallback"
        assert result.photo_count == 2
        assert len(result.pipeline.embeddings) == 2
        assert report.exists()
        assert report.with_suffix(".csv").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_smoke_test_auto_reports_openclip_fallback_reason() -> None:
    import app.model_pipeline.smoke_test as smoke_module

    original = smoke_module.OpenClipEmbeddingExtractor

    class _BrokenOpenClip:
        def __init__(self, *_args, **_kwargs):
            raise smoke_module.OpenClipExtractorUnavailable("forced OpenCLIP missing")

    root = Path("_tmp_model_pipeline_smoke_auto")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        _make_image(root / "sample.jpg", (120, 130, 140))
        smoke_module.OpenClipEmbeddingExtractor = _BrokenOpenClip
        report = root / "auto_smoke.md"

        result = run_model_pipeline_smoke(
            root,
            backend="auto",
            batch_size=1,
            similarity_threshold=0.94,
            force_refresh_cache=True,
            output_report=report,
        )

        assert result.backend_used == "fallback"
        assert any("forced OpenCLIP missing" in reason for reason in result.openclip_reasons)
        assert "forced OpenCLIP missing" in report.read_text(encoding="utf-8")
    finally:
        smoke_module.OpenClipEmbeddingExtractor = original
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_smoke_test_mock_backend_writes_reports()
    test_smoke_test_fallback_backend_writes_reports_without_database()
    test_smoke_test_auto_reports_openclip_fallback_reason()
    print("model pipeline smoke tests passed")
