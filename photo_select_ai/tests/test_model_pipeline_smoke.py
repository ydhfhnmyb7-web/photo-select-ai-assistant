from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.model_pipeline.smoke_test import (
    evaluate_pair_groups,
    load_manual_groups_csv,
    parse_thresholds,
    run_model_pipeline_smoke,
)


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
        text = report.read_text(encoding="utf-8-sig")
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
        assert "forced OpenCLIP missing" in report.read_text(encoding="utf-8-sig")
    finally:
        smoke_module.OpenClipEmbeddingExtractor = original
        shutil.rmtree(root, ignore_errors=True)


def test_threshold_sweep_mock_backend_writes_comparison_report() -> None:
    root = Path("_tmp_model_pipeline_smoke_sweep")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        for name in ["same_1.jpg", "same_2.jpg", "same_3.jpg", "different.jpg"]:
            _make_image(root / name, (160, 160, 160))
        report = root / "sweep.md"

        result = run_model_pipeline_smoke(
            root,
            backend="mock",
            batch_size=2,
            similarity_threshold=0.86,
            thresholds=[0.78, 0.86, 0.94],
            force_refresh_cache=True,
            output_report=report,
        )

        assert [entry.threshold for entry in result.threshold_results] == [0.78, 0.86, 0.94]
        assert all(entry.group_count == 1 for entry in result.threshold_results)
        text = report.read_text(encoding="utf-8-sig")
        assert "Threshold Summary" in text
        assert "推荐阈值" in text
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_manual_group_csv_is_parsed() -> None:
    root = Path("_tmp_model_pipeline_manual_csv")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        csv_path = root / "manual.csv"
        csv_path.write_text("file_name,manual_group\n001.jpg,A\n002.jpg,A\n003.jpg,B\n", encoding="utf-8-sig")
        groups = load_manual_groups_csv(csv_path)
        assert groups == {"001.jpg": "A", "002.jpg": "A", "003.jpg": "B"}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_pair_metrics_are_calculated_correctly() -> None:
    class _Item:
        def __init__(self, filename: str):
            self.filename = filename

    grouped_by_id = {
        "auto_001": [_Item("001.jpg"), _Item("002.jpg")],
        "auto_002": [_Item("003.jpg"), _Item("004.jpg")],
    }
    manual_groups = {
        "001.jpg": "A",
        "002.jpg": "A",
        "003.jpg": "B",
        "004.jpg": "C",
    }
    metrics = evaluate_pair_groups(["001.jpg", "002.jpg", "003.jpg", "004.jpg"], grouped_by_id, manual_groups)
    assert metrics.true_positive_pairs == 1
    assert metrics.false_positive_pairs == 1
    assert metrics.false_negative_pairs == 0
    assert metrics.pair_precision == 0.5
    assert metrics.pair_recall == 1.0
    assert round(metrics.pair_f1, 4) == 0.6667
    assert metrics.over_merge_count == 1
    assert metrics.over_split_count == 0


def test_smoke_test_with_manual_groups_computes_metrics() -> None:
    root = Path("_tmp_model_pipeline_smoke_manual")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        for name in ["same_1.jpg", "same_2.jpg", "same_3.jpg", "different.jpg"]:
            _make_image(root / name, (160, 160, 160))
        manual_csv = root / "manual.csv"
        manual_csv.write_text(
            "file_name,manual_group\nsame_1.jpg,A\nsame_2.jpg,A\nsame_3.jpg,A\ndifferent.jpg,B\n",
            encoding="utf-8-sig",
        )
        report = root / "manual_sweep.md"

        result = run_model_pipeline_smoke(
            root,
            backend="mock",
            batch_size=2,
            similarity_threshold=0.86,
            thresholds=parse_thresholds("0.78,0.86,0.94"),
            manual_groups_csv=manual_csv,
            force_refresh_cache=True,
            output_report=report,
        )

        assert result.threshold_results[0].evaluation is not None
        assert result.threshold_results[0].evaluation.pair_f1 == 1.0
        assert result.recommended_threshold in {0.78, 0.86, 0.94}
        assert "人工标注评估" in report.read_text(encoding="utf-8-sig")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_smoke_test_mock_backend_writes_reports()
    test_smoke_test_fallback_backend_writes_reports_without_database()
    test_smoke_test_auto_reports_openclip_fallback_reason()
    test_threshold_sweep_mock_backend_writes_comparison_report()
    test_manual_group_csv_is_parsed()
    test_pair_metrics_are_calculated_correctly()
    test_smoke_test_with_manual_groups_computes_metrics()
    print("model pipeline smoke tests passed")
