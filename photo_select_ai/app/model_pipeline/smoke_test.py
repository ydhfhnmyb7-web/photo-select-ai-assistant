from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from app.core.config import AppConfig
from app.core.mvp_models import PhotoItem
from app.model_pipeline.embedding_cache import EmbeddingCache
from app.model_pipeline.embedding_extractor import EmbeddingExtractor, FallbackEmbeddingExtractor, MockEmbeddingExtractor
from app.model_pipeline.model_pipeline_v1 import ModelPipelineResult, run_model_pipeline_v1
from app.model_pipeline.openclip_extractor import (
    OpenClipEmbeddingExtractor,
    OpenClipExtractorUnavailable,
    diagnose_openclip_availability,
)
from app.model_pipeline.runtime_detector import RuntimeInfo, detect_model_runtime


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class SmokeTestResult:
    report_path: Path
    csv_path: Path
    backend_requested: str
    backend_used: str
    device: str
    openclip_available: bool
    openclip_reasons: list[str]
    pipeline: ModelPipelineResult
    photo_count: int


def run_model_pipeline_smoke(
    input_dir: Path | str,
    backend: str = "auto",
    batch_size: int = 16,
    similarity_threshold: float = 0.86,
    force_refresh_cache: bool = False,
    output_report: Path | str | None = None,
    max_photos: int | None = None,
) -> SmokeTestResult:
    input_path = Path(input_dir)
    output_path = Path(output_report) if output_report else input_path / "model_pipeline_smoke_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")

    config = AppConfig(
        embedding_backend=backend,
        embedding_batch_size=batch_size,
        batch_size=batch_size,
        embedding_similarity_threshold=similarity_threshold,
    )
    runtime = detect_model_runtime(config.use_gpu)
    extractor, openclip_reasons = _build_smoke_extractor(config, runtime)
    cache = EmbeddingCache()
    image_paths = scan_image_paths(input_path)
    if max_photos:
        image_paths = image_paths[: max(0, int(max_photos))]
    items = [_photo_item_from_path(path) for path in image_paths]

    if force_refresh_cache:
        preprocess_version = getattr(extractor, "preprocess_version", "v1")
        for path in image_paths:
            cache.delete(path, extractor.model_name, extractor.model_version, preprocess_version)

    started = time.perf_counter()
    pipeline = run_model_pipeline_v1(
        items,
        config=config,
        extractor=extractor,
        cache=cache,
        threshold=similarity_threshold,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    pipeline.elapsed_ms = elapsed_ms

    write_smoke_markdown(output_path, input_path, config, runtime, extractor, openclip_reasons, items, pipeline)
    write_smoke_csv(csv_path, items)
    return SmokeTestResult(
        report_path=output_path,
        csv_path=csv_path,
        backend_requested=backend,
        backend_used=_backend_label(pipeline.method),
        device=_device_label(extractor, runtime),
        openclip_available=pipeline.method == "embedding_openclip",
        openclip_reasons=openclip_reasons,
        pipeline=pipeline,
        photo_count=len(items),
    )


def scan_image_paths(input_dir: Path | str) -> list[Path]:
    root = Path(input_dir)
    return sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and "cache" not in {part.lower() for part in path.parts}
        ],
        key=lambda value: str(value).lower(),
    )


def write_smoke_markdown(
    output_path: Path,
    input_path: Path,
    config: AppConfig,
    runtime: RuntimeInfo,
    extractor: EmbeddingExtractor,
    openclip_reasons: list[str],
    items: list[PhotoItem],
    pipeline: ModelPipelineResult,
) -> None:
    grouped = [item for item in items if item.auto_group_id]
    groups: dict[str, list[PhotoItem]] = {}
    for item in grouped:
        groups.setdefault(item.auto_group_id, []).append(item)
    lines = [
        "# Model Pipeline Smoke Test",
        "",
        "## Summary",
        f"- 输入目录：`{input_path}`",
        f"- backend 请求：`{config.embedding_backend}`",
        f"- backend 实际：`{_backend_label(pipeline.method)}`",
        f"- grouping_method：`{pipeline.method}`",
        f"- device 实际：`{_device_label(extractor, runtime)}`",
        f"- embedding_model_name：`{extractor.model_name}`",
        f"- embedding_dim：`{_embedding_dim(items)}`",
        f"- 总照片数：{len(items)}",
        f"- 成功提取 embedding 数：{len(pipeline.embeddings)}",
        f"- cache 命中数：{pipeline.cache_hits}",
        f"- fallback 数：{len(pipeline.embeddings) if pipeline.method == 'ahash_fallback' else 0}",
        f"- skipped 数：{len(pipeline.skipped)}",
        f"- 自动分组数量：{len(pipeline.grouping.groups)}",
        f"- 总耗时：{pipeline.elapsed_ms:.1f} ms",
        "",
        "## Runtime",
        f"- torch installed：{runtime.torch_installed}",
        f"- torch CUDA available：{runtime.torch_cuda_available}",
        f"- GPU：{runtime.gpu_name or '-'}",
        f"- onnxruntime installed：{runtime.onnxruntime_installed}",
        f"- onnx providers：{', '.join(runtime.onnx_providers) or '-'}",
        "",
        "## OpenCLIP Availability",
    ]
    if pipeline.method == "embedding_openclip":
        lines.append("- OpenCLIP：可用")
    else:
        lines.append("- OpenCLIP：不可用或未使用")
        for reason in openclip_reasons or ["未选择 OpenCLIP backend"]:
            lines.append(f"- 原因：{reason}")
    lines.extend(["", "## Groups"])
    if not groups:
        lines.append("未生成自动相似组。")
    group_similarity = {group.group_id: group.average_similarity for group in pipeline.grouping.groups}
    for group_id, group_items in sorted(groups.items()):
        avg_confidence = sum(item.auto_group_confidence for item in group_items) / max(1, len(group_items))
        avg_similarity = group_similarity.get(group_id, _average_nearest_similarity(group_items))
        lines.extend(
            [
                f"### {group_id}",
                f"- 数量：{len(group_items)}",
                f"- 平均 confidence：{avg_confidence:.3f}",
                f"- 平均 nearest similarity：{avg_similarity:.3f}",
                "- 文件：",
            ]
        )
        for item in group_items:
            lines.append(f"  - {item.filename}")
    lines.extend(["", "## Per Photo"])
    for item in items:
        lines.append(
            f"- `{item.filename}` | group={item.auto_group_id or '-'} | "
            f"rank={item.auto_group_rank}/{item.auto_group_size} | "
            f"confidence={item.auto_group_confidence:.3f} | method={item.grouping_method or '-'} | "
            f"reason={item.auto_group_reason or '-'}"
        )
    if pipeline.skipped:
        lines.extend(["", "## Skipped"])
        lines.extend([f"- {value}" for value in pipeline.skipped])
    output_path.write_text("\n".join(lines), encoding="utf-8-sig")


def write_smoke_csv(csv_path: Path, items: list[PhotoItem]) -> None:
    fields = [
        "file_name",
        "path",
        "auto_group_id",
        "auto_group_confidence",
        "auto_group_rank",
        "auto_group_size",
        "auto_group_reason",
        "grouping_method",
        "embedding_model_name",
        "embedding_dim",
        "embedding_cached_at",
        "embedding_cache_key",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "file_name": item.filename,
                    "path": str(item.path),
                    "auto_group_id": item.auto_group_id,
                    "auto_group_confidence": item.auto_group_confidence,
                    "auto_group_rank": item.auto_group_rank,
                    "auto_group_size": item.auto_group_size,
                    "auto_group_reason": item.auto_group_reason,
                    "grouping_method": item.grouping_method,
                    "embedding_model_name": item.embedding_model_name,
                    "embedding_dim": item.embedding_dim,
                    "embedding_cached_at": item.embedding_cached_at,
                    "embedding_cache_key": item.embedding_cache_key,
                }
            )


def _build_smoke_extractor(config: AppConfig, runtime: RuntimeInfo) -> tuple[EmbeddingExtractor, list[str]]:
    if config.embedding_backend == "mock":
        return MockEmbeddingExtractor(), []
    if config.embedding_backend == "fallback":
        return FallbackEmbeddingExtractor(), []
    reasons = diagnose_openclip_availability(config)
    try:
        return OpenClipEmbeddingExtractor(config=config, runtime=runtime), reasons
    except OpenClipExtractorUnavailable as exc:
        return FallbackEmbeddingExtractor(), [*reasons, str(exc)]
    except Exception as exc:
        return FallbackEmbeddingExtractor(), [*reasons, f"加载失败：{exc}"]


def _photo_item_from_path(path: Path) -> PhotoItem:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
    except Exception:
        width, height = 0, 0
    size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
    return PhotoItem(
        path=path,
        filename=path.name,
        thumbnail_path=None,
        width=width,
        height=height,
        file_size_mb=round(size_mb, 4),
    )


def _device_label(extractor: EmbeddingExtractor, runtime: RuntimeInfo) -> str:
    if hasattr(extractor, "device_label"):
        return str(getattr(extractor, "device_label"))
    if extractor.method == "mock_embedding":
        return "mock"
    if extractor.method == "ahash_fallback":
        return "cpu/fallback"
    return runtime.preferred_backend


def _backend_label(method: str) -> str:
    if method == "embedding_openclip":
        return "openclip"
    if method == "mock_embedding":
        return "mock"
    if method == "ahash_fallback":
        return "fallback"
    return method or "unknown"


def _embedding_dim(items: list[PhotoItem]) -> int:
    for item in items:
        if item.embedding_dim:
            return item.embedding_dim
    return 0


def _average_nearest_similarity(items: list[PhotoItem]) -> float:
    values = []
    for item in items:
        reason = item.auto_group_reason or ""
        if "fallback" in reason.lower():
            values.append(item.auto_group_confidence / 0.72 if item.auto_group_confidence else 0.0)
        else:
            values.append(item.auto_group_confidence / 0.96 if item.auto_group_confidence else 0.0)
    return sum(values) / max(1, len(values))


def result_to_json(result: SmokeTestResult) -> str:
    return json.dumps(
        {
            "report_path": str(result.report_path),
            "csv_path": str(result.csv_path),
            "backend_requested": result.backend_requested,
            "backend_used": result.backend_used,
            "device": result.device,
            "openclip_available": result.openclip_available,
            "openclip_reasons": result.openclip_reasons,
            "photo_count": result.photo_count,
            "embedding_count": len(result.pipeline.embeddings),
            "cache_hits": result.pipeline.cache_hits,
            "skipped": len(result.pipeline.skipped),
            "groups": len(result.pipeline.grouping.groups),
        },
        ensure_ascii=False,
        indent=2,
    )
