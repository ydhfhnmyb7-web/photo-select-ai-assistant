from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps

from app.core.config import AppConfig
from app.core.mvp_models import PhotoItem
from app.model_pipeline.embedding_cache import EmbeddingCache
from app.model_pipeline.embedding_extractor import EmbeddingExtractor, FallbackEmbeddingExtractor, MockEmbeddingExtractor
from app.model_pipeline.embedding_grouping import group_embeddings
from app.model_pipeline.embedding_types import EmbeddingGroupingResult, EmbeddingResult
from app.model_pipeline.model_pipeline_v1 import ModelPipelineResult, run_model_pipeline_v1
from app.model_pipeline.openclip_extractor import (
    OpenClipEmbeddingExtractor,
    OpenClipExtractorUnavailable,
    diagnose_openclip_availability,
)
from app.model_pipeline.runtime_detector import RuntimeInfo, detect_model_runtime


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LOW_CONFIDENCE_GROUP_THRESHOLD = 0.74


@dataclass
class PairDiagnostic:
    file_a: str
    file_b: str
    manual_same_group: bool
    predicted_same_group: bool
    error_type: str
    similarity: float = 0.0
    predicted_group: str = ""
    manual_group_a: str = ""
    manual_group_b: str = ""


@dataclass
class PairEvaluation:
    pair_precision: float = 0.0
    pair_recall: float = 0.0
    pair_f1: float = 0.0
    true_positive_pairs: int = 0
    false_positive_pairs: int = 0
    false_negative_pairs: int = 0
    predicted_pair_count: int = 0
    manual_pair_count: int = 0
    over_merge_count: int = 0
    over_split_count: int = 0
    pair_diagnostics: list[PairDiagnostic] = field(default_factory=list)


@dataclass
class SmokeGroupDetail:
    group_id: str
    files: list[str]
    average_similarity: float = 0.0
    min_similarity: float = 0.0
    max_similarity: float = 0.0
    average_confidence: float = 0.0
    low_confidence: bool = False
    suspected_over_merged: bool = False
    high_risk_overmerge: bool = False
    sequence_break_reason: str = ""

    @property
    def size(self) -> int:
        return len(self.files)


@dataclass
class ThresholdSweepResult:
    threshold: float
    group_count: int
    max_group_size: int
    singleton_count: int
    average_group_size: float
    low_confidence_group_count: int
    high_risk_group_count: int = 0
    worst_group_min_similarity: float = 0.0
    groups: list[SmokeGroupDetail] = field(default_factory=list)
    ungrouped_files: list[str] = field(default_factory=list)
    low_confidence_groups: list[str] = field(default_factory=list)
    suspected_over_merged_groups: list[str] = field(default_factory=list)
    near_miss_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    evaluation: PairEvaluation | None = None


@dataclass
class SmokeTestResult:
    report_path: Path
    csv_path: Path
    pair_diagnostics_path: Path | None
    backend_requested: str
    backend_used: str
    device: str
    openclip_available: bool
    openclip_reasons: list[str]
    pipeline: ModelPipelineResult
    photo_count: int
    threshold_results: list[ThresholdSweepResult] = field(default_factory=list)
    recommended_threshold: float = 0.0
    recommended_reason: str = ""
    manual_groups_path: Path | None = None


def run_model_pipeline_smoke(
    input_dir: Path | str,
    backend: str = "auto",
    batch_size: int = 16,
    similarity_threshold: float = 0.94,
    force_refresh_cache: bool = False,
    output_report: Path | str | None = None,
    max_photos: int | None = None,
    thresholds: Sequence[float] | None = None,
    manual_groups_csv: Path | str | None = None,
    grouping_strategy: str = "complete_linkage",
    group_min_similarity_threshold: float = 0.92,
    max_group_size: int = 25,
    sequence_window_size: int = 5,
    max_filename_gap: int = 80,
    max_time_gap_seconds: int = 120,
    filename_continuity_bonus: float = 0.01,
) -> SmokeTestResult:
    input_path = Path(input_dir)
    output_path = Path(output_report) if output_report else input_path / "model_pipeline_smoke_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")
    pair_diagnostics_path = output_path.with_name(f"{output_path.stem}_manual_pairs.csv") if manual_groups_csv else None

    config = AppConfig(
        embedding_backend=backend,
        embedding_batch_size=batch_size,
        batch_size=batch_size,
        embedding_similarity_threshold=similarity_threshold,
        grouping_strategy=grouping_strategy,
        group_min_similarity_threshold=group_min_similarity_threshold,
        max_embedding_group_size=max_group_size,
        sequence_window_size=sequence_window_size,
        max_filename_gap=max_filename_gap,
        max_time_gap_seconds=max_time_gap_seconds,
        filename_continuity_bonus=filename_continuity_bonus,
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

    manual_groups = load_manual_groups_csv(manual_groups_csv) if manual_groups_csv else {}
    sweep_thresholds = _normalize_thresholds(thresholds or [similarity_threshold])
    threshold_results = build_threshold_sweep(
        items=items,
        embeddings=pipeline.embeddings,
        method=pipeline.method,
        thresholds=sweep_thresholds,
        manual_groups=manual_groups,
        grouping_strategy=config.grouping_strategy,
        group_min_similarity_threshold=config.group_min_similarity_threshold,
        max_group_size=config.max_embedding_group_size,
        sequence_window_size=config.sequence_window_size,
        max_filename_gap=config.max_filename_gap,
        max_time_gap_seconds=config.max_time_gap_seconds,
        filename_continuity_bonus=config.filename_continuity_bonus,
    )
    recommended_threshold, recommended_reason = recommend_threshold(threshold_results, bool(manual_groups))

    write_smoke_markdown(
        output_path=output_path,
        input_path=input_path,
        config=config,
        runtime=runtime,
        extractor=extractor,
        openclip_reasons=openclip_reasons,
        items=items,
        pipeline=pipeline,
        threshold_results=threshold_results,
        recommended_threshold=recommended_threshold,
        recommended_reason=recommended_reason,
        manual_groups_path=Path(manual_groups_csv) if manual_groups_csv else None,
    )
    write_smoke_csv(csv_path, items, manual_groups)
    if pair_diagnostics_path:
        primary_result = _find_threshold_result(threshold_results, config.embedding_similarity_threshold)
        diagnostics = primary_result.evaluation.pair_diagnostics if primary_result and primary_result.evaluation else []
        write_pair_diagnostics_csv(pair_diagnostics_path, diagnostics)
    return SmokeTestResult(
        report_path=output_path,
        csv_path=csv_path,
        pair_diagnostics_path=pair_diagnostics_path,
        backend_requested=backend,
        backend_used=_backend_label(pipeline.method),
        device=_device_label(extractor, runtime),
        openclip_available=pipeline.method == "embedding_openclip",
        openclip_reasons=openclip_reasons,
        pipeline=pipeline,
        photo_count=len(items),
        threshold_results=threshold_results,
        recommended_threshold=recommended_threshold,
        recommended_reason=recommended_reason,
        manual_groups_path=Path(manual_groups_csv) if manual_groups_csv else None,
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


def load_manual_groups_csv(csv_path: Path | str | None) -> dict[str, str]:
    if not csv_path:
        return {}
    path = Path(csv_path)
    manual_groups: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            file_name = (row.get("file_name") or row.get("filename") or "").strip()
            manual_group = (row.get("manual_group") or row.get("group") or "").strip()
            if file_name and manual_group:
                manual_groups[file_name] = manual_group
    return manual_groups


def parse_thresholds(value: str | None) -> list[float]:
    if not value:
        return []
    thresholds: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        thresholds.append(float(part))
    return _normalize_thresholds(thresholds)


def build_threshold_sweep(
    items: Sequence[PhotoItem],
    embeddings: Sequence[EmbeddingResult],
    method: str,
    thresholds: Sequence[float],
    manual_groups: dict[str, str] | None = None,
    grouping_strategy: str = "complete_linkage",
    group_min_similarity_threshold: float = 0.92,
    max_group_size: int = 25,
    sequence_window_size: int = 5,
    max_filename_gap: int = 80,
    max_time_gap_seconds: int = 120,
    filename_continuity_bonus: float = 0.01,
) -> list[ThresholdSweepResult]:
    manual_groups = manual_groups or {}
    results: list[ThresholdSweepResult] = []
    embedding_map = _embedding_map(embeddings)
    for threshold in _normalize_thresholds(thresholds):
        grouping = group_embeddings(
            items,
            embeddings,
            threshold=threshold,
            method=method,
            grouping_strategy=grouping_strategy,
            group_min_similarity_threshold=group_min_similarity_threshold,
            max_group_size=max_group_size,
            sequence_window_size=sequence_window_size,
            max_filename_gap=max_filename_gap,
            max_time_gap_seconds=max_time_gap_seconds,
            filename_continuity_bonus=filename_continuity_bonus,
        )
        result = build_threshold_result(items, grouping, embedding_map, threshold, manual_groups)
        results.append(result)
    return results


def build_threshold_result(
    items: Sequence[PhotoItem],
    grouping: EmbeddingGroupingResult,
    embedding_map: dict[Path, np.ndarray],
    threshold: float,
    manual_groups: dict[str, str],
) -> ThresholdSweepResult:
    grouped_by_id: dict[str, list[PhotoItem]] = {}
    for item in items:
        assignment = grouping.assignments.get(Path(item.path))
        if assignment and assignment.auto_group_id:
            grouped_by_id.setdefault(assignment.auto_group_id, []).append(item)

    groups: list[SmokeGroupDetail] = []
    grouped_files: set[str] = set()
    grouping_group_by_id = {group.group_id: group for group in grouping.groups}
    for group_id, group_items in sorted(grouped_by_id.items()):
        files = [item.filename for item in group_items]
        grouped_files.update(files)
        grouping_group = grouping_group_by_id.get(group_id)
        similarities = _pair_similarities(group_items, embedding_map)
        average_similarity = grouping_group.average_similarity if grouping_group else _average(similarities)
        min_similarity = grouping_group.min_similarity if grouping_group else (min(similarities) if similarities else 0.0)
        max_similarity = grouping_group.max_similarity if grouping_group else (max(similarities) if similarities else 0.0)
        confidences = [
            grouping.assignments.get(Path(item.path)).auto_group_confidence
            for item in group_items
            if grouping.assignments.get(Path(item.path))
        ]
        average_confidence = _average(confidences)
        manual_labels = {manual_groups.get(file_name, "") for file_name in files if manual_groups.get(file_name, "")}
        suspected_over_merged = (
            len(manual_labels) > 1
            or bool(grouping_group and grouping_group.high_risk_overmerge)
            or len(files) >= max(8, int(len(items) * 0.35) if items else 8)
            or (len(files) > 2 and min_similarity < max(0.0, threshold - 0.06))
        )
        low_confidence = average_confidence < LOW_CONFIDENCE_GROUP_THRESHOLD or (
            min_similarity and min_similarity < max(0.0, threshold - 0.03)
        )
        groups.append(
            SmokeGroupDetail(
                group_id=group_id,
                files=files,
                average_similarity=round(average_similarity, 4),
                min_similarity=round(min_similarity, 4),
                max_similarity=round(max_similarity, 4),
                average_confidence=round(average_confidence, 4),
                low_confidence=low_confidence,
                suspected_over_merged=suspected_over_merged,
                high_risk_overmerge=bool(grouping_group and grouping_group.high_risk_overmerge),
                sequence_break_reason=grouping_group.sequence_break_reason if grouping_group else "",
            )
        )

    ungrouped_files = [item.filename for item in items if item.filename not in grouped_files]
    singleton_count = len(ungrouped_files)
    group_sizes = [group.size for group in groups]
    similarity_lookup = _pair_similarity_lookup(items, embedding_map)
    evaluation = (
        evaluate_pair_groups([item.filename for item in items], grouped_by_id, manual_groups, similarity_lookup)
        if manual_groups
        else None
    )
    near_miss_pairs = find_near_miss_pairs(items, embedding_map, grouping, threshold)
    return ThresholdSweepResult(
        threshold=threshold,
        group_count=len(groups),
        max_group_size=max(group_sizes) if group_sizes else 0,
        singleton_count=singleton_count,
        average_group_size=round(_average(group_sizes), 2),
        low_confidence_group_count=sum(1 for group in groups if group.low_confidence),
        high_risk_group_count=sum(1 for group in groups if group.high_risk_overmerge),
        worst_group_min_similarity=round(min((group.min_similarity for group in groups), default=0.0), 4),
        groups=groups,
        ungrouped_files=ungrouped_files,
        low_confidence_groups=[group.group_id for group in groups if group.low_confidence],
        suspected_over_merged_groups=[group.group_id for group in groups if group.suspected_over_merged],
        near_miss_pairs=near_miss_pairs,
        evaluation=evaluation,
    )


def evaluate_pair_groups(
    file_names: Sequence[str],
    grouped_by_id: dict[str, list[PhotoItem]],
    manual_groups: dict[str, str],
    similarity_lookup: dict[tuple[str, str], float] | None = None,
) -> PairEvaluation:
    similarity_lookup = similarity_lookup or {}
    considered = [file_name for file_name in file_names if manual_groups.get(file_name)]
    predicted_group: dict[str, str] = {file_name: f"single:{file_name}" for file_name in considered}
    for group_id, items in grouped_by_id.items():
        for item in items:
            if item.filename in predicted_group:
                predicted_group[item.filename] = group_id

    tp = fp = fn = 0
    diagnostics: list[PairDiagnostic] = []
    for left, right in combinations(considered, 2):
        manual_same = manual_groups[left] == manual_groups[right]
        predicted_same = predicted_group[left] == predicted_group[right] and not predicted_group[left].startswith("single:")
        error_type = "TN"
        if predicted_same and manual_same:
            tp += 1
            error_type = "TP"
        elif predicted_same and not manual_same:
            fp += 1
            error_type = "FP"
        elif manual_same and not predicted_same:
            fn += 1
            error_type = "FN"
        predicted_label = (
            predicted_group[left]
            if predicted_group[left] == predicted_group[right]
            else f"{_display_group(predicted_group[left])};{_display_group(predicted_group[right])}"
        )
        diagnostics.append(
            PairDiagnostic(
                file_a=left,
                file_b=right,
                manual_same_group=manual_same,
                predicted_same_group=predicted_same,
                error_type=error_type,
                similarity=round(_lookup_pair_similarity(similarity_lookup, left, right), 4),
                predicted_group=_display_group(predicted_label),
                manual_group_a=manual_groups[left],
                manual_group_b=manual_groups[right],
            )
        )

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    over_merge = 0
    for items in grouped_by_id.values():
        labels = {manual_groups.get(item.filename, "") for item in items if manual_groups.get(item.filename, "")}
        if len(labels) > 1:
            over_merge += 1

    manual_to_files: dict[str, list[str]] = {}
    for file_name in considered:
        manual_to_files.setdefault(manual_groups[file_name], []).append(file_name)
    over_split = 0
    for files in manual_to_files.values():
        if len(files) < 2:
            continue
        predicted_clusters = {predicted_group[file_name] for file_name in files}
        if len(predicted_clusters) > 1:
            over_split += 1

    return PairEvaluation(
        pair_precision=round(precision, 4),
        pair_recall=round(recall, 4),
        pair_f1=round(f1, 4),
        true_positive_pairs=tp,
        false_positive_pairs=fp,
        false_negative_pairs=fn,
        predicted_pair_count=tp + fp,
        manual_pair_count=tp + fn,
        over_merge_count=over_merge,
        over_split_count=over_split,
        pair_diagnostics=diagnostics,
    )


def find_near_miss_pairs(
    items: Sequence[PhotoItem],
    embedding_map: dict[Path, np.ndarray],
    grouping: EmbeddingGroupingResult,
    threshold: float,
    limit: int = 8,
) -> list[tuple[str, str, float]]:
    group_by_file = {
        path.name: assignment.auto_group_id
        for path, assignment in grouping.assignments.items()
        if assignment.auto_group_id
    }
    near_misses: list[tuple[str, str, float]] = []
    lower_bound = max(-1.0, threshold - 0.03)
    for left, right in combinations(items, 2):
        if group_by_file.get(left.filename) and group_by_file.get(left.filename) == group_by_file.get(right.filename):
            continue
        similarity = _cosine_for_items(left, right, embedding_map)
        if lower_bound <= similarity < threshold:
            near_misses.append((left.filename, right.filename, round(similarity, 4)))
    return sorted(near_misses, key=lambda value: value[2], reverse=True)[:limit]


def recommend_threshold(results: Sequence[ThresholdSweepResult], has_manual_groups: bool) -> tuple[float, str]:
    if not results:
        return 0.0, "没有可用阈值结果。"
    if has_manual_groups:
        best = max(
            results,
            key=lambda result: (
                result.evaluation.pair_f1 if result.evaluation else 0.0,
                result.evaluation.pair_precision if result.evaluation else 0.0,
                result.evaluation.pair_recall if result.evaluation else 0.0,
                -result.evaluation.over_merge_count if result.evaluation else 0,
                -result.evaluation.over_split_count if result.evaluation else 0,
            ),
        )
        score = best.evaluation.pair_f1 if best.evaluation else 0.0
        return best.threshold, f"基于人工标注 pair F1 最高推荐，F1={score:.3f}。"

    total = max(1, results[0].singleton_count + sum(group.size for group in results[0].groups))

    def heuristic(result: ThresholdSweepResult) -> float:
        grouped_count = total - result.singleton_count
        too_large_penalty = max(0, result.max_group_size - max(6, int(total * 0.35))) * 0.8
        near_miss_bonus = min(2, len(result.near_miss_pairs)) * 0.08
        return (
            grouped_count * 0.12
            + result.group_count * 0.45
            - result.low_confidence_group_count * 0.9
            - len(result.suspected_over_merged_groups) * 1.4
            - too_large_penalty
            - abs(result.threshold - 0.94) * 0.6
            + near_miss_bonus
        )

    best = max(results, key=heuristic)
    return (
        best.threshold,
        "未提供人工标注，按组数量、低置信度组、疑似过度合并和默认阈值距离综合推荐。",
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
    threshold_results: Sequence[ThresholdSweepResult],
    recommended_threshold: float,
    recommended_reason: str,
    manual_groups_path: Path | None = None,
) -> None:
    primary_result = _find_threshold_result(threshold_results, config.embedding_similarity_threshold)
    lines = [
        "# Model Pipeline Smoke Test",
        "",
        "## Summary",
        f"- 输入目录：`{input_path}`",
        f"- backend 请求：`{config.embedding_backend}`",
        f"- backend 实际：`{_backend_label(pipeline.method)}`",
        f"- grouping_method：`{pipeline.method}`",
        f"- grouping_strategy：`{config.grouping_strategy}`",
        f"- group_min_similarity_threshold：`{config.group_min_similarity_threshold}`",
        f"- max_group_size：`{config.max_embedding_group_size}`",
        f"- sequence_window_size：`{config.sequence_window_size}`",
        f"- max_filename_gap：`{config.max_filename_gap}`",
        f"- max_time_gap_seconds：`{config.max_time_gap_seconds}`",
        f"- filename_continuity_bonus：`{config.filename_continuity_bonus}`",
        f"- device 实际：`{_device_label(extractor, runtime)}`",
        f"- embedding_model_name：`{extractor.model_name}`",
        f"- embedding_dim：`{_embedding_dim(items)}`",
        f"- 总照片数：{len(items)}",
        f"- 成功提取 embedding 数：{len(pipeline.embeddings)}",
        f"- cache 命中数：{pipeline.cache_hits}",
        f"- fallback 数：{len(pipeline.embeddings) if pipeline.method == 'ahash_fallback' else 0}",
        f"- skipped 数：{len(pipeline.skipped)}",
        f"- 当前阈值 AI 相似候选组数量：{primary_result.group_count if primary_result else len(pipeline.grouping.groups)}",
        f"- 推荐阈值：{recommended_threshold:.2f}",
        f"- 推荐理由：{recommended_reason}",
        f"- 人工标注 CSV：`{manual_groups_path}`" if manual_groups_path else "- 人工标注 CSV：未提供",
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

    lines.extend(["", "## Threshold Summary", ""])
    header = [
        "| threshold | groups | max group | singletons | avg group | worst min sim | low confidence | high risk | over merge? | near split hints | precision | recall | F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(header)
    for result in threshold_results:
        if result.evaluation:
            precision_text = f"{result.evaluation.pair_precision:.3f}"
            recall_text = f"{result.evaluation.pair_recall:.3f}"
            f1_text = f"{result.evaluation.pair_f1:.3f}"
        else:
            precision_text = recall_text = f1_text = "-"
        lines.append(
            f"| {result.threshold:.2f} | {result.group_count} | {result.max_group_size} | "
            f"{result.singleton_count} | {result.average_group_size:.2f} | "
            f"{result.worst_group_min_similarity:.4f} | {result.low_confidence_group_count} | "
            f"{result.high_risk_group_count} | {len(result.suspected_over_merged_groups)} | "
            f"{len(result.near_miss_pairs)} | {precision_text} | {recall_text} | {f1_text} |"
        )

    for result in threshold_results:
        lines.extend(["", f"## Threshold {result.threshold:.2f} Details"])
        if not result.groups:
            lines.append("未生成自动相似组。")
        for group in result.groups:
            flags = []
            if group.low_confidence:
                flags.append("低置信度")
            if group.high_risk_overmerge:
                flags.append("高风险过度合并")
            if group.suspected_over_merged:
                flags.append("疑似过度合并")
            flag_text = f"（{' / '.join(flags)}）" if flags else ""
            lines.extend(
                [
                    f"### {group.group_id} {flag_text}",
                    f"- 数量：{group.size}",
                    f"- 平均相似度：{group.average_similarity:.4f}",
                    f"- 最低相似度：{group.min_similarity:.4f}",
                    f"- 最高相似度：{group.max_similarity:.4f}",
                    f"- 平均 confidence：{group.average_confidence:.4f}",
                    f"- sequence_break_reason：{group.sequence_break_reason or '-'}",
                    "- 文件：",
                ]
            )
            for file_name in group.files:
                lines.append(f"  - {file_name}")

        lines.extend(["", "### 未分组照片"])
        if result.ungrouped_files:
            lines.extend([f"- {file_name}" for file_name in result.ungrouped_files])
        else:
            lines.append("- 无")

        lines.extend(["", "### 低置信度组"])
        lines.append(", ".join(result.low_confidence_groups) if result.low_confidence_groups else "无")

        lines.extend(["", "### 疑似过度合并组"])
        lines.append(", ".join(result.suspected_over_merged_groups) if result.suspected_over_merged_groups else "无")

        lines.extend(["", "### 疑似过度拆分提示"])
        if result.near_miss_pairs:
            for left, right, similarity in result.near_miss_pairs:
                lines.append(f"- {left} ↔ {right}：similarity={similarity:.4f}，接近阈值但未进入同组")
        else:
            lines.append("无明显 near-threshold 拆分提示。")

        if result.evaluation:
            eval_result = result.evaluation
            lines.extend(
                [
                    "",
                    "### 人工标注评估",
                    f"- pair precision：{eval_result.pair_precision:.4f}",
                    f"- pair recall：{eval_result.pair_recall:.4f}",
                    f"- pair F1：{eval_result.pair_f1:.4f}",
                    f"- true positive pairs：{eval_result.true_positive_pairs}",
                    f"- false positive pairs：{eval_result.false_positive_pairs}",
                    f"- false negative pairs：{eval_result.false_negative_pairs}",
                    f"- predicted pair count：{eval_result.predicted_pair_count}",
                    f"- manual pair count：{eval_result.manual_pair_count}",
                    f"- over_merge 数量：{eval_result.over_merge_count}",
                    f"- over_split 数量：{eval_result.over_split_count}",
                ]
            )
            false_positive_examples = [value for value in eval_result.pair_diagnostics if value.error_type == "FP"][:30]
            false_negative_examples = [value for value in eval_result.pair_diagnostics if value.error_type == "FN"][:30]
            lines.extend(["", "### AI 错误合并示例（False Positive Pairs，最多 30 对）"])
            _append_pair_diagnostic_table(lines, false_positive_examples)
            lines.extend(["", "### AI 漏掉的同组示例（False Negative Pairs，最多 30 对）"])
            _append_pair_diagnostic_table(lines, false_negative_examples)

    lines.extend(["", "## Per Photo"])
    for item in items:
        lines.append(
            f"- `{item.filename}` | AI 相似候选组={item.auto_group_id or '-'} | "
            f"rank={item.auto_group_rank}/{item.auto_group_size} | "
            f"confidence={item.auto_group_confidence:.3f} | method={item.grouping_method or '-'} | "
            f"reason={item.auto_group_reason or '-'}"
        )
    if pipeline.skipped:
        lines.extend(["", "## Skipped"])
        lines.extend([f"- {value}" for value in pipeline.skipped])
    output_path.write_text("\n".join(lines), encoding="utf-8-sig")


def write_smoke_csv(csv_path: Path, items: list[PhotoItem], manual_groups: dict[str, str] | None = None) -> None:
    manual_groups = manual_groups or {}
    fields = [
        "file_name",
        "path",
        "manual_group",
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
                    "manual_group": manual_groups.get(item.filename, ""),
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


def write_pair_diagnostics_csv(csv_path: Path, diagnostics: Sequence[PairDiagnostic]) -> None:
    fields = [
        "file_a",
        "file_b",
        "manual_same_group",
        "predicted_same_group",
        "error_type",
        "similarity",
        "predicted_group",
        "manual_group_a",
        "manual_group_b",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in diagnostics:
            writer.writerow(
                {
                    "file_a": item.file_a,
                    "file_b": item.file_b,
                    "manual_same_group": item.manual_same_group,
                    "predicted_same_group": item.predicted_same_group,
                    "error_type": item.error_type,
                    "similarity": f"{item.similarity:.4f}",
                    "predicted_group": item.predicted_group,
                    "manual_group_a": item.manual_group_a,
                    "manual_group_b": item.manual_group_b,
                }
            )


def _append_pair_diagnostic_table(lines: list[str], diagnostics: Sequence[PairDiagnostic]) -> None:
    if not diagnostics:
        lines.append("暂无。")
        return
    lines.extend(
        [
            "| file_a | file_b | predicted_group | manual_group_a | manual_group_b | similarity |",
            "|---|---|---|---|---|---:|",
        ]
    )
    for item in diagnostics:
        lines.append(
            f"| {item.file_a} | {item.file_b} | {item.predicted_group or '-'} | "
            f"{item.manual_group_a or '-'} | {item.manual_group_b or '-'} | {item.similarity:.4f} |"
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


def _normalize_thresholds(thresholds: Sequence[float]) -> list[float]:
    normalized = sorted({round(float(value), 4) for value in thresholds})
    return [value for value in normalized if -1.0 <= value <= 1.0]


def _embedding_map(embeddings: Sequence[EmbeddingResult]) -> dict[Path, np.ndarray]:
    return {Path(result.image_path): _normalize_array(result.vector) for result in embeddings if result.vector}


def _normalize_array(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype="float32").reshape(-1)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 1e-12 else array


def _pair_similarities(items: Sequence[PhotoItem], embedding_map: dict[Path, np.ndarray]) -> list[float]:
    values: list[float] = []
    for left, right in combinations(items, 2):
        values.append(_cosine_for_items(left, right, embedding_map))
    return values


def _pair_similarity_lookup(items: Sequence[PhotoItem], embedding_map: dict[Path, np.ndarray]) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    for left, right in combinations(items, 2):
        key = tuple(sorted((left.filename, right.filename)))
        lookup[key] = _cosine_for_items(left, right, embedding_map)
    return lookup


def _lookup_pair_similarity(lookup: dict[tuple[str, str], float], left: str, right: str) -> float:
    return float(lookup.get(tuple(sorted((left, right))), 0.0))


def _display_group(group_id: str) -> str:
    return "-" if group_id.startswith("single:") else group_id


def _cosine_for_items(left: PhotoItem, right: PhotoItem, embedding_map: dict[Path, np.ndarray]) -> float:
    left_vector = embedding_map.get(Path(left.path))
    right_vector = embedding_map.get(Path(right.path))
    if left_vector is None or right_vector is None:
        return 0.0
    return float(np.dot(left_vector, right_vector))


def _average(values: Sequence[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _find_threshold_result(
    results: Sequence[ThresholdSweepResult],
    threshold: float,
) -> ThresholdSweepResult | None:
    for result in results:
        if abs(result.threshold - threshold) < 1e-6:
            return result
    return results[0] if results else None


def result_to_json(result: SmokeTestResult) -> str:
    return json.dumps(
        {
            "report_path": str(result.report_path),
            "csv_path": str(result.csv_path),
            "pair_diagnostics_path": str(result.pair_diagnostics_path) if result.pair_diagnostics_path else "",
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
            "recommended_threshold": result.recommended_threshold,
            "recommended_reason": result.recommended_reason,
            "grouping_strategy": result.pipeline.grouping.groups[0].grouping_strategy
            if result.pipeline.grouping.groups
            else "",
            "thresholds": [
                {
                    "threshold": threshold.threshold,
                    "group_count": threshold.group_count,
                    "max_group_size": threshold.max_group_size,
                    "singleton_count": threshold.singleton_count,
                    "worst_group_min_similarity": threshold.worst_group_min_similarity,
                    "low_confidence_group_count": threshold.low_confidence_group_count,
                    "high_risk_group_count": threshold.high_risk_group_count,
                    "pair_precision": threshold.evaluation.pair_precision if threshold.evaluation else None,
                    "pair_recall": threshold.evaluation.pair_recall if threshold.evaluation else None,
                    "pair_f1": threshold.evaluation.pair_f1 if threshold.evaluation else None,
                    "true_positive_pairs": threshold.evaluation.true_positive_pairs if threshold.evaluation else None,
                    "false_positive_pairs": threshold.evaluation.false_positive_pairs if threshold.evaluation else None,
                    "false_negative_pairs": threshold.evaluation.false_negative_pairs if threshold.evaluation else None,
                    "predicted_pair_count": threshold.evaluation.predicted_pair_count if threshold.evaluation else None,
                    "manual_pair_count": threshold.evaluation.manual_pair_count if threshold.evaluation else None,
                }
                for threshold in result.threshold_results
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
