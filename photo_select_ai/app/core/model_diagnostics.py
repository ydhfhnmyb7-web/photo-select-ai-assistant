from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

from app.core.config import AppConfig, save_config
from app.core.gpu import detect_ai_environment
from app.core.model_manager import ModelManager, gpu_memory_text


def model_profile_advice(config: AppConfig, avg_ms: float = 0.0) -> str:
    env = detect_ai_environment(config.use_gpu)
    if config.model_profile == "ultra":
        if env.gpu_name and "4070 Ti" in env.gpu_name:
            return "RTX 4070 Ti 日常不建议长期使用 ultra；日常推荐 balanced，精筛优先 accurate。"
        if config.batch_size < 8 or avg_ms > 650:
            return "ultra 吞吐不佳，建议改用 accurate 或 balanced。"
        return "ultra 是实验模式，只建议在少量高价值照片上使用。"
    if config.model_profile == "fast":
        return "fast 适合大量粗筛，正式精筛建议切换 balanced 或 accurate。"
    if config.model_profile == "accurate":
        return "accurate 适合精度优先，RTX 4070 Ti 可优先尝试。"
    return "balanced 是日常审片推荐档位。"


def gpu_low_usage_explanation() -> str:
    return "\n".join(
        [
            "GPU占用率低不一定代表程序错误，常见原因：",
            "1. 图片读取和解码在CPU。",
            "2. batch_size太小。",
            "3. 模型太轻或当前处于fallback。",
            "4. 模型太重导致batch被压低。",
            "5. embedding缓存命中，不需要重新跑GPU。",
            "6. YOLO和OpenCLIP不是同时运行。",
            "7. UI线程等待或进度刷新太频繁。",
            "8. 硬盘读取慢。",
            "9. CPU预处理慢。",
            "10. 当前任务不是GPU任务。",
            "",
            "优先目标：不爆显存、推理稳定、UI不卡、结果可复核、任务可中断。",
        ]
    )


def run_model_self_test(config: AppConfig, sample_paths: list[Path]) -> dict:
    started = time.perf_counter()
    env = detect_ai_environment(config.use_gpu)
    status = ModelManager(config).status()
    sample_paths = sample_paths[:10]
    inference_summary = "当前没有项目照片，已跳过小样本推理。"
    avg_ms = 0.0
    top3_lines: list[str] = []
    smoke_line = "OpenCLIP smoke test：未执行"
    pipeline_status = "fallback"

    if sample_paths and env.cv2_installed:
        try:
            from app.core.ai_classifier import LocalAIClassifier

            classifier = LocalAIClassifier(config)
            smoke = classifier.openclip_smoke_test(sample_paths[0])
            smoke_line = _format_smoke_result(smoke)
            results = classifier.classify_paths(sample_paths)
            avg_ms = float(classifier.performance_stats.get("avg_ms", 0.0))
            pipeline_status = classifier.backend_label
            for result in results[:5]:
                top3 = ", ".join(result.top3_semantic_matches) or "-"
                top3_lines.append(f"- {result.path.name}: {result.ai_primary_category}/{result.ai_secondary_category} | top3 {top3}")
            inference_summary = f"小样本推理完成：{len(results)} 张，平均 {avg_ms:.1f} ms/张。"
        except Exception as exc:
            inference_summary = f"小样本推理失败：{exc}"
            pipeline_status = "异常"

    if "已加载" in status.semantic_status and "已加载" in status.detector_status:
        model_state = "正常"
    elif status.semantic_downloaded or status.detector_downloaded:
        model_state = "部分可用"
    else:
        model_state = "fallback"

    report = "\n".join(
        [
            "模型体检报告",
            "",
            "环境自测：",
            f"- Python版本：{sys.version.split()[0]}",
            f"- torch版本：{env.torch_version or '-'}",
            f"- CUDA可用：{'是' if env.cuda_available else '否'}",
            f"- GPU名称：{env.gpu_name or '-'}",
            f"- GPU显存：{gpu_memory_text()}",
            f"- batch_size：{config.batch_size}",
            f"- use_fp16：{config.use_fp16}",
            "",
            "模型自测：",
            f"- OpenCLIP：{status.semantic_status}",
            f"- YOLO：{status.detector_status}",
            f"- 当前管线：{pipeline_status}",
            f"- 模型状态：{model_state}",
            f"- {smoke_line}",
            "",
            "小样本推理：",
            inference_summary,
            *top3_lines,
            "",
            "推荐设置：",
            f"- {model_profile_advice(config, avg_ms)}",
            f"- 建议batch_size：{recommend_batch_size(config, avg_ms)}",
            "",
            "主要瓶颈：",
            _bottleneck_text(config, status.semantic_status, status.detector_status, avg_ms),
            "",
            gpu_low_usage_explanation(),
        ]
    )
    return {
        "ok": True,
        "report": report,
        "model_state": model_state,
        "avg_ms": avg_ms,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "recommended_batch_size": recommend_batch_size(config, avg_ms),
    }


def auto_tune_config(config: AppConfig, sample_paths: list[Path]) -> dict:
    env = detect_ai_environment(config.use_gpu)
    tested: list[dict] = []
    successful: list[dict] = []
    candidate_batches = _candidate_batches(config.model_profile)

    if not sample_paths:
        fallback_config = replace(config, model_profile="accurate", batch_size=8, use_fp16=False)
        tested.append(_trial_row(8, False, False, 0.0, "失败", "没有样本图片，无法验证配置。"))
        return _auto_tune_result(config, fallback_config, tested, False, "没有样本图片，未写入配置。请导入照片后重新调优。", save=False)

    for batch in candidate_batches:
        row = _run_tune_trial(config, sample_paths, batch, use_fp16=bool(config.use_fp16 and env.cuda_available))
        tested.append(row)
        if row["success"]:
            successful.append(row)

    fp16_attempted = bool(config.use_fp16 and env.cuda_available)
    fp16_all_failed = fp16_attempted and not any(
        row["success"] and row["use_fp16"] and not row.get("fp32_fallback") for row in tested
    )
    if fp16_all_failed:
        for batch in _candidate_batches("accurate"):
            row = _run_tune_trial(replace(config, model_profile="accurate"), sample_paths, batch, use_fp16=False)
            row["fp32_fallback"] = True
            tested.append(row)
            if row["success"]:
                successful.append(row)

    if not successful:
        fallback_config = replace(config, model_profile="accurate", batch_size=8, use_fp16=False)
        return _auto_tune_result(
            config,
            fallback_config,
            tested,
            False,
            "全部测试失败，没有写入推荐配置。建议先用 accurate + batch_size=8 + use_fp16=false 手动测试。",
            save=False,
        )

    if fp16_all_failed:
        fp32_success = [row for row in successful if not row["use_fp16"]]
        preferred = [row for row in fp32_success if int(row["batch_size"]) == 8]
        best = preferred[0] if preferred else _choose_best_trial(fp32_success or successful)
        updated = replace(
            config,
            model_profile="accurate",
            batch_size=int(best["batch_size"]),
            use_fp16=False,
            num_workers=max(config.num_workers, 4),
            prefetch=True,
            pin_memory=True,
        )
        reason = "use_fp16=True 的测试全部失败，已回退到实际通过测试的 accurate + FP32 配置。"
        return _auto_tune_result(config, updated, tested, True, reason, save=True)

    best = _choose_best_trial(successful)
    recommended_profile = config.model_profile
    reason = model_profile_advice(config, float(best["avg_ms"]))
    if config.model_profile == "ultra" and (int(best["batch_size"]) < 8 or "4070 Ti" in (env.gpu_name or "")):
        recommended_profile = "accurate"
        reason = "ultra 在当前显存和吞吐下不一定优于 accurate；推荐切换 accurate，并使用已通过测试的 batch_size。"

    updated = replace(
        config,
        model_profile=recommended_profile,
        batch_size=int(best["batch_size"]),
        use_fp16=bool(best["use_fp16"]),
        num_workers=max(config.num_workers, 4),
        prefetch=True,
        pin_memory=True,
    )
    return _auto_tune_result(config, updated, tested, True, reason, save=True)


def _candidate_batches(profile: str) -> list[int]:
    if profile == "fast":
        return [8, 16, 32]
    if profile == "balanced":
        return [4, 8, 16, 32]
    if profile == "accurate":
        return [4, 8, 12, 16]
    return [2, 4, 6, 8]


def _run_tune_trial(config: AppConfig, sample_paths: list[Path], batch_size: int, use_fp16: bool) -> dict:
    started = time.perf_counter()
    temp_config = replace(config, batch_size=batch_size, use_fp16=use_fp16)
    try:
        from app.core.ai_classifier import LocalAIClassifier

        classifier = LocalAIClassifier(temp_config)
        smoke = classifier.openclip_smoke_test(sample_paths[0])
        results = classifier.classify_paths(sample_paths[: min(8, len(sample_paths))])
        avg_ms = float(classifier.performance_stats.get("avg_ms", 0.0))
        if not avg_ms:
            avg_ms = (time.perf_counter() - started) * 1000.0 / max(1, len(results))
        fp32_fallback = bool(classifier.performance_stats.get("openclip_fp32_fallback") or smoke.get("fp32_fallback"))
        smoke_note = _format_smoke_result(smoke)
        return _trial_row(
            batch_size,
            use_fp16,
            bool(results),
            avg_ms,
            "成功" + ("（FP16失败后回退FP32）" if fp32_fallback else ""),
            "",
            fp32_fallback=fp32_fallback,
            smoke=smoke,
            smoke_note=smoke_note,
        )
    except RuntimeError as exc:
        message = str(exc)
        return _trial_row(batch_size, use_fp16, False, 0.0, "失败", "显存不足" if "out of memory" in message.lower() else message)
    except Exception as exc:
        return _trial_row(batch_size, use_fp16, False, 0.0, "失败", str(exc))


def _trial_row(
    batch_size: int,
    use_fp16: bool,
    success: bool,
    avg_ms: float,
    status: str,
    error: str,
    fp32_fallback: bool = False,
    smoke: dict | None = None,
    smoke_note: str = "",
) -> dict:
    return {
        "batch_size": batch_size,
        "use_fp16": use_fp16,
        "success": success,
        "avg_ms": avg_ms,
        "status": status,
        "error": error,
        "fp32_fallback": fp32_fallback,
        "smoke": smoke or {},
        "smoke_note": smoke_note,
    }


def _choose_best_trial(rows: list[dict]) -> dict:
    return min(rows, key=lambda row: row["avg_ms"] or float("inf"))


def _auto_tune_result(
    original: AppConfig,
    updated: AppConfig,
    tested: list[dict],
    success: bool,
    reason: str,
    save: bool,
) -> dict:
    if save:
        save_config(updated)
    lines = [
        "自动调优报告",
        "",
        f"成功：{'是' if success else '否'}",
        f"推荐配置是否真实通过测试：{'是' if success else '否'}",
        f"失败原因：{'' if success else reason}",
        f"是否写入配置：{'是' if save else '否'}",
        f"推荐档位：{updated.model_profile}",
        f"推荐batch_size：{updated.batch_size}",
        f"use_fp16：{updated.use_fp16}",
        f"是否回退FP32：{'是' if any(row.get('fp32_fallback') for row in tested) or not updated.use_fp16 else '否'}",
        f"num_workers：{updated.num_workers}",
        f"prefetch：{updated.prefetch}",
        f"pin_memory：{updated.pin_memory}",
        "",
        "测试记录：",
    ]
    for row in tested:
        error = f"，失败原因：{row['error']}" if row.get("error") else ""
        smoke = f"，smoke：{row['smoke_note']}" if row.get("smoke_note") else ""
        lines.append(
            f"- batch_size={row['batch_size']} use_fp16={row['use_fp16']}：{row['status']}，"
            f"{row['avg_ms']:.1f} ms/张，真实通过：{'是' if row['success'] else '否'}{error}{smoke}"
        )
    lines.extend(["", "推荐原因：", reason])
    return {
        "ok": success,
        "config": updated,
        "report": "\n".join(lines),
        "tested": tested,
        "recommended_batch_size": updated.batch_size,
        "recommended_profile": updated.model_profile,
        "recommended_use_fp16": updated.use_fp16,
        "fp32_fallback": any(row.get("fp32_fallback") for row in tested) or not updated.use_fp16,
        "failed_reason": "" if success else reason,
        "original_batch_size": original.batch_size,
        "original_use_fp16": original.use_fp16,
    }


def _format_smoke_result(smoke: dict) -> str:
    if not smoke:
        return "OpenCLIP smoke test：未执行"
    if not smoke.get("ok"):
        return f"OpenCLIP smoke test：跳过/失败（{smoke.get('status', '-')}: {smoke.get('reason', '-')}）"
    return (
        "OpenCLIP smoke test：通过"
        f"，image={smoke.get('image_feature_dtype')}"
        f"，text={smoke.get('text_feature_dtype')}"
        f"，similarity={smoke.get('similarity_dtype')}"
        f"，fp32_fallback={'是' if smoke.get('fp32_fallback') else '否'}"
    )


def recommend_batch_size(config: AppConfig, avg_ms: float = 0.0) -> int:
    env = detect_ai_environment(config.use_gpu)
    if not env.cuda_available:
        return min(config.batch_size, 8)
    if config.model_profile == "fast":
        return 32
    if config.model_profile == "balanced":
        return 16 if avg_ms > 0 else max(16, min(config.batch_size, 32))
    if config.model_profile == "accurate":
        return 12 if "4070 Ti" in (env.gpu_name or "") else 8
    return 6


def _bottleneck_text(config: AppConfig, semantic_status: str, detector_status: str, avg_ms: float) -> str:
    if "依赖缺失" in semantic_status or "模型未下载" in semantic_status:
        return "OpenCLIP 未就绪，当前语义分类为 fallback，精度有限。"
    if "依赖缺失" in detector_status or "模型未下载" in detector_status:
        return "YOLO 未就绪，人数和主体完整性判断主要依赖 fallback。"
    if config.model_profile == "ultra" and config.batch_size < 8:
        return "ultra 模型导致 batch 较低，GPU 可能频繁等待数据，建议 accurate 或 balanced。"
    if avg_ms > 800:
        return "单张耗时较高，可能是图片解码、磁盘读取或模型过重。"
    return "当前瓶颈不明显；如果 GPU 占用低，多半是数据读取、缓存命中或 UI 等待造成。"
