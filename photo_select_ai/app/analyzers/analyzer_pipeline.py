from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from app.analyzers.analyzer_result import AnalyzerResult, error_result
from app.analyzers.basic_quality_analyzer import analyze_basic_quality
from app.analyzers.business_type_classifier import classify_business_type
from app.analyzers.composition_analyzer import analyze_composition
from app.analyzers.expression_analyzer import analyze_expression
from app.analyzers.face_analyzer import analyze_face
from app.analyzers.pose_analyzer import analyze_pose
from app.analyzers.review_decision_engine import make_review_decision
from app.core.config import PROJECT_ROOT
from app.core.review_models import REVIEW_STATUS_AI_REVIEWED, REVIEW_STATUS_HUMAN_CONFIRMED


ANALYZER_LOG_PATH = PROJECT_ROOT / "logs" / "analyzer.log"
CURRENT_AI_SUGGESTION_VERSION = "business_classifier_v2"
AI_ANALYZER_NAME = "analyzer_pipeline"
STALE_AI_SUGGESTION_MESSAGE = "AI建议可能来自旧版本，建议重新分析。"
MISMATCHED_AI_SUGGESTION_MESSAGE = "AI建议与当前照片不匹配，建议重新分析。"


def analyze_photo(image_path: Path | str, photo_record: Any = None) -> dict:
    path = Path(image_path)
    started = time.perf_counter()
    _write_analyzer_log(f"分析开始：{path}")
    results: dict[str, AnalyzerResult] = {}

    results["basic_quality"] = _safe_run("basic_quality", lambda: analyze_basic_quality(path))
    results["face"] = _safe_run("face", lambda: analyze_face(path))
    results["pose"] = _safe_run("pose", lambda: analyze_pose(path, results.get("face")))
    results["expression"] = _safe_run("expression", lambda: analyze_expression(path, results.get("face")))
    results["composition"] = _safe_run("composition", lambda: analyze_composition(path, results.get("face")))
    results["business_type"] = _safe_run("business_type", lambda: classify_business_type(path, results.get("face")))
    decision = make_review_decision(results, photo_record)

    elapsed_ms = (time.perf_counter() - started) * 1000
    payload = {
        "version": CURRENT_AI_SUGGESTION_VERSION,
        "analyzer_name": AI_ANALYZER_NAME,
        "image_path": str(path),
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_ms": round(elapsed_ms, 2),
        "basic_quality": results["basic_quality"].to_dict(),
        "face": results["face"].to_dict(),
        "pose": results["pose"].to_dict(),
        "expression": results["expression"].to_dict(),
        "composition": results["composition"].to_dict(),
        "business_type": results["business_type"].to_dict(),
        "decision": decision,
    }
    _write_analyzer_log(
        f"分析完成：{path.name} {elapsed_ms:.1f}ms "
        f"{decision.get('photo_type')} / {decision.get('quality_rating')} / confidence={decision.get('confidence')}"
    )
    return payload


def apply_analysis_to_item(item: Any, analysis: dict) -> None:
    decision = analysis.get("decision") or {}
    item.ai_suggestion = json.dumps(analysis, ensure_ascii=False)
    if getattr(item, "review_status", "") == REVIEW_STATUS_HUMAN_CONFIRMED:
        return

    item.photo_type = str(decision.get("photo_type") or "")
    item.subtype = str(decision.get("subtype") or "")
    item.quality_rating = str(decision.get("quality_rating") or "")
    item.commercial_score = float(decision.get("commercial_score") or 0)
    item.portfolio_score = float(decision.get("portfolio_score") or 0)
    item.delivery_use = list(decision.get("delivery_use") or [])
    item.issue_tags = list(decision.get("issue_tags") or [])
    item.review_status = REVIEW_STATUS_AI_REVIEWED
    item.ai_primary_category = item.photo_type
    item.ai_secondary_category = item.subtype
    item.ai_confidence = float(decision.get("confidence") or 0.0)
    item.ai_quality_tags = list(item.issue_tags)
    item.ai_reason = str(decision.get("reason") or "")
    item.screening_reason = item.ai_reason
    item.final_reason = item.ai_reason
    item.final_recommendation = item.quality_rating


def module_summary_text(analysis: dict | str | None, module_name: str) -> str:
    loaded = _loads_analysis(analysis)
    if not loaded:
        return "未分析"
    if module_name == "decision":
        decision = loaded.get("decision") or {}
        if not decision:
            return "未分析"
        return (
            f"品质等级：{decision.get('quality_rating') or '-'}\n"
            f"商业交付：{decision.get('commercial_score') or '-'}\n"
            f"作品集价值：{decision.get('portfolio_score') or '-'}\n"
            f"建议用途：{' / '.join(decision.get('delivery_use') or []) or '-'}\n"
            f"风险：{'、'.join(decision.get('issue_tags') or []) or '-'}\n"
            f"原因：{decision.get('reason') or '-'}"
        )
    result = loaded.get(module_name) or {}
    if not result:
        return "未分析"
    problems = "、".join(result.get("problems") or []) or "-"
    return (
        f"分数：{result.get('score', '-')}\n"
        f"置信度：{result.get('confidence', '-')}\n"
        f"问题：{problems}\n"
        f"原因：{result.get('reason') or '-'}"
    )


def compact_ai_suggestion_text(analysis: dict | str | None) -> str:
    loaded = _loads_analysis(analysis)
    if not loaded:
        return "暂无 AI 建议，请先运行 AI 预分析或手动筛片。"
    decision = loaded.get("decision") or {}
    if not decision:
        return "暂无 AI 建议，请先运行 AI 预分析或手动筛片。"
    return (
        f"AI推荐照片类型：{decision.get('photo_type') or '-'}\n"
        f"AI推荐细分类别：{decision.get('subtype') or '-'}\n"
        f"AI推荐品质等级：{decision.get('quality_rating') or '-'}\n"
        f"AI推荐交付用途：{'、'.join(decision.get('delivery_use') or []) or '-'}\n"
        f"AI推荐问题标签：{'、'.join(decision.get('issue_tags') or []) or '-'}\n"
        f"置信度：{decision.get('confidence') or '-'}\n"
        f"简短理由：{decision.get('reason') or '-'}"
    )


def ai_suggestion_status(value: dict | str | None, image_path: Path | str | None = None) -> tuple[str, dict, str]:
    """Return (status, payload, message) for an AI suggestion.

    status is one of: missing, invalid, legacy, stale, mismatch, current.
    Only "current" should be displayed as trusted advice in the review UI.
    """
    if not value:
        return "missing", {}, "暂无 AI 建议，请先运行 AI 预分析或手动筛片。"
    loaded = _loads_analysis(value)
    if not loaded:
        return "invalid", {}, STALE_AI_SUGGESTION_MESSAGE

    if image_path is not None:
        expected = Path(image_path)
        suggestion_path = str(loaded.get("image_path") or "")
        if not suggestion_path:
            return "stale", loaded, STALE_AI_SUGGESTION_MESSAGE
        if not _same_path(Path(suggestion_path), expected):
            return "mismatch", loaded, MISMATCHED_AI_SUGGESTION_MESSAGE

    version = str(loaded.get("version") or "")
    if not version:
        return "legacy", loaded, STALE_AI_SUGGESTION_MESSAGE
    if version != CURRENT_AI_SUGGESTION_VERSION:
        return "stale", loaded, STALE_AI_SUGGESTION_MESSAGE
    return "current", loaded, ""


def _safe_run(module_name: str, callback) -> AnalyzerResult:
    started = time.perf_counter()
    try:
        result = callback()
        _write_analyzer_log(f"模块完成：{module_name} {(time.perf_counter() - started) * 1000:.1f}ms")
        return result
    except Exception as exc:
        _write_analyzer_log(f"模块失败：{module_name} {exc}\n{traceback.format_exc()}")
        return error_result(module_name, str(exc))


def _loads_analysis(value: dict | str | None) -> dict:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        loaded = json.loads(value)
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except Exception:
        return left == right


def _write_analyzer_log(message: str) -> None:
    ANALYZER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANALYZER_LOG_PATH.open("a", encoding="utf-8").write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
