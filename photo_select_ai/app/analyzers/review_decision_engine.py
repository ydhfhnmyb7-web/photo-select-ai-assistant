from __future__ import annotations

from typing import Any

from app.analyzers.analyzer_result import AnalyzerResult
from app.core.review_config import get_issue_tags, get_review_preset


def make_review_decision(results: dict[str, AnalyzerResult], photo_record: Any = None) -> dict:
    basic = results.get("basic_quality")
    face = results.get("face")
    pose = results.get("pose")
    expression = results.get("expression")
    composition = results.get("composition")
    business = results.get("business_type")

    basic_score = _score(basic, 60)
    face_score = _score(face, 60)
    pose_score = _score(pose, 55)
    expression_score = _score(expression, 55)
    composition_score = _score(composition, 60)
    business_conf = _confidence(business)

    commercial_score = _clamp(
        basic_score * 0.42
        + face_score * 0.22
        + expression_score * 0.14
        + pose_score * 0.08
        + composition_score * 0.14
    )
    portfolio_score = _clamp(
        basic_score * 0.24
        + composition_score * 0.34
        + expression_score * 0.16
        + business_conf * 100 * 0.10
        + max(0, basic_score - 65) * 0.16
    )

    issue_tags = _collect_issue_tags(results)
    if issue_tags and commercial_score >= 55 and "技术可修" not in issue_tags:
        issue_tags.append("技术可修")

    photo_type = str((business.raw_data or {}).get("photo_type", "")) if business else ""
    subtype = str((business.raw_data or {}).get("subtype", "")) if business else ""
    if not photo_type:
        photo_type = "无法判断"
    if not subtype:
        subtype = "未细分"

    quality_rating = _quality_from_score(commercial_score)
    if "虚焦" in issue_tags and basic_score < 30 and photo_type not in {"风景/环境", "产品/静物"}:
        quality_rating = "X"
    elif "虚焦" in issue_tags and basic_score < 58:
        quality_rating = "C"
    elif "闭眼" in issue_tags and commercial_score < 58:
        quality_rating = "B"
    elif "技术可修" in issue_tags and quality_rating == "X":
        quality_rating = "C"

    delivery_use = _delivery_from_scores(quality_rating, commercial_score, portfolio_score, issue_tags)

    confidence = _clamp01(
        (_confidence(basic) * 0.30)
        + (_confidence(face) * 0.18)
        + (_confidence(composition) * 0.16)
        + (_confidence(business) * 0.18)
        + (_confidence(expression) * 0.10)
        + (_confidence(pose) * 0.08)
    )
    preset = get_review_preset(photo_type)
    reason = _decision_reason(quality_rating, commercial_score, portfolio_score, issue_tags, results)

    return {
        "photo_type": photo_type,
        "subtype": subtype,
        "quality_rating": quality_rating,
        "commercial_score": round(commercial_score, 1),
        "portfolio_score": round(portfolio_score, 1),
        "delivery_use": delivery_use,
        "issue_tags": issue_tags,
        "confidence": round(confidence, 3),
        "reason": reason,
        "review_preset": preset,
    }


def _collect_issue_tags(results: dict[str, AnalyzerResult]) -> list[str]:
    valid = set(get_issue_tags())
    tags: list[str] = []
    for result in results.values():
        for problem in result.problems:
            if problem in valid and problem not in tags:
                tags.append(problem)
    return tags


def _delivery_from_scores(quality: str, commercial: float, portfolio: float, issues: list[str]) -> list[str]:
    uses: list[str] = []
    if quality == "S":
        uses.extend(["精修候选", "客户可选"])
    elif quality == "A":
        uses.append("客户可选")
    elif quality == "B":
        uses.append("客户可选" if commercial >= 70 else "仅留档")
    elif quality == "C":
        uses.extend(["修图练习", "仅留档"])
    elif quality == "X":
        uses.append("不导出")
    if portfolio >= 78 and quality in {"S", "A", "B"}:
        uses.append("作品集候选")
    if "技术可修" in issues and "修图练习" not in uses:
        uses.append("修图练习")
    return _dedupe(uses)


def _decision_reason(quality: str, commercial: float, portfolio: float, issues: list[str], results: dict[str, AnalyzerResult]) -> str:
    quality_text = {"S": "强烈推荐", "A": "可交付", "B": "备选", "C": "留档/练习", "X": "废片"}.get(quality, quality)
    parts = [f"综合建议为 {quality} {quality_text}，商业交付分 {commercial:.0f}，作品集分 {portfolio:.0f}"]
    if issues:
        parts.append("主要风险：" + "、".join(issues))
    basic = results.get("basic_quality")
    if basic and basic.reason:
        parts.append(basic.reason.rstrip("。"))
    composition = results.get("composition")
    if composition and composition.problems:
        parts.append("构图问题优先视为可修或可裁切风险，不直接判废")
    return "；".join(parts) + "。"


def _quality_from_score(score: float) -> str:
    if score >= 90:
        return "S"
    if score >= 75:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    return "X"


def _score(result: AnalyzerResult | None, default: float) -> float:
    return float(result.score) if result else default


def _confidence(result: AnalyzerResult | None) -> float:
    return float(result.confidence) if result else 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
