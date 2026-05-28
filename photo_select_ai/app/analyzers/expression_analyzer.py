from __future__ import annotations

from pathlib import Path

from app.analyzers.analyzer_result import AnalyzerResult


def analyze_expression(image_path: Path | str, face_result: AnalyzerResult | None = None) -> AnalyzerResult:
    face_count = int((face_result.raw_data or {}).get("face_count", 0)) if face_result else 0
    if face_count <= 0:
        return AnalyzerResult(
            module_name="expression",
            score=50,
            confidence=0.12,
            tags=["表情模型未接入"],
            problems=[],
            reason="未检测到明确人脸，表情自然度无法准确判断，建议按照片用途人工复核。",
            raw_data={"model": "placeholder", "face_count": face_count, "path": str(image_path)},
        )

    problems: list[str] = []
    tags = ["表情自然度需人工复核"]
    score = 62
    if face_result and "闭眼" in face_result.problems:
        problems.append("闭眼")
        problems.append("客户可能不喜欢")
        tags.append("可能存在闭眼或半闭眼风险")
        score = 42
    reason = "表情模型暂未接入，当前仅根据人脸检测结果给出谨慎提示。"
    if problems:
        reason = "可能存在闭眼或眼部状态异常风险，建议人工确认，不做绝对情绪判断。"
    return AnalyzerResult(
        module_name="expression",
        score=score,
        confidence=0.22,
        tags=tags,
        problems=problems,
        reason=reason,
        raw_data={"model": "placeholder", "face_count": face_count, "path": str(image_path)},
    )
