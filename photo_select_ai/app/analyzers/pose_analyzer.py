from __future__ import annotations

from pathlib import Path

from app.analyzers.analyzer_result import AnalyzerResult


def analyze_pose(image_path: Path | str, face_result: AnalyzerResult | None = None) -> AnalyzerResult:
    problems: list[str] = []
    tags = ["姿态模型未接入"]
    if face_result and "裁切风险" in face_result.problems:
        problems.append("裁切风险")
        tags.append("可能存在裁切风险")
    return AnalyzerResult(
        module_name="pose",
        score=52 if not problems else 46,
        confidence=0.12,
        tags=tags,
        problems=problems,
        reason="姿态分析模型暂未接入，已跳过；当前仅保留来自人脸位置的裁切风险提示。",
        raw_data={"model": "placeholder", "path": str(image_path)},
    )
