from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AnalyzerResult:
    module_name: str
    score: float = 0.0
    confidence: float = 0.0
    tags: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    reason: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["score"] = round(float(self.score), 2)
        data["confidence"] = round(float(self.confidence), 3)
        return data


def error_result(module_name: str, message: str) -> AnalyzerResult:
    return AnalyzerResult(
        module_name=module_name,
        score=0.0,
        confidence=0.0,
        tags=["模块异常"],
        problems=[message],
        reason=f"{module_name} 分析失败，已跳过。",
        raw_data={"error": message},
    )
