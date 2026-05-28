from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app.analyzers.analyzer_result import AnalyzerResult


def analyze_composition(image_path: Path | str, face_result: AnalyzerResult | None = None) -> AnalyzerResult:
    path = Path(image_path)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        gray = np.asarray(image.convert("L"), dtype=np.float32)

    aspect = width / max(1, height)
    edge_density = _edge_density(gray)
    brightness = float(gray.mean())
    problems: list[str] = []
    tags: list[str] = []
    score = 72.0
    confidence = 0.46

    if 0.72 <= aspect <= 1.55:
        tags.append("常规画幅")
    else:
        tags.append("特殊画幅")
        score -= 5

    face_box = None
    if face_result:
        face_box = (face_result.raw_data or {}).get("main_face_box")
    if face_box:
        x, y, w, h = [float(v) for v in face_box]
        center_x = (x + w / 2) / width
        center_y = (y + h / 2) / height
        if center_x < 0.18 or center_x > 0.82 or center_y < 0.12 or center_y > 0.82:
            problems.append("裁切风险")
            tags.append("主体位置靠边")
            score -= 12
        else:
            tags.append("主体位置基本可用")
            score += 8
        confidence = 0.62
    else:
        tags.append("主体位置需人工确认")

    if edge_density > 42:
        problems.append("背景杂乱")
        tags.append("背景复杂度偏高")
        score -= 8
    elif edge_density < 14:
        tags.append("背景较简洁")

    if brightness < 80 or brightness > 200:
        problems.append("曝光问题")
        score -= 6

    if not problems:
        reason = "构图基础规则未发现明显风险，主体与背景关系仍建议人工结合内容判断。"
    else:
        reason = "构图存在" + "、".join(_dedupe(problems)) + "提示，优先作为可修可裁切风险，不直接判废。"

    return AnalyzerResult(
        module_name="composition",
        score=max(0, min(100, score)),
        confidence=confidence,
        tags=_dedupe(tags),
        problems=_dedupe(problems),
        reason=reason,
        raw_data={"aspect_ratio": round(aspect, 3), "edge_density": round(edge_density, 2), "brightness": round(brightness, 2)},
    )


def _edge_density(gray: np.ndarray) -> float:
    gy, gx = np.gradient(gray)
    magnitude = np.sqrt(gx * gx + gy * gy)
    return float(magnitude.mean())


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
