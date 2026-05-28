from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from app.analyzers.analyzer_result import AnalyzerResult


def analyze_basic_quality(image_path: Path | str) -> AnalyzerResult:
    path = Path(image_path)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        gray = np.asarray(image.convert("L"), dtype=np.float32)

    sharpness = _laplacian_variance(gray)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    noise = _noise_estimate(path)
    megapixels = width * height / 1_000_000

    sharpness_score = min(100.0, sharpness / 2.0)
    exposure_score = max(0.0, 100.0 - abs(brightness - 128.0) * 100.0 / 128.0)
    contrast_score = min(100.0, contrast * 2.2)
    resolution_score = min(100.0, megapixels / 12.0 * 100.0)
    noise_score = max(0.0, 100.0 - noise * 1.5)
    score = (
        sharpness_score * 0.34
        + exposure_score * 0.24
        + contrast_score * 0.18
        + resolution_score * 0.14
        + noise_score * 0.10
    )

    tags: list[str] = []
    problems: list[str] = []
    if sharpness < 55:
        tags.append("清晰度风险")
        problems.append("虚焦")
    elif sharpness < 110:
        tags.append("轻微模糊")
        problems.append("技术可修")
    else:
        tags.append("清晰度正常")

    if brightness < 72:
        tags.append("严重欠曝")
        problems.append("曝光问题")
    elif brightness < 95:
        tags.append("轻微欠曝")
        problems.append("曝光问题")
        problems.append("技术可修")
    elif brightness > 220:
        tags.append("严重过曝")
        problems.append("曝光问题")
    elif brightness > 190:
        tags.append("轻微过曝")
        problems.append("曝光问题")
        problems.append("技术可修")
    else:
        tags.append("曝光基本正常")

    if contrast < 24:
        tags.append("对比度偏低")
        problems.append("色彩问题")
    else:
        tags.append("对比度可用")

    if noise > 28:
        tags.append("噪点偏高")
        problems.append("技术可修")

    reason_parts = []
    if "虚焦" in problems:
        reason_parts.append("清晰度不足")
    else:
        reason_parts.append("主体清晰度基本可用")
    if "曝光问题" in problems:
        reason_parts.append("曝光存在可复核问题")
    else:
        reason_parts.append("曝光较稳定")
    if "色彩问题" in problems:
        reason_parts.append("画面对比度偏弱")
    reason = "，".join(reason_parts) + "。"

    return AnalyzerResult(
        module_name="basic_quality",
        score=score,
        confidence=0.92,
        tags=_dedupe(tags),
        problems=_dedupe(problems),
        reason=reason,
        raw_data={
            "sharpness": round(sharpness, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "noise": round(noise, 2),
            "width": width,
            "height": height,
            "megapixels": round(megapixels, 2),
        },
    )


def _laplacian_variance(gray: np.ndarray) -> float:
    try:
        import cv2  # type: ignore

        return float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())
    except Exception:
        gy, gx = np.gradient(gray)
        return float((gx.var() + gy.var()) * 4.0)


def _noise_estimate(path: Path) -> float:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("L")
            small = image.resize((max(1, image.width // 2), max(1, image.height // 2)))
            smooth = small.filter(ImageFilter.GaussianBlur(radius=1.2))
            residual = np.asarray(small, dtype=np.float32) - np.asarray(smooth, dtype=np.float32)
            return float(np.std(residual))
    except Exception:
        return 0.0


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
