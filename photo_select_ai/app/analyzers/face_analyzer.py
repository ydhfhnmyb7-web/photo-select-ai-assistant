from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app.analyzers.analyzer_result import AnalyzerResult


def analyze_face(image_path: Path | str) -> AnalyzerResult:
    try:
        import cv2  # type: ignore
    except Exception:
        return AnalyzerResult(
            module_name="face",
            score=50,
            confidence=0.1,
            tags=["人脸模型未接入"],
            problems=[],
            reason="人脸模型未接入，已跳过；不会影响人工筛片。",
            raw_data={"face_count": 0, "model": "no_cv2"},
        )

    path = Path(image_path)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        gray = np.asarray(image.convert("L"), dtype=np.uint8)

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        return AnalyzerResult(
            module_name="face",
            score=50,
            confidence=0.1,
            tags=["人脸模型未接入"],
            problems=[],
            reason="OpenCV Haar 人脸模型不可用，已跳过。",
            raw_data={"face_count": 0, "model": "haar_missing"},
        )

    detector = cv2.CascadeClassifier(str(cascade_path))
    faces = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(32, 32))
    boxes = [tuple(int(v) for v in face) for face in faces]
    if not boxes:
        return AnalyzerResult(
            module_name="face",
            score=58,
            confidence=0.35,
            tags=["未检测到人脸"],
            problems=[],
            reason="未检测到明确人脸；若这是风景、环境或静物照片可忽略，否则建议人工复核。",
            raw_data={"face_count": 0, "image_size": [width, height]},
        )

    boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
    main = boxes[0]
    x, y, w, h = main
    face_area_ratio = (w * h) / max(1, width * height)
    face_gray = gray[y : y + h, x : x + w]
    clarity = _face_clarity(face_gray)
    center_x = (x + w / 2) / width
    center_y = (y + h / 2) / height
    center_score = max(0.0, 1.0 - (abs(center_x - 0.5) + abs(center_y - 0.45)))

    score = min(100.0, 55.0 + min(30.0, clarity / 7.0) + center_score * 15.0)
    tags = [f"检测到{len(boxes)}张人脸"]
    problems: list[str] = []
    if clarity > 90:
        tags.append("主人脸清晰")
    elif clarity > 45:
        tags.append("主人脸略软")
        problems.append("技术可修")
    else:
        tags.append("主人脸清晰度风险")
        problems.append("虚焦")

    if x < width * 0.04 or y < height * 0.04 or x + w > width * 0.96 or y + h > height * 0.96:
        problems.append("裁切风险")
        tags.append("人脸靠近边缘")

    eye_count = _eye_count(cv2, gray, main)
    if eye_count == 0:
        problems.append("闭眼")
        tags.append("可能闭眼或眼部遮挡")

    reason = "检测到较明确人脸。"
    if problems:
        reason = "检测到人脸，但存在" + "、".join(_dedupe(problems)) + "风险，建议人工复核。"
    elif len(boxes) > 1:
        reason = f"检测到 {len(boxes)} 张人脸，主人脸清晰度基本可用。"
    else:
        reason = "检测到一张较清晰人脸，未发现明显闭眼风险。"

    return AnalyzerResult(
        module_name="face",
        score=score,
        confidence=0.72,
        tags=_dedupe(tags),
        problems=_dedupe(problems),
        reason=reason,
        raw_data={
            "face_count": len(boxes),
            "main_face_box": list(main),
            "face_area_ratio": round(face_area_ratio, 4),
            "face_clarity": round(clarity, 2),
            "eye_count": eye_count,
        },
    )


def _face_clarity(face_gray: np.ndarray) -> float:
    try:
        import cv2  # type: ignore

        return float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
    except Exception:
        gy, gx = np.gradient(face_gray.astype(np.float32))
        return float((gx.var() + gy.var()) * 4.0)


def _eye_count(cv2, gray: np.ndarray, face_box: tuple[int, int, int, int]) -> int:
    try:
        eye_path = Path(cv2.data.haarcascades) / "haarcascade_eye.xml"
        if not eye_path.exists():
            return -1
        x, y, w, h = face_box
        roi = gray[y : y + max(1, h // 2), x : x + w]
        eyes = cv2.CascadeClassifier(str(eye_path)).detectMultiScale(roi, scaleFactor=1.1, minNeighbors=4, minSize=(8, 8))
        return int(len(eyes))
    except Exception:
        return -1


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
