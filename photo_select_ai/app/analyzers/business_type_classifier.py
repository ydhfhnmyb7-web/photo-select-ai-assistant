from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from app.analyzers.analyzer_result import AnalyzerResult
from app.core.review_config import get_subtypes


def classify_business_type(image_path: Path | str, face_result: AnalyzerResult | None = None) -> AnalyzerResult:
    path = Path(image_path)
    face_raw = face_result.raw_data or {} if face_result else {}
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        evidence = _extract_visual_evidence(image, face_raw)

    photo_type, subtype, confidence, reason, tags, problems = _classify_from_evidence(evidence)
    if confidence < 0.65:
        candidate_type = photo_type
        candidate_subtype = subtype
        weak_reason = reason
        photo_type = "无法判断"
        subtype = "未细分"
        problems = _dedupe([*problems, "建议人工确认"])
        reason = "业务类型置信度较低，建议人工确认。"
        if candidate_type and candidate_type != "无法判断":
            reason += f" 较弱倾向：{candidate_type} / {candidate_subtype}。{weak_reason}"
        evidence["candidate_photo_type"] = candidate_type
        evidence["candidate_subtype"] = candidate_subtype

    tags = _dedupe([*tags, photo_type])
    return AnalyzerResult(
        module_name="business_type",
        score=confidence * 100,
        confidence=confidence,
        tags=tags,
        problems=problems,
        reason=reason,
        raw_data={**evidence, "photo_type": photo_type, "subtype": subtype, "image_size": [width, height]},
    )


def _extract_visual_evidence(image: Image.Image, face_raw: dict[str, Any]) -> dict[str, Any]:
    width, height = image.size
    rgb = np.asarray(image.resize((96, 96)), dtype=np.float32)
    dominant = tuple(int(v) for v in image.resize((1, 1)).getpixel((0, 0)))
    color_std = float(rgb.std())
    white_mask = (rgb[:, :, 0] > 185) & (rgb[:, :, 1] > 185) & (rgb[:, :, 2] > 185)
    white_ratio = float(white_mask.mean())
    lower_white_ratio = float(white_mask[48:, :].mean())
    center_white_ratio = float(white_mask[24:88, 24:72].mean())
    red, green, blue = dominant
    green_ratio = float(((rgb[:, :, 1] > rgb[:, :, 0] + 18) & (rgb[:, :, 1] > rgb[:, :, 2] + 12)).mean())
    blue_ratio = float(((rgb[:, :, 2] > rgb[:, :, 0] + 18) & (rgb[:, :, 2] > rgb[:, :, 1] + 8)).mean())
    face_count = _safe_int(face_raw.get("face_count"), 0)
    person_count = _safe_int(face_raw.get("person_count") or face_raw.get("detected_person_count"), face_count)
    main_face_box = face_raw.get("main_face_box") or []
    face_area_ratio = float(face_raw.get("face_area_ratio") or 0.0)
    if not face_area_ratio and len(main_face_box) == 4:
        try:
            _x, _y, w, h = [float(v) for v in main_face_box]
            face_area_ratio = (w * h) / max(1.0, float(width * height))
        except Exception:
            face_area_ratio = 0.0
    full_body_or_half_body = str(face_raw.get("full_body_or_half_body") or _body_scale_from_face(face_area_ratio))
    background_hint = str(face_raw.get("background_hint") or _background_hint(dominant, color_std, green_ratio, blue_ratio))
    clothing_hint = str(face_raw.get("clothing_hint") or _clothing_hint(white_ratio, lower_white_ratio, center_white_ratio, color_std, background_hint))
    scene_hint = str(face_raw.get("scene_hint") or _scene_hint(background_hint, color_std, green_ratio, blue_ratio))
    orientation = "竖图" if height >= width else "横图"
    return {
        "face_count": face_count,
        "person_count": person_count,
        "is_single_person": person_count == 1 or face_count == 1,
        "is_multi_person": person_count >= 3 or face_count >= 3,
        "scene_hint": scene_hint,
        "clothing_hint": clothing_hint,
        "background_hint": background_hint,
        "image_orientation": orientation,
        "full_body_or_half_body": full_body_or_half_body,
        "dominant_color": dominant,
        "color_std": round(color_std, 2),
        "white_ratio": round(white_ratio, 3),
        "lower_white_ratio": round(lower_white_ratio, 3),
        "center_white_ratio": round(center_white_ratio, 3),
    }


def _classify_from_evidence(evidence: dict[str, Any]) -> tuple[str, str, float, str, list[str], list[str]]:
    face_count = _safe_int(evidence.get("face_count"), 0)
    person_count = _safe_int(evidence.get("person_count"), face_count)
    scene_hint = str(evidence.get("scene_hint") or "未知")
    clothing_hint = str(evidence.get("clothing_hint") or "未知")
    background_hint = str(evidence.get("background_hint") or "未知")
    body_scale = str(evidence.get("full_body_or_half_body") or "未知")
    tags = [
        f"人数线索：{person_count or face_count}",
        f"场景线索：{scene_hint}",
        f"服装线索：{clothing_hint}",
        f"背景线索：{background_hint}",
    ]
    problems: list[str] = []

    single_or_double = (0 < person_count <= 2) or (0 < face_count <= 2)
    strong_wedding = clothing_hint == "婚纱" and single_or_double
    studio_wedding = strong_wedding and scene_hint == "棚拍" and body_scale in {"全身", "半身", "未知"}
    if strong_wedding:
        photo_type = "婚纱照"
        subtype = _first_valid_subtype(photo_type, "单人新娘" if person_count <= 1 or face_count <= 1 else "双人互动")
        if studio_wedding:
            subtype = _first_valid_subtype(photo_type, "单人新娘" if person_count <= 1 or face_count <= 1 else "棚内婚纱")
        confidence = 0.78 if studio_wedding else 0.70
        reason = "检测到单人/双人主体与明显白纱、大裙摆或新娘造型线索，优先按婚纱照建议。"
        return photo_type, subtype, confidence, reason, tags + ["婚纱优先"], problems

    if face_count <= 0 and person_count <= 0:
        photo_type = "风景/环境"
        subtype = _first_valid_subtype(photo_type, "环境交代")
        confidence = 0.50 if scene_hint in {"户外", "风景"} else 0.34
        reason = "未检测到明确人物主体，暂按风景/环境或空镜素材低置信推荐。"
        return photo_type, subtype, confidence, reason, tags + ["未检测到人脸"], problems

    if face_count <= 1 or person_count <= 1:
        if _is_id_background(tuple(evidence.get("dominant_color") or (0, 0, 0)), float(evidence.get("color_std") or 0)) and body_scale in {"特写", "半身", "未知"}:
            photo_type = "证件照"
            subtype = _id_subtype(tuple(evidence.get("dominant_color") or (255, 255, 255)))
            confidence = 0.68
            reason = "单人主体且背景较纯，按证件照建议。"
            return photo_type, subtype, confidence, reason, tags, problems
        photo_type = "个人写真"
        subtype = _first_valid_subtype(photo_type, "棚拍写真" if scene_hint == "棚拍" else "半身")
        confidence = 0.58
        reason = "检测到单人主体；单人照片禁止输出全家福、会议合影或团队合影，建议人工确认具体类型。"
        return photo_type, subtype, confidence, reason, tags + ["单人硬约束"], problems

    if face_count == 2 or person_count == 2:
        photo_type = "情侣照"
        subtype = _first_valid_subtype(photo_type, "未细分")
        confidence = 0.56
        reason = "检测到双人主体，不判全家福；可按情侣照/双人合影低置信候选，建议人工确认。"
        return photo_type, subtype, confidence, reason, tags + ["双人硬约束"], problems

    if face_count >= 6 or person_count >= 6:
        photo_type = "会议/活动照" if scene_hint in {"会议", "棚拍", "未知"} else "全家福"
        subtype = _first_valid_subtype(photo_type, "会议合影" if photo_type == "会议/活动照" else "正式合影")
        confidence = 0.66 if photo_type == "会议/活动照" else 0.60
        reason = "检测到多人主体；只有多人数量满足条件时才允许会议/全家福候选。"
        return photo_type, subtype, confidence, reason, tags + ["多人候选"], problems

    if face_count >= 3 or person_count >= 3:
        photo_type = "全家福"
        subtype = _first_valid_subtype(photo_type, "多人家庭")
        confidence = 0.58
        reason = "检测到 3 人以上主体，但关系与场景证据不足，不能强行判定全家福。"
        return photo_type, subtype, confidence, reason, tags + ["多人低置信候选"], problems

    return "无法判断", "未细分", 0.25, "业务类型证据不足，建议人工确认。", tags, ["建议人工确认"]


def _first_valid_subtype(photo_type: str, preferred: str) -> str:
    subtypes = get_subtypes(photo_type)
    if preferred in subtypes:
        return preferred
    return subtypes[0] if subtypes else "未细分"


def _is_id_background(dominant: tuple[int, int, int], color_std: float) -> bool:
    red, green, blue = dominant
    is_white = red > 185 and green > 185 and blue > 185
    is_blue = blue > 135 and blue > red + 25
    is_red = red > 145 and red > blue + 25
    return color_std < 72 and (is_white or is_blue or is_red)


def _background_hint(dominant: tuple[int, int, int], color_std: float, green_ratio: float, blue_ratio: float) -> str:
    red, green, blue = dominant
    if color_std < 58 and red > 185 and green > 185 and blue > 185:
        return "白底"
    if color_std < 70 and blue > 135 and blue > red + 25:
        return "蓝底"
    if color_std < 70 and red > 145 and red > blue + 25:
        return "红底"
    if green_ratio > 0.22 or blue_ratio > 0.28:
        return "户外"
    if color_std < 95:
        return "棚拍背景"
    return "未知"


def _clothing_hint(white_ratio: float, lower_white_ratio: float, center_white_ratio: float, color_std: float, background_hint: str) -> str:
    if background_hint in {"白底", "蓝底", "红底"} and color_std < 62:
        return "证件照服装"
    if (lower_white_ratio > 0.30 or center_white_ratio > 0.34) and white_ratio > 0.16 and color_std >= 42:
        return "婚纱"
    if white_ratio > 0.38 and background_hint == "棚拍背景" and color_std >= 35:
        return "婚纱"
    return "未知"


def _scene_hint(background_hint: str, color_std: float, green_ratio: float, blue_ratio: float) -> str:
    if background_hint in {"白底", "蓝底", "红底", "棚拍背景"}:
        return "棚拍"
    if green_ratio > 0.22 or blue_ratio > 0.28:
        return "户外"
    if color_std > 112:
        return "会议"
    return "未知"


def _body_scale_from_face(face_area_ratio: float) -> str:
    if face_area_ratio <= 0:
        return "未知"
    if face_area_ratio < 0.035:
        return "全身"
    if face_area_ratio < 0.11:
        return "半身"
    return "特写"


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _id_subtype(dominant: tuple[int, int, int]) -> str:
    red, green, blue = dominant
    if blue > 135 and blue > red + 25:
        return "蓝底"
    if red > 145 and red > blue + 25:
        return "红底"
    return "白底"


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
