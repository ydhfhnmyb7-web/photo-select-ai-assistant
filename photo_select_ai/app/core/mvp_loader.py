from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps

from app.core.app_logging import get_logger
from app.core.categories import LEGACY_MANUAL_CATEGORY_MAP, MANUAL_PENDING
from app.core.config import AppConfig
from app.core.mvp_models import PhotoItem, decode_review_list, decode_tags
from app.core.review_models import REVIEW_STATUS_UNREVIEWED
from app.utils.exif_utils import read_taken_at


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

ProgressCallback = Callable[[str, int, int, str], None]
CancelCallback = Callable[[], bool]


def import_photo_folder(
    folder: Path,
    saved_items: dict[str, dict],
    config: AppConfig,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> tuple[list[PhotoItem], list[str]]:
    logger = get_logger()
    skipped: list[str] = []

    if progress_callback:
        progress_callback("扫描文件", 0, 0, "")
    image_paths = scan_photo_files(folder)
    total = len(image_paths)
    logger.info("导入文件夹路径：%s", folder)
    logger.info("扫描到照片数量：%s", total)

    items: list[PhotoItem] = []
    thumbnail_dir = folder / "cache" / "thumbnails"
    thumbnail_dir.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(image_paths, start=1):
        if cancel_callback and cancel_callback():
            logger.info("导入已取消：%s/%s", index - 1, total)
            break

        if progress_callback:
            progress_callback("生成缩略图", index, total, image_path.name)

        try:
            item = build_photo_item(image_path, thumbnail_dir, saved_items.get(str(image_path), {}), config)
            items.append(item)
        except Exception as exc:
            message = f"{image_path.name}: {exc}"
            skipped.append(message)
            logger.exception("无法读取图片，已跳过：%s", image_path)

    if progress_callback:
        progress_callback("完成", len(items), total, "")
    logger.info("导入照片数量：%s，跳过：%s", len(items), len(skipped))
    return items, skipped


def load_photo_folder(folder: Path, saved_items: dict[str, dict] | None = None) -> list[PhotoItem]:
    from app.core.config import load_config

    items, _skipped = import_photo_folder(folder, saved_items or {}, load_config())
    return items


def scan_photo_files(folder: Path) -> list[Path]:
    excluded_parts = {".photoselect_cache", "cache", "PhotoSelect_Output", "logs"}
    return sorted(
        [
            path
            for path in folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and not any(part in excluded_parts for part in path.parts)
        ],
        key=lambda item: str(item).lower(),
    )


def build_photo_item(image_path: Path, thumbnail_dir: Path, saved: dict, config: AppConfig) -> PhotoItem:
    width, height = read_image_size(image_path)
    thumbnail_path = generate_thumbnail(
        image_path,
        thumbnail_dir,
        max_size=_thumbnail_cache_size(config.thumbnail_size),
        mode=config.thumbnail_mode,
    )

    raw_manual_category = saved.get("manual_category") or saved.get("label") or MANUAL_PENDING
    manual_category = LEGACY_MANUAL_CATEGORY_MAP.get(raw_manual_category, raw_manual_category)
    manual_override = bool(saved.get("manual_override") or raw_manual_category in {"精选", "精修候选", "废片", "可交付", "重复", "作品集候选", "练习片"})

    item = PhotoItem(
        path=image_path,
        filename=image_path.name,
        thumbnail_path=thumbnail_path,
        width=width,
        height=height,
        file_size_mb=round(image_path.stat().st_size / (1024 * 1024), 2),
        taken_at=saved.get("taken_at") or read_taken_at(image_path),
        manual_category=manual_category,
        manual_override=manual_override,
        user_note=saved.get("user_note") or saved.get("notes") or "",
        ai_primary_category=saved.get("ai_primary_category") or "",
        ai_secondary_category=saved.get("ai_secondary_category") or "",
        ai_quality_tags=decode_tags(saved.get("ai_quality_tags")),
        ai_confidence=float(saved.get("ai_confidence") or 0.0),
        ai_source=saved.get("ai_source") or saved.get("ai_device") or "",
        people_label=saved.get("people_label") or saved.get("ai_people_label") or "",
        people_count=int(saved.get("people_count") or saved.get("ai_people_count") or 0),
        shooting_type=saved.get("shooting_type") or saved.get("ai_shoot_type") or "",
        appearance_tags=decode_tags(saved.get("appearance_tags") or saved.get("ai_appearance_tags")),
        final_category=saved.get("final_category") or "",
        similar_group_id=saved.get("similar_group_id") or "",
        similar_group_size=int(saved.get("similar_group_size") or 0),
        recommended_keep=bool(saved.get("recommended_keep") or False),
        user_label=saved.get("user_label") or "",
        user_label_time=saved.get("user_label_time") or "",
        user_preference_score=float(saved.get("user_preference_score") or 0.0),
        user_preference_reason=saved.get("user_preference_reason") or "",
        preference_model_version=saved.get("preference_model_version") or "",
        absolute_quality_score=float(saved.get("absolute_quality_score") or 0.0),
        relative_quality_score=float(saved.get("relative_quality_score") or 0.0),
        group_rank=int(saved.get("group_rank") or 0),
        group_size=int(saved.get("group_size") or saved.get("similar_group_size") or 0),
        scene_group_id=saved.get("scene_group_id") or "",
        outfit_group_id=saved.get("outfit_group_id") or "",
        pose_sequence_id=saved.get("pose_sequence_id") or "",
        recommended_in_group=bool(saved.get("recommended_in_group") or saved.get("recommended_keep") or False),
        ai_reason=saved.get("ai_reason") or "",
        preference_reason=saved.get("preference_reason") or "",
        final_reason=saved.get("final_reason") or "",
        embedding_path=saved.get("embedding_path") or "",
        embedding_cached=bool(saved.get("embedding_cached") or False),
        ai_runtime_ms=float(saved.get("ai_runtime_ms") or 0.0),
        ai_device=saved.get("ai_device") or saved.get("ai_source") or "",
        ai_batch_size=int(saved.get("ai_batch_size") or 0),
        semantic_score=float(saved.get("semantic_score") or 0.0),
        semantic_confidence=float(saved.get("semantic_confidence") or 0.0),
        top3_semantic_matches=decode_tags(saved.get("top3_semantic_matches")),
        final_pick_score=float(saved.get("final_pick_score") or 0.0),
        detected_person_count=int(saved.get("detected_person_count") or 0),
        main_subject_bbox=saved.get("main_subject_bbox") or "",
        subject_area_ratio=float(saved.get("subject_area_ratio") or 0.0),
        subject_center_score=float(saved.get("subject_center_score") or 0.0),
        edge_cutoff_risk=float(saved.get("edge_cutoff_risk") or 0.0),
        group_photo_score=float(saved.get("group_photo_score") or 0.0),
        detection_confidence=float(saved.get("detection_confidence") or 0.0),
        embedding_model=saved.get("embedding_model") or "",
        embedding_created_at=saved.get("embedding_created_at") or "",
        image_feature_hash=saved.get("image_feature_hash") or "",
        screening_reason=saved.get("screening_reason") or "",
        style_label=saved.get("style_label") or "",
        retouch_suggestion=saved.get("retouch_suggestion") or "",
        crop_suggestion=saved.get("crop_suggestion") or "",
        portfolio_suggestion=saved.get("portfolio_suggestion") or "",
        delivery_suggestion=saved.get("delivery_suggestion") or "",
        final_recommendation=saved.get("final_recommendation") or "",
        aesthetic_like_similarity=float(saved.get("aesthetic_like_similarity") or 0.0),
        aesthetic_dislike_similarity=float(saved.get("aesthetic_dislike_similarity") or 0.0),
        similar_reference_count=int(saved.get("similar_reference_count") or 0),
        photo_type=saved.get("photo_type") or "",
        subtype=saved.get("subtype") or "",
        quality_rating=saved.get("quality_rating") or "",
        delivery_use=decode_review_list(saved.get("delivery_use")),
        issue_tags=decode_review_list(saved.get("issue_tags")),
        commercial_score=float(saved.get("commercial_score") or 0.0),
        portfolio_score=float(saved.get("portfolio_score") or 0.0),
        ai_suggestion=saved.get("ai_suggestion") or "",
        human_decision=saved.get("human_decision") or "",
        review_status=saved.get("review_status") or REVIEW_STATUS_UNREVIEWED,
        best_in_group=bool(saved.get("best_in_group") or False),
        review_note=saved.get("review_note") or saved.get("user_note") or "",
        similar_group_rank=int(saved.get("similar_group_rank") or 0),
        similar_group_status=saved.get("similar_group_status") or "",
        similarity_score=float(saved.get("similarity_score") or 0.0),
        similar_group_note=saved.get("similar_group_note") or "",
        ai_recommended_best=bool(saved.get("ai_recommended_best") or False),
        ai_similarity_reason=saved.get("ai_similarity_reason") or "",
        human_group_decision=saved.get("human_group_decision") or "",
        similarity_hash=saved.get("similarity_hash") or "",
        similarity_hash_mtime=float(saved.get("similarity_hash_mtime") or 0.0),
    )
    item.compute_final_category(config.confidence_threshold)
    return item


def read_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        return image.size


def generate_thumbnail(
    image_path: Path,
    thumbnail_dir: Path,
    max_size: tuple[int, int] = (360, 270),
    mode: str = "contain",
) -> Path:
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = thumbnail_dir / _thumbnail_filename(image_path, max_size, mode)

    source_mtime = int(image_path.stat().st_mtime)
    if thumb_path.exists() and int(thumb_path.stat().st_mtime) >= source_mtime:
        return thumb_path

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = _ensure_rgb(image)
        if mode == "cover":
            image = ImageOps.fit(image, max_size, Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        else:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        image.save(thumb_path, "JPEG", quality=86, optimize=True)
    return thumb_path


def preview_pixmap_source_size(image_path: Path, max_edge: int) -> tuple[int, int]:
    width, height = read_image_size(image_path)
    edge = max(width, height)
    if edge <= max_edge:
        return width, height
    scale = max_edge / edge
    return max(1, int(width * scale)), max(1, int(height * scale))


def _thumbnail_cache_size(size: int) -> tuple[int, int]:
    if size <= 130:
        return 240, 180
    if size <= 180:
        return 320, 240
    return 440, 330


def _thumbnail_filename(image_path: Path, max_size: tuple[int, int], mode: str) -> str:
    stat = image_path.stat()
    digest_source = (
        f"{image_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{max_size}:{mode}".encode("utf-8", "ignore")
    )
    digest = hashlib.sha1(digest_source).hexdigest()[:16]
    return f"{image_path.stem}_{digest}.jpg"


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGB", image.size, "white")
        alpha = image.getchannel("A")
        background.paste(image.convert("RGB"), mask=alpha)
        return background
    return image.convert("RGB")
