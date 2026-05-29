from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.core.categories import (
    MANUAL_PENDING,
    MANUAL_REJECTED,
    MANUAL_SELECTED,
    PRIMARY_REVIEW,
    category_path,
    manual_category_path,
)
from app.core.review_models import (
    REVIEW_STATUS_UNREVIEWED,
    decode_multi_select,
    encode_multi_select,
)


LABEL_SELECTED = MANUAL_SELECTED
LABEL_PENDING = MANUAL_PENDING
LABEL_REJECTED = MANUAL_REJECTED
LABELS = [LABEL_SELECTED, LABEL_PENDING, LABEL_REJECTED]


def encode_tags(tags: list[str]) -> str:
    return json.dumps(tags, ensure_ascii=False)


def decode_tags(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(loaded, list):
        return []
    return [str(item) for item in loaded]


def encode_review_list(values: list[str]) -> str:
    return encode_multi_select(values)


def decode_review_list(value: str | None) -> list[str]:
    return decode_multi_select(value)


@dataclass
class PhotoItem:
    path: Path
    filename: str
    thumbnail_path: Path | None
    width: int
    height: int
    file_size_mb: float
    taken_at: str = ""
    manual_category: str = MANUAL_PENDING
    manual_override: bool = False
    user_note: str = ""
    ai_primary_category: str = ""
    ai_secondary_category: str = ""
    ai_quality_tags: list[str] = field(default_factory=list)
    ai_confidence: float = 0.0
    ai_source: str = ""
    people_label: str = ""
    people_count: int = 0
    shooting_type: str = ""
    appearance_tags: list[str] = field(default_factory=list)
    final_category: str = ""
    similar_group_id: str = ""
    similar_group_size: int = 0
    recommended_keep: bool = False
    user_label: str = ""
    user_label_time: str = ""
    user_preference_score: float = 0.0
    user_preference_reason: str = ""
    preference_model_version: str = ""
    absolute_quality_score: float = 0.0
    relative_quality_score: float = 0.0
    group_rank: int = 0
    group_size: int = 0
    scene_group_id: str = ""
    outfit_group_id: str = ""
    pose_sequence_id: str = ""
    recommended_in_group: bool = False
    ai_reason: str = ""
    preference_reason: str = ""
    final_reason: str = ""
    embedding_path: str = ""
    embedding_cached: bool = False
    ai_runtime_ms: float = 0.0
    ai_device: str = ""
    ai_batch_size: int = 0
    semantic_score: float = 0.0
    semantic_confidence: float = 0.0
    top3_semantic_matches: list[str] = field(default_factory=list)
    final_pick_score: float = 0.0
    detected_person_count: int = 0
    main_subject_bbox: str = ""
    subject_area_ratio: float = 0.0
    subject_center_score: float = 0.0
    edge_cutoff_risk: float = 0.0
    group_photo_score: float = 0.0
    detection_confidence: float = 0.0
    embedding_model: str = ""
    embedding_created_at: str = ""
    image_feature_hash: str = ""
    screening_reason: str = ""
    style_label: str = ""
    retouch_suggestion: str = ""
    crop_suggestion: str = ""
    portfolio_suggestion: str = ""
    delivery_suggestion: str = ""
    final_recommendation: str = ""
    aesthetic_like_similarity: float = 0.0
    aesthetic_dislike_similarity: float = 0.0
    similar_reference_count: int = 0
    photo_type: str = ""
    subtype: str = ""
    quality_rating: str = ""
    delivery_use: list[str] = field(default_factory=list)
    issue_tags: list[str] = field(default_factory=list)
    commercial_score: float = 0.0
    portfolio_score: float = 0.0
    ai_suggestion: str = ""
    human_decision: str = ""
    review_status: str = REVIEW_STATUS_UNREVIEWED
    best_in_group: bool = False
    review_note: str = ""
    similar_group_rank: int = 0
    similar_group_status: str = ""
    similarity_score: float = 0.0
    similar_group_note: str = ""
    ai_recommended_best: bool = False
    ai_similarity_reason: str = ""
    human_group_decision: str = ""
    similarity_hash: str = ""
    similarity_hash_mtime: float = 0.0
    auto_group_id: str = ""
    auto_group_confidence: float = 0.0
    auto_group_reason: str = ""
    embedding_model_name: str = ""
    embedding_cached_at: str = ""
    grouping_method: str = ""
    embedding_cache_key: str = ""
    embedding_dim: int = 0
    embedding_version: str = ""
    auto_group_rank: int = 0
    auto_group_size: int = 0

    @property
    def label(self) -> str:
        return self.manual_category

    @label.setter
    def label(self, value: str) -> None:
        self.manual_category = value

    @property
    def notes(self) -> str:
        return self.user_note

    @notes.setter
    def notes(self, value: str) -> None:
        self.user_note = value

    @property
    def has_ai_result(self) -> bool:
        return bool(self.ai_primary_category)

    @property
    def ai_confidence_text(self) -> str:
        return f"{self.ai_confidence:.2f}" if self.has_ai_result else "-"

    @property
    def ai_category_path(self) -> str:
        if not self.ai_primary_category:
            return ""
        return category_path(self.ai_primary_category, self.ai_secondary_category)

    @property
    def export_category(self) -> str:
        return self.final_category or self.compute_final_category()

    @property
    def resolution_text(self) -> str:
        return f"{self.width} x {self.height}"

    def compute_final_category(self, confidence_threshold: float = 0.65) -> str:
        if self.manual_override:
            self.final_category = manual_category_path(self.manual_category)
            return self.final_category
        if self.ai_primary_category:
            if self.ai_confidence < confidence_threshold:
                self.final_category = PRIMARY_REVIEW
            else:
                self.final_category = self.ai_category_path
        else:
            self.final_category = manual_category_path(self.manual_category)
        return self.final_category

    def to_db_row(self) -> dict:
        return {
            "path": str(self.path),
            "filename": self.filename,
            "thumbnail_path": str(self.thumbnail_path) if self.thumbnail_path else "",
            "width": self.width,
            "height": self.height,
            "file_size_mb": self.file_size_mb,
            "taken_at": self.taken_at,
            "manual_category": self.manual_category,
            "manual_override": int(self.manual_override),
            "final_category": self.final_category,
            "user_note": self.user_note,
            "ai_primary_category": self.ai_primary_category,
            "ai_secondary_category": self.ai_secondary_category,
            "ai_quality_tags": encode_tags(self.ai_quality_tags),
            "ai_confidence": self.ai_confidence,
            "ai_source": self.ai_source,
            "people_label": self.people_label,
            "people_count": self.people_count,
            "shooting_type": self.shooting_type,
            "appearance_tags": encode_tags(self.appearance_tags),
            "similar_group_id": self.similar_group_id,
            "similar_group_size": self.similar_group_size,
            "recommended_keep": int(self.recommended_keep),
            "user_label": self.user_label,
            "user_label_time": self.user_label_time,
            "user_preference_score": self.user_preference_score,
            "user_preference_reason": self.user_preference_reason,
            "preference_model_version": self.preference_model_version,
            "absolute_quality_score": self.absolute_quality_score,
            "relative_quality_score": self.relative_quality_score,
            "group_rank": self.group_rank,
            "group_size": self.group_size,
            "scene_group_id": self.scene_group_id,
            "outfit_group_id": self.outfit_group_id,
            "pose_sequence_id": self.pose_sequence_id,
            "recommended_in_group": int(self.recommended_in_group),
            "ai_reason": self.ai_reason,
            "preference_reason": self.preference_reason,
            "final_reason": self.final_reason,
            "embedding_path": self.embedding_path,
            "embedding_cached": int(self.embedding_cached),
            "ai_runtime_ms": self.ai_runtime_ms,
            "ai_device": self.ai_device,
            "ai_batch_size": self.ai_batch_size,
            "semantic_score": self.semantic_score,
            "semantic_confidence": self.semantic_confidence,
            "top3_semantic_matches": encode_tags(self.top3_semantic_matches),
            "final_pick_score": self.final_pick_score,
            "detected_person_count": self.detected_person_count,
            "main_subject_bbox": self.main_subject_bbox,
            "subject_area_ratio": self.subject_area_ratio,
            "subject_center_score": self.subject_center_score,
            "edge_cutoff_risk": self.edge_cutoff_risk,
            "group_photo_score": self.group_photo_score,
            "detection_confidence": self.detection_confidence,
            "embedding_model": self.embedding_model,
            "embedding_created_at": self.embedding_created_at,
            "image_feature_hash": self.image_feature_hash,
            "screening_reason": self.screening_reason,
            "style_label": self.style_label,
            "retouch_suggestion": self.retouch_suggestion,
            "crop_suggestion": self.crop_suggestion,
            "portfolio_suggestion": self.portfolio_suggestion,
            "delivery_suggestion": self.delivery_suggestion,
            "final_recommendation": self.final_recommendation,
            "aesthetic_like_similarity": self.aesthetic_like_similarity,
            "aesthetic_dislike_similarity": self.aesthetic_dislike_similarity,
            "similar_reference_count": self.similar_reference_count,
            "photo_type": self.photo_type,
            "subtype": self.subtype,
            "quality_rating": self.quality_rating,
            "delivery_use": encode_review_list(self.delivery_use),
            "issue_tags": encode_review_list(self.issue_tags),
            "commercial_score": self.commercial_score,
            "portfolio_score": self.portfolio_score,
            "ai_suggestion": self.ai_suggestion,
            "human_decision": self.human_decision,
            "review_status": self.review_status,
            "best_in_group": int(self.best_in_group),
            "review_note": self.review_note,
            "similar_group_rank": self.similar_group_rank,
            "similar_group_status": self.similar_group_status,
            "similarity_score": self.similarity_score,
            "similar_group_note": self.similar_group_note,
            "ai_recommended_best": int(self.ai_recommended_best),
            "ai_similarity_reason": self.ai_similarity_reason,
            "human_group_decision": self.human_group_decision,
            "similarity_hash": self.similarity_hash,
            "similarity_hash_mtime": self.similarity_hash_mtime,
            "auto_group_id": self.auto_group_id,
            "auto_group_confidence": self.auto_group_confidence,
            "auto_group_reason": self.auto_group_reason,
            "embedding_model_name": self.embedding_model_name,
            "embedding_cached_at": self.embedding_cached_at,
            "grouping_method": self.grouping_method,
            "embedding_cache_key": self.embedding_cache_key,
            "embedding_dim": self.embedding_dim,
            "embedding_version": self.embedding_version,
            "auto_group_rank": self.auto_group_rank,
            "auto_group_size": self.auto_group_size,
        }
