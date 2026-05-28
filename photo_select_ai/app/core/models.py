from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


AUTO_STRONG = "01_Strong_Select_强烈推荐"
AUTO_DELIVERABLE = "02_Deliverable_可交付"
AUTO_BACKUP = "03_Backup_备选"
AUTO_DUPLICATE = "04_Duplicate_重复"
AUTO_REJECT = "05_Reject_废片"

MANUAL_CATEGORIES = ["", "精修", "可交付", "备选", "重复", "废片"]

MANUAL_TO_OUTPUT_CATEGORY = {
    "精修": AUTO_STRONG,
    "可交付": AUTO_DELIVERABLE,
    "备选": AUTO_BACKUP,
    "重复": AUTO_DUPLICATE,
    "废片": AUTO_REJECT,
}


def clamp_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def category_from_score(total_score: float, is_duplicate: bool) -> str:
    if is_duplicate:
        return AUTO_DUPLICATE
    if total_score >= 90:
        return AUTO_STRONG
    if total_score >= 75:
        return AUTO_DELIVERABLE
    if total_score >= 60:
        return AUTO_BACKUP
    return AUTO_REJECT


@dataclass
class PhotoAnalysis:
    path: Path
    filename: str
    thumbnail_path: Optional[Path]
    width: int
    height: int
    file_size_mb: float
    taken_at: str
    orientation: str
    clarity_score: float
    exposure_score: float
    contrast_score: float
    resolution_score: float
    perceptual_hash: str
    similarity_bonus: float = 100.0
    total_score: float = 0.0
    auto_category: str = AUTO_REJECT
    manual_category: str = ""
    notes: str = ""
    similar_group_id: Optional[int] = None
    recommended_in_group: bool = True
    is_duplicate: bool = False

    @property
    def effective_category(self) -> str:
        return MANUAL_TO_OUTPUT_CATEGORY.get(self.manual_category, self.auto_category)

    @property
    def resolution_text(self) -> str:
        megapixels = self.width * self.height / 1_000_000
        return f"{self.width} x {self.height} ({megapixels:.1f} MP)"

    def recompute_total_and_category(self) -> None:
        self.similarity_bonus = 0.0 if self.is_duplicate else 100.0
        self.total_score = round(
            self.clarity_score * 0.35
            + self.exposure_score * 0.25
            + self.contrast_score * 0.15
            + self.resolution_score * 0.10
            + self.similarity_bonus * 0.15,
            2,
        )
        self.auto_category = category_from_score(self.total_score, self.is_duplicate)

    def to_db_row(self) -> dict:
        return {
            "path": str(self.path),
            "filename": self.filename,
            "thumbnail_path": str(self.thumbnail_path) if self.thumbnail_path else "",
            "width": self.width,
            "height": self.height,
            "file_size_mb": self.file_size_mb,
            "taken_at": self.taken_at,
            "orientation": self.orientation,
            "clarity_score": self.clarity_score,
            "exposure_score": self.exposure_score,
            "contrast_score": self.contrast_score,
            "resolution_score": self.resolution_score,
            "perceptual_hash": self.perceptual_hash,
            "similarity_bonus": self.similarity_bonus,
            "total_score": self.total_score,
            "auto_category": self.auto_category,
            "manual_category": self.manual_category,
            "notes": self.notes,
            "similar_group_id": self.similar_group_id,
            "recommended_in_group": int(self.recommended_in_group),
            "is_duplicate": int(self.is_duplicate),
        }
