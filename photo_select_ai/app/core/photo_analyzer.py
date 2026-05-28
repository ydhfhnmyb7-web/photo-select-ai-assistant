from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from app.core.models import PhotoAnalysis, clamp_score
from app.core.similarity import assign_similarity_groups
from app.utils.exif_utils import read_taken_at
from app.utils.image_utils import (
    average_hash,
    describe_orientation,
    generate_thumbnail,
    read_bgr_image,
    read_image_size,
    scan_image_files,
    scan_raw_candidates,
)


ProgressCallback = Callable[[int, int, str], None]
LogCallback = Callable[[str], None]


class PhotoAnalyzer:
    def __init__(self, source_folder: Path):
        self.source_folder = Path(source_folder)
        self.cache_dir = self.source_folder / ".photoselect_cache"
        self.thumbnail_dir = self.cache_dir / "thumbnails"

    def analyze_folder(
        self,
        progress_callback: Optional[ProgressCallback] = None,
        log_callback: Optional[LogCallback] = None,
    ) -> list[PhotoAnalysis]:
        image_files = scan_image_files(self.source_folder)
        raw_candidates = scan_raw_candidates(self.source_folder)

        if log_callback:
            log_callback(f"发现 {len(image_files)} 张可分析照片。")
            if raw_candidates:
                log_callback(f"发现 {len(raw_candidates)} 个 RAW 候选文件，MVP 暂不解码，已预留扩展入口。")

        analyses: list[PhotoAnalysis] = []
        total = len(image_files)

        for index, image_path in enumerate(image_files, start=1):
            try:
                analyses.append(self.analyze_photo(image_path))
                if log_callback:
                    log_callback(f"已分析：{image_path.name}")
            except Exception as exc:
                if log_callback:
                    log_callback(f"跳过 {image_path.name}：{exc}")
            finally:
                if progress_callback:
                    progress_callback(index, total, str(image_path))

        if analyses:
            if log_callback:
                log_callback("正在检测相似照片组。")
            assign_similarity_groups(analyses)
            for analysis in analyses:
                analysis.recompute_total_and_category()

        return analyses

    def analyze_photo(self, image_path: Path) -> PhotoAnalysis:
        image = read_bgr_image(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        width, height = read_image_size(image_path)

        clarity_score = _calculate_clarity_score(gray)
        exposure_score = _calculate_exposure_score(gray)
        contrast_score = _calculate_contrast_score(gray)
        resolution_score = _calculate_resolution_score(width, height)

        thumbnail_path = generate_thumbnail(image_path, self.thumbnail_dir)
        file_size_mb = round(image_path.stat().st_size / (1024 * 1024), 2)

        analysis = PhotoAnalysis(
            path=image_path,
            filename=image_path.name,
            thumbnail_path=thumbnail_path,
            width=width,
            height=height,
            file_size_mb=file_size_mb,
            taken_at=read_taken_at(image_path),
            orientation=describe_orientation(width, height),
            clarity_score=round(clarity_score, 2),
            exposure_score=round(exposure_score, 2),
            contrast_score=round(contrast_score, 2),
            resolution_score=round(resolution_score, 2),
            perceptual_hash=average_hash(image_path),
        )
        analysis.recompute_total_and_category()
        return analysis


def _calculate_clarity_score(gray: np.ndarray) -> float:
    # Laplacian variance is widely used as a blur proxy. log1p keeps very sharp files from dominating the scale.
    laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    score = math.log1p(laplacian_variance) / math.log1p(1200.0) * 100.0
    return clamp_score(score)


def _calculate_exposure_score(gray: np.ndarray) -> float:
    mean_brightness = float(gray.mean())
    mean_score = 100.0 - abs(mean_brightness - 128.0) / 128.0 * 100.0

    dark_clip_ratio = float(np.mean(gray <= 5))
    bright_clip_ratio = float(np.mean(gray >= 250))
    clip_penalty = min(60.0, (dark_clip_ratio + bright_clip_ratio) * 300.0)

    return clamp_score(mean_score - clip_penalty)


def _calculate_contrast_score(gray: np.ndarray) -> float:
    stddev = float(gray.std())
    score = (stddev - 15.0) / (75.0 - 15.0) * 100.0
    return clamp_score(score)


def _calculate_resolution_score(width: int, height: int) -> float:
    megapixels = width * height / 1_000_000
    if megapixels >= 20:
        return 100.0
    if megapixels >= 12:
        return 85.0 + (megapixels - 12.0) / 8.0 * 15.0
    if megapixels >= 6:
        return 65.0 + (megapixels - 6.0) / 6.0 * 20.0
    if megapixels >= 2:
        return 40.0 + (megapixels - 2.0) / 4.0 * 25.0
    return clamp_score(megapixels / 2.0 * 40.0)
