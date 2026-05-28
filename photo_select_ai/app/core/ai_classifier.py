from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, ImageOps

from app.core.app_logging import get_logger
from app.core.categories import (
    PRIMARY_DUPLICATE,
    PRIMARY_REJECT,
    PRIMARY_REVIEW,
    QUALITY_BLURRY,
    QUALITY_CLEAR,
    QUALITY_CLEAR_REJECT,
    QUALITY_DUPLICATE,
    QUALITY_EDGE_ISSUE,
    QUALITY_KEEP,
    QUALITY_RE_CROP,
    QUALITY_REVIEW,
    QUALITY_SEVERE_BLURRY,
    QUALITY_SEVERE_OVEREXPOSED,
    QUALITY_SEVERE_UNDEREXPOSED,
    QUALITY_SLIGHT_BLURRY,
    QUALITY_SLIGHT_OVEREXPOSED,
    QUALITY_SLIGHT_UNDEREXPOSED,
    QUALITY_UNCERTAIN,
    QUALITY_USABLE_CROP,
    QUALITY_USABLE_RETOUCH,
)
from app.core.config import AppConfig, load_config
from app.core.gpu import detect_ai_environment
from app.core.model_manager import (
    ModelManager,
    auto_batch_size_for_profile,
    get_openclip_spec,
    get_yolo_spec,
    load_embedding_cache,
    save_embedding_cache,
)


ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


CAT_SINGLE = "单人写真"
CAT_DOUBLE = "双人写真"
CAT_MULTI = "多人写真"
CAT_WEDDING = "婚纱照"
CAT_FAMILY = "全家福"
CAT_PARENT_CHILD = "亲子照"
CAT_CHILD = "儿童写真"
CAT_ELDERLY = "中老年写真"
CAT_BUSINESS = "商务会议合影"
CAT_EVENT = "活动纪实"
CAT_ID = "证件/形象照"
CAT_ENV = "风景/环境"


@dataclass
class SemanticResult:
    primary: str = ""
    secondary: str = ""
    confidence: float = 0.0
    score: float = 0.0
    top3: list[str] = field(default_factory=list)
    embedding_path: str = ""
    embedding_model: str = ""
    embedding_cached: bool = False
    embedding_created_at: str = ""
    image_feature_hash: str = ""
    screening_reason: str = ""
    style_label: str = ""
    retouch_suggestion: str = ""
    crop_suggestion: str = ""
    portfolio_suggestion: str = ""
    delivery_suggestion: str = ""
    final_recommendation: str = ""
    source: str = "fallback"


@dataclass
class DetectionResult:
    detected_person_count: int = 0
    main_subject_bbox: str = ""
    subject_area_ratio: float = 0.0
    subject_center_score: float = 0.0
    edge_cutoff_risk: float = 0.0
    group_photo_score: float = 0.0
    detection_confidence: float = 0.0
    source: str = "fallback"


@dataclass
class AIAnalysisResult:
    path: Path
    people_label: str
    people_count: int
    shooting_type: str
    appearance_tags: list[str]
    ai_primary_category: str
    ai_secondary_category: str
    ai_quality_tags: list[str]
    ai_confidence: float
    ai_source: str
    image_hash: str
    similar_group_id: str = ""
    similar_group_size: int = 0
    recommended_keep: bool = False
    absolute_quality_score: float = 0.0
    relative_quality_score: float = 0.0
    group_rank: int = 0
    group_size: int = 0
    recommended_in_group: bool = False
    ai_reason: str = ""
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
    embedding_path: str = ""
    embedding_model: str = ""
    embedding_cached: bool = False
    embedding_created_at: str = ""
    image_feature_hash: str = ""
    screening_reason: str = ""
    style_label: str = ""
    retouch_suggestion: str = ""
    crop_suggestion: str = ""
    portfolio_suggestion: str = ""
    delivery_suggestion: str = ""
    final_recommendation: str = ""


class LocalAIClassifier:
    """Local AI pipeline: OpenCV quality + OpenCLIP semantics + YOLO subject detection."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.manager = ModelManager(self.config)
        self.env = detect_ai_environment(self.config.use_gpu)
        self.logger = get_logger()

        self.cv2 = None
        self.np = None
        self.torch = None
        self.device = "cpu"
        self.device_label = self.env.source
        self.face_detector = None
        self.profile_detector = None
        self.eye_detector = None
        self.semantic_backend = None
        self.detector_backend = None
        self.batch_size = auto_batch_size_for_profile(self.config)
        self.performance_stats = {
            "total": 0,
            "analyzed": 0,
            "total_ms": 0.0,
            "avg_ms": 0.0,
            "device": self.device_label,
            "batch_size": self.batch_size,
            "embedding_cache_hits": 0,
            "embedding_new": 0,
            "fallback": 0,
        }

        if not self.env.cv2_installed:
            raise RuntimeError("AI依赖缺失：请安装 opencv-python 后再运行 AI 自动分析。")

        import cv2
        import numpy as np

        self.cv2 = cv2
        self.np = np

        if self.env.torch_installed:
            try:
                import torch

                self.torch = torch
                if self.env.cuda_available:
                    self.device = torch.device("cuda")
                    self.device_label = f"GPU: CUDA · {torch.cuda.get_device_name(0)}"
                else:
                    self.device = torch.device("cpu")
                    self.device_label = "CPU"
            except Exception:
                self.torch = None
                self.device = "cpu"
                self.device_label = "fallback/no_torch"

        self.face_detector = self._load_haar("haarcascade_frontalface_default.xml")
        self.profile_detector = self._load_haar("haarcascade_profileface.xml")
        self.eye_detector = self._load_haar("haarcascade_eye.xml")
        self.semantic_backend = _OpenClipSemanticBackend(self.config, self.manager, self.device, self.torch)
        self.detector_backend = _YoloDetectorBackend(self.config, self.manager, self.device_label)
        self.performance_stats["device"] = self.device_label

    @property
    def backend_label(self) -> str:
        parts = ["opencv"]
        parts.append("torch" if self.torch is not None else "no_torch")
        parts.append("openclip" if self.semantic_backend and self.semantic_backend.enabled else "semantic_fallback")
        parts.append("yolo" if self.detector_backend and self.detector_backend.enabled else "detector_fallback")
        return "+".join(parts)

    def openclip_smoke_test(self, image_path: Path) -> dict:
        if not self.semantic_backend or not self.semantic_backend.enabled:
            return {
                "ok": False,
                "status": self.semantic_backend.status if self.semantic_backend else "semantic_backend_missing",
                "reason": "OpenCLIP 未加载，当前处于 fallback。",
            }
        return self.semantic_backend.smoke_test(image_path)

    def classify_paths(
        self,
        image_paths: Iterable[Path],
        progress_callback: ProgressCallback | None = None,
        cancel_callback: CancelCallback | None = None,
    ) -> list[AIAnalysisResult]:
        paths = list(image_paths)
        results: list[AIAnalysisResult] = []
        total = len(paths)
        started = time.perf_counter()

        for start in range(0, total, self.batch_size):
            if cancel_callback and cancel_callback():
                break

            batch_paths = paths[start : start + self.batch_size]
            records = []
            batch_started = time.perf_counter()
            for path in batch_paths:
                if cancel_callback and cancel_callback():
                    break
                try:
                    record = self._analyze_image_signals(path)
                except Exception:
                    self.logger.exception("AI分析图片失败：%s", path)
                    continue
                records.append(record)
                if progress_callback:
                    progress_callback(min(start + len(records), total), total, path.name)

            semantic_map = self.semantic_backend.classify_records(records) if self.semantic_backend else {}
            detection_map = self.detector_backend.detect_paths([record["path"] for record in records]) if self.detector_backend else {}
            torch_stats = self._torch_batch_stats([record["small_rgb"] for record in records])

            runtime_ms = (time.perf_counter() - batch_started) * 1000.0 / max(1, len(records))
            for record, stats in zip(records, torch_stats):
                if record.get("pil_image") is not None:
                    record["pil_image"].close()
                semantic = semantic_map.get(record["path"], SemanticResult())
                detection = detection_map.get(record["path"], DetectionResult())
                result = self._build_result(record, stats, semantic, detection, runtime_ms)
                results.append(result)

            if self.semantic_backend:
                self.performance_stats["embedding_cache_hits"] += self.semantic_backend.cache_hits
                self.performance_stats["embedding_new"] += self.semantic_backend.new_embeddings
                self.semantic_backend.cache_hits = 0
                self.semantic_backend.new_embeddings = 0
            if not (self.semantic_backend and self.semantic_backend.enabled):
                self.performance_stats["fallback"] += len(records)

        if self.config.enable_similarity:
            self._assign_similarity_groups(results)

        total_ms = (time.perf_counter() - started) * 1000.0
        self.performance_stats.update(
            {
                "total": total,
                "analyzed": len(results),
                "total_ms": total_ms,
                "avg_ms": total_ms / max(1, len(results)),
                "device": self.device_label,
                "batch_size": self.batch_size,
                "openclip_fp32_fallback": bool(self.semantic_backend and self.semantic_backend.fp32_fallback),
            }
        )
        self.logger.info(
            "AI性能统计：total=%s analyzed=%s avg_ms=%.2f device=%s batch=%s cache_hit=%s new=%s fallback=%s",
            total,
            len(results),
            self.performance_stats["avg_ms"],
            self.device_label,
            self.batch_size,
            self.performance_stats["embedding_cache_hits"],
            self.performance_stats["embedding_new"],
            self.performance_stats["fallback"],
        )
        return results

    def _analyze_image_signals(self, path: Path) -> dict:
        image = _read_bgr(path, self.cv2, self.np)
        if image is None:
            raise ValueError(f"无法读取图片：{path}")

        gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        small_gray = _resize_max_edge(gray, 1400, self.cv2)
        faces = self._detect_faces(small_gray)
        eyes_count = self._detect_eyes(small_gray, faces)

        laplacian_variance = float(self.cv2.Laplacian(gray, self.cv2.CV_64F).var())
        mean_brightness = float(gray.mean())
        bright_clip_ratio = float(self.np.mean(gray >= 250))
        dark_clip_ratio = float(self.np.mean(gray <= 5))
        contrast = float(gray.std())
        image_hash = _average_hash_from_gray(gray, self.cv2)
        resized = _resize_max_edge(image, self.config.max_image_size, self.cv2)
        small_rgb = self.cv2.cvtColor(self.cv2.resize(resized, (224, 224)), self.cv2.COLOR_BGR2RGB)

        pil_image = None
        if self.semantic_backend and self.semantic_backend.enabled:
            pil_image = Image.fromarray(self.cv2.cvtColor(resized, self.cv2.COLOR_BGR2RGB))

        return {
            "path": path,
            "faces": faces,
            "eyes_count": eyes_count,
            "laplacian_variance": laplacian_variance,
            "mean_brightness": mean_brightness,
            "bright_clip_ratio": bright_clip_ratio,
            "dark_clip_ratio": dark_clip_ratio,
            "contrast": contrast,
            "image_hash": image_hash,
            "small_rgb": small_rgb,
            "pil_image": pil_image,
        }

    def _build_result(
        self,
        record: dict,
        stats: dict,
        semantic: SemanticResult,
        detection: DetectionResult,
        runtime_ms: float,
    ) -> AIAnalysisResult:
        detected_count = detection.detected_person_count
        face_count = len(record["faces"])
        people_count = detected_count if detection.detection_confidence >= 0.25 else face_count
        if people_count == 0 and semantic.primary in {CAT_SINGLE, CAT_DOUBLE, CAT_MULTI, CAT_WEDDING, CAT_FAMILY, CAT_PARENT_CHILD, CAT_BUSINESS}:
            people_count = max(1, face_count)
        people_label = _people_label(people_count)
        appearance_tags = self._appearance_tags(record["path"], people_count)
        quality_tags = self._quality_tags(record, detection)
        shooting_type = self._shooting_type(record["path"], people_label, appearance_tags, semantic)

        primary, secondary, rule_confidence = self._recommend_category(
            people_label=people_label,
            shooting_type=shooting_type,
            appearance_tags=appearance_tags,
            quality_tags=quality_tags,
            semantic=semantic,
            detection=detection,
            mean=float(stats.get("mean", record["mean_brightness"] / 255.0)),
            contrast=float(stats.get("std", record["contrast"] / 255.0)),
        )
        confidence = _combined_confidence(rule_confidence, semantic.confidence, detection.detection_confidence)
        if confidence < self.config.confidence_threshold:
            primary, secondary = PRIMARY_REVIEW, ""
            _append_once(quality_tags, QUALITY_REVIEW)
            _append_once(quality_tags, QUALITY_UNCERTAIN)

        absolute_score = _absolute_quality_score(record, quality_tags, detection)
        semantic_score = round(max(semantic.score, semantic.confidence) * 100.0, 2)
        relative_score = absolute_score
        final_pick_score = _final_pick_score(self.config, absolute_score, semantic_score, relative_score, 0.0)
        ai_reason = _build_ai_reason(
            primary=primary,
            secondary=secondary,
            tags=quality_tags,
            confidence=confidence,
            score=absolute_score,
            semantic=semantic,
            detection=detection,
            final_pick_score=final_pick_score,
        )
        suggestions = _build_photo_suggestions(primary, secondary, quality_tags, semantic, detection, final_pick_score)

        return AIAnalysisResult(
            path=record["path"],
            people_label=people_label,
            people_count=people_count,
            shooting_type=shooting_type,
            appearance_tags=appearance_tags,
            ai_primary_category=primary,
            ai_secondary_category=secondary,
            ai_quality_tags=quality_tags,
            ai_confidence=round(confidence, 3),
            ai_source=self.backend_label,
            image_hash=record["image_hash"],
            absolute_quality_score=round(absolute_score, 2),
            relative_quality_score=round(relative_score, 2),
            ai_reason=ai_reason,
            ai_runtime_ms=round(runtime_ms, 2),
            ai_device=self.device_label,
            ai_batch_size=self.batch_size,
            semantic_score=semantic_score,
            semantic_confidence=round(semantic.confidence, 3),
            top3_semantic_matches=semantic.top3,
            final_pick_score=round(final_pick_score, 2),
            detected_person_count=detection.detected_person_count,
            main_subject_bbox=detection.main_subject_bbox,
            subject_area_ratio=round(detection.subject_area_ratio, 4),
            subject_center_score=round(detection.subject_center_score, 3),
            edge_cutoff_risk=round(detection.edge_cutoff_risk, 3),
            group_photo_score=round(detection.group_photo_score, 3),
            detection_confidence=round(detection.detection_confidence, 3),
            embedding_path=semantic.embedding_path,
            embedding_model=semantic.embedding_model,
            embedding_cached=semantic.embedding_cached,
            embedding_created_at=semantic.embedding_created_at,
            image_feature_hash=semantic.image_feature_hash,
            screening_reason=suggestions["screening_reason"],
            style_label=suggestions["style_label"],
            retouch_suggestion=suggestions["retouch_suggestion"],
            crop_suggestion=suggestions["crop_suggestion"],
            portfolio_suggestion=suggestions["portfolio_suggestion"],
            delivery_suggestion=suggestions["delivery_suggestion"],
            final_recommendation=suggestions["final_recommendation"],
        )

    def _quality_tags(self, record: dict, detection: DetectionResult) -> list[str]:
        tags: list[str] = []
        lap = record["laplacian_variance"]
        mean = record["mean_brightness"]
        severe_problem = False

        if lap < 18.0:
            tags.extend([QUALITY_SEVERE_BLURRY, QUALITY_CLEAR_REJECT])
            severe_problem = True
        elif lap < 85.0:
            tags.extend([QUALITY_SLIGHT_BLURRY, QUALITY_USABLE_RETOUCH])
        else:
            tags.append(QUALITY_CLEAR)

        if mean > 238.0 or record["bright_clip_ratio"] > 0.42:
            tags.extend([QUALITY_SEVERE_OVEREXPOSED, QUALITY_CLEAR_REJECT])
            severe_problem = True
        elif mean > 198.0 or record["bright_clip_ratio"] > 0.16:
            tags.extend([QUALITY_SLIGHT_OVEREXPOSED, QUALITY_USABLE_RETOUCH])

        if mean < 28.0 or record["dark_clip_ratio"] > 0.45:
            tags.extend([QUALITY_SEVERE_UNDEREXPOSED, QUALITY_CLEAR_REJECT])
            severe_problem = True
        elif mean < 62.0 or record["dark_clip_ratio"] > 0.20:
            tags.extend([QUALITY_SLIGHT_UNDEREXPOSED, QUALITY_USABLE_RETOUCH])

        if detection.edge_cutoff_risk >= 0.55:
            tags.extend([QUALITY_EDGE_ISSUE, QUALITY_RE_CROP, QUALITY_USABLE_CROP])
        if detection.subject_center_score and detection.subject_center_score < 0.35:
            tags.extend([QUALITY_RE_CROP, QUALITY_USABLE_CROP])
        if record["faces"] and record["eyes_count"] < max(1, len(record["faces"])):
            tags.append("疑似闭眼")
        if not severe_problem:
            _append_once(tags, QUALITY_KEEP if QUALITY_CLEAR in tags else QUALITY_USABLE_RETOUCH)
        return _dedupe_tags(tags)

    def _appearance_tags(self, path: Path, people_count: int) -> list[str]:
        if people_count <= 0:
            return []

        name = path.stem.lower()
        tags: list[str] = []
        if _has_keyword(name, ["child", "kid", "baby", "儿童", "宝宝", "孩子", "亲子"]):
            tags.append("儿童外观")
        elif _has_keyword(name, ["elder", "senior", "grand", "老人", "爷爷", "奶奶", "长辈"]):
            tags.append("中老年外观")
        else:
            tags.append("年轻成人外观")

        female_hint = _has_keyword(name, ["female", "girl", "woman", "lady", "bride", "女生", "女孩", "女士", "新娘", "闺蜜", "姐妹"])
        male_hint = _has_keyword(name, ["male", "boy", "man", "groom", "男生", "男士", "新郎", "兄弟"])
        if female_hint and male_hint:
            tags.append("混合性别外观")
        elif female_hint:
            tags.append("女性外观")
        elif male_hint:
            tags.append("男性外观")
        return tags

    def _shooting_type(self, path: Path, people_label: str, appearance_tags: list[str], semantic: SemanticResult) -> str:
        if semantic.primary:
            mapping = {
                CAT_SINGLE: "portrait",
                CAT_DOUBLE: "portrait",
                CAT_MULTI: "portrait",
                CAT_WEDDING: "wedding",
                CAT_FAMILY: "family",
                CAT_PARENT_CHILD: "parent_child",
                CAT_BUSINESS: "business",
                CAT_EVENT: "event",
                CAT_CHILD: "child",
                CAT_ELDERLY: "elderly",
                CAT_ENV: "environment",
            }
            if semantic.primary in mapping:
                return mapping[semantic.primary]

        name = path.stem.lower()
        if _has_keyword(name, ["wedding", "bridal", "bride", "groom", "婚纱", "婚礼", "新娘", "新郎"]):
            return "wedding"
        if _has_keyword(name, ["family", "全家福", "家庭"]):
            return "family"
        if _has_keyword(name, ["parent", "亲子", "母子", "母女", "父子", "父女"]):
            return "parent_child"
        if _has_keyword(name, ["meeting", "conference", "event", "会议", "活动", "纪实"]):
            return "event"
        if _has_keyword(name, ["business", "company", "team", "商务", "公司", "团队", "合影"]):
            return "business"
        if _has_keyword(name, ["travel", "outdoor", "park", "beach", "street", "landscape", "外景", "旅行", "旅拍", "户外", "风景"]):
            return "outdoor"
        if "儿童外观" in appearance_tags:
            return "child"
        if "中老年外观" in appearance_tags:
            return "elderly"
        if people_label == "none":
            return "environment"
        return "portrait"

    def _recommend_category(
        self,
        people_label: str,
        shooting_type: str,
        appearance_tags: list[str],
        quality_tags: list[str],
        semantic: SemanticResult,
        detection: DetectionResult,
        mean: float,
        contrast: float,
    ) -> tuple[str, str, float]:
        confidence = 0.40
        if people_label != "none":
            confidence += 0.15
        if shooting_type not in {"portrait", "environment"}:
            confidence += 0.12
        if semantic.primary:
            confidence += min(0.22, semantic.confidence * 0.28)
        if detection.detection_confidence > 0:
            confidence += min(0.10, detection.detection_confidence * 0.12)
        if QUALITY_CLEAR in quality_tags:
            confidence += 0.05
        if QUALITY_KEEP in quality_tags:
            confidence += 0.04
        if 0.18 <= contrast <= 0.42 and 0.24 <= mean <= 0.80:
            confidence += 0.03
        confidence = min(confidence, 0.97)

        severe_tags = {QUALITY_SEVERE_BLURRY, QUALITY_SEVERE_OVEREXPOSED, QUALITY_SEVERE_UNDEREXPOSED}
        if QUALITY_CLEAR_REJECT in quality_tags and any(tag in quality_tags for tag in severe_tags):
            return PRIMARY_REJECT, "", max(confidence, 0.66)

        if semantic.primary and semantic.primary != PRIMARY_REJECT:
            primary, secondary = _refine_semantic_category(semantic, people_label, detection, appearance_tags)
            if primary:
                return primary, secondary, confidence

        if shooting_type == "wedding":
            if people_label == "single_person":
                return CAT_WEDDING, "单人婚纱", confidence
            return CAT_WEDDING, "双人婚纱", confidence
        if shooting_type == "family":
            return CAT_FAMILY, "多人家庭" if people_label in {"three_people", "group_people"} else "其他家庭合影", confidence
        if shooting_type == "parent_child":
            return CAT_PARENT_CHILD, "其他亲子", confidence
        if shooting_type == "business":
            return CAT_BUSINESS, "大合影" if people_label == "group_people" else "小组合影", confidence
        if shooting_type == "event":
            return CAT_EVENT, "", confidence
        if shooting_type == "child":
            return CAT_CHILD, "", confidence
        if shooting_type == "elderly":
            return CAT_ELDERLY, "", confidence
        if shooting_type == "environment" or people_label == "none":
            return CAT_ENV, "", min(confidence, 0.72)
        if people_label == "single_person":
            if "女性外观" in appearance_tags:
                return CAT_SINGLE, "年轻女生", confidence
            if "男性外观" in appearance_tags:
                return CAT_SINGLE, "年轻男生", confidence
            return CAT_SINGLE, "其他单人", min(confidence, 0.72)
        if people_label == "two_people":
            if "女性外观" in appearance_tags:
                return CAT_DOUBLE, "闺蜜姐妹", confidence
            if "男性外观" in appearance_tags:
                return CAT_DOUBLE, "兄弟朋友", confidence
            return CAT_DOUBLE, "情侣", confidence
        if people_label in {"three_people", "group_people"}:
            return CAT_MULTI, "小组合影" if people_label == "three_people" else "其他多人", confidence
        return PRIMARY_REVIEW, "", min(confidence, 0.5)

    def _assign_similarity_groups(self, results: list[AIAnalysisResult]) -> None:
        if not results:
            return
        parent = list(range(len(results)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for left in range(len(results)):
            for right in range(left + 1, len(results)):
                if _hamming_distance(results[left].image_hash, results[right].image_hash) <= self.config.similarity_hamming_threshold:
                    union(left, right)

        groups: dict[int, list[int]] = {}
        for index in range(len(results)):
            groups.setdefault(find(index), []).append(index)

        group_number = 1
        for indexes in groups.values():
            if len(indexes) <= 1:
                continue
            group_id = f"{group_number:03d}"
            ranked = sorted(indexes, key=lambda idx: _quality_score(results[idx]), reverse=True)
            keep_count = min(self.config.max_keep_per_group, max(self.config.min_keep_per_group, 1))
            for index in indexes:
                result = results[index]
                result.similar_group_id = group_id
                result.similar_group_size = len(indexes)
                result.group_size = len(indexes)
                result.group_rank = ranked.index(index) + 1
                result.relative_quality_score = round((len(indexes) - result.group_rank + 1) / len(indexes) * 100, 2)
                result.final_pick_score = round(
                    _final_pick_score(
                        self.config,
                        result.absolute_quality_score,
                        result.semantic_score,
                        result.relative_quality_score,
                        0.0,
                    ),
                    2,
                )
                result.recommended_keep = result.group_rank <= keep_count
                result.recommended_in_group = result.recommended_keep
                if result.recommended_keep:
                    _append_once(result.ai_quality_tags, QUALITY_KEEP)
                    result.ai_reason = (
                        f"相似组 {group_id}，共 {len(indexes)} 张，本图排名 {result.group_rank}/{len(indexes)}。"
                        "清晰度、语义匹配和主体完整性在本组较好，建议保留复核。"
                    )
                    result.screening_reason = result.ai_reason
                    result.final_recommendation = "相似组推荐保留"
                else:
                    result.ai_primary_category = PRIMARY_DUPLICATE
                    result.ai_secondary_category = ""
                    result.ai_confidence = max(result.ai_confidence, 0.82)
                    _append_once(result.ai_quality_tags, QUALITY_DUPLICATE)
                    result.ai_reason = (
                        f"与同组照片高度相似，当前排名 {result.group_rank}/{len(indexes)}，"
                        "建议作为相似重复待选，不自动删除。"
                    )
                    result.screening_reason = result.ai_reason
                    result.final_recommendation = "相似重复待选"
            group_number += 1
        self.logger.info("相似检测结果：%s 组", group_number - 1)

    def _detect_faces(self, gray) -> list[tuple[int, int, int, int]]:
        faces = []
        if self.face_detector is not None:
            faces.extend(self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(28, 28)))
        if self.profile_detector is not None:
            faces.extend(self.profile_detector.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(28, 28)))
        return _dedupe_boxes([tuple(map(int, face)) for face in faces])

    def _detect_eyes(self, gray, faces: list[tuple[int, int, int, int]]) -> int:
        if self.eye_detector is None:
            return 0
        total = 0
        for x, y, w, h in faces[:8]:
            roi = gray[y : y + max(1, h // 2), x : x + w]
            eyes = self.eye_detector.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=4, minSize=(8, 8))
            total += min(2, len(eyes))
        return total

    def _torch_batch_stats(self, small_images: list) -> list[dict]:
        if not small_images:
            return []
        if self.torch is None:
            return [{"mean": float(img.mean() / 255.0), "std": float(img.std() / 255.0)} for img in small_images]

        array = self.np.stack(small_images).astype("float32") / 255.0
        tensor = self.torch.from_numpy(array).to(self.device, non_blocking=self.config.pin_memory)
        if self.config.use_fp16 and getattr(self.device, "type", "") == "cuda":
            tensor = tensor.half()
        with self.torch.no_grad():
            means = tensor.mean(dim=(1, 2, 3)).detach().cpu().numpy()
            stds = tensor.std(dim=(1, 2, 3)).detach().cpu().numpy()
        return [{"mean": float(mean), "std": float(std)} for mean, std in zip(means, stds)]

    def _load_haar(self, filename: str):
        cascade_path = Path(self.cv2.data.haarcascades) / filename
        if not cascade_path.exists():
            return None
        detector = self.cv2.CascadeClassifier(str(cascade_path))
        return detector if not detector.empty() else None


class _OpenClipSemanticBackend:
    def __init__(self, config: AppConfig, manager: ModelManager, device, torch_module):
        self.config = config
        self.manager = manager
        self.enabled = False
        self.status = "fallback"
        self.cache_hits = 0
        self.new_embeddings = 0
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._label_features = None
        self._label_keys: list[tuple[str, str]] = []
        self._torch = torch_module
        self._device = device
        self._spec = get_openclip_spec(config.semantic_model, config.model_profile)
        self._use_fp16 = bool(config.use_fp16 and getattr(device, "type", "") == "cuda")
        self.fp32_fallback = False
        self.last_error = ""

        if not config.use_clip or torch_module is None:
            self.status = "no_torch_or_disabled"
            return
        if not manager.semantic_marker(self._spec).exists():
            self.status = "model_missing"
            return
        try:
            self._initialize_openclip(self._use_fp16)
            self.enabled = True
            self.status = "loaded"
        except Exception as exc:
            self.last_error = str(exc)
            if self._use_fp16:
                get_logger().exception("OpenCLIP FP16 初始化失败，正在回退 FP32")
                if self._retry_fp32("初始化失败"):
                    return
            get_logger().exception("OpenCLIP 初始化失败，已回退 fallback")
            self.status = "load_failed"

    def _initialize_openclip(self, use_fp16: bool) -> None:
        import open_clip

        self._use_fp16 = bool(use_fp16 and getattr(self._device, "type", "") == "cuda")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self._spec.model_name,
            pretrained=self._spec.pretrained,
            device=self._device,
            cache_dir=str(self.manager.semantic_cache_dir(self._spec)),
        )
        self._model.eval()
        if self._use_fp16:
            self._model = self._model.half()
        self._tokenizer = open_clip.get_tokenizer(self._spec.model_name)
        self._build_text_features()

    def _retry_fp32(self, reason: str) -> bool:
        try:
            if getattr(self._device, "type", "") == "cuda":
                self._torch.cuda.empty_cache()
            self.fp32_fallback = True
            self._initialize_openclip(False)
            self.enabled = True
            self.status = "loaded_fp32_fallback"
            self.last_error = f"FP16 {reason}，已回退 FP32"
            get_logger().warning("OpenCLIP FP16 %s，已回退 FP32", reason)
            return True
        except Exception as exc:
            self.last_error = str(exc)
            get_logger().exception("OpenCLIP FP32 回退失败")
            return False

    def _autocast_context(self):
        if self._use_fp16 and getattr(self._device, "type", "") == "cuda" and hasattr(self._torch, "autocast"):
            return self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
        return nullcontext()

    @property
    def embedding_model_name(self) -> str:
        return f"{self._spec.model_name}/{self._spec.pretrained}"

    def _build_text_features(self) -> None:
        prompts = semantic_prompt_groups()
        label_vectors = []
        label_keys = []
        with self._torch.no_grad():
            for key, prompt_list in prompts.items():
                tokens = self._tokenizer(prompt_list).to(self._device)
                with self._autocast_context():
                    text_features = self._model.encode_text(tokens)
                text_features = text_features.float()
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                label_feature = text_features.mean(dim=0, keepdim=True)
                label_feature = label_feature / label_feature.norm(dim=-1, keepdim=True)
                label_vectors.append(label_feature)
                label_keys.append(key)
        self._label_features = self._torch.cat(label_vectors, dim=0).float()
        self._label_keys = label_keys

    def classify_records(self, records: list[dict]) -> dict[Path, SemanticResult]:
        if not self.enabled or not records:
            return {}

        results: dict[Path, SemanticResult] = {}
        to_compute: list[dict] = []
        cached_features = {}
        for record in records:
            path = record["path"]
            vector, cache_path, digest = load_embedding_cache(path, self.config)
            if vector:
                cached_features[path] = self._torch.tensor(vector, device=self._device).unsqueeze(0)
                self.cache_hits += 1
                results[path] = SemanticResult(
                    embedding_path=str(cache_path),
                    embedding_model=self.embedding_model_name,
                    embedding_cached=True,
                    embedding_created_at="",
                    image_feature_hash=digest,
                    source="openclip_cache",
                )
            else:
                to_compute.append(record)

        if to_compute:
            computed = self._encode_uncached_records(to_compute)
            for path, feature in computed.items():
                cache_path, digest = save_embedding_cache(path, self.config, feature.detach().float().cpu().tolist())
                cached_features[path] = feature.unsqueeze(0)
                self.new_embeddings += 1
                results[path] = SemanticResult(
                    embedding_path=str(cache_path),
                    embedding_model=self.embedding_model_name,
                    embedding_cached=False,
                    embedding_created_at=datetime.now().isoformat(timespec="seconds"),
                    image_feature_hash=digest,
                    source="openclip",
                )

        for path, feature in cached_features.items():
            enriched = self._classify_feature(path, feature)
            base = results.get(path, SemanticResult())
            enriched.embedding_path = base.embedding_path
            enriched.embedding_model = base.embedding_model
            enriched.embedding_cached = base.embedding_cached
            enriched.embedding_created_at = base.embedding_created_at
            enriched.image_feature_hash = base.image_feature_hash
            enriched.source = base.source
            results[path] = enriched
        return results

    def _encode_uncached_records(self, records: list[dict]) -> dict[Path, object]:
        if not records:
            return {}
        try:
            return self._encode_records_once(records)
        except RuntimeError as exc:
            message = str(exc).lower()
            if self._use_fp16 and ("dtype" in message or "half" in message or "float" in message):
                if self._retry_fp32("推理 dtype 不一致"):
                    return self._encode_records_once(records)
            if "out of memory" not in str(exc).lower() or len(records) == 1:
                raise
            if getattr(self._device, "type", "") == "cuda":
                self._torch.cuda.empty_cache()
            midpoint = len(records) // 2
            encoded = self._encode_uncached_records(records[:midpoint])
            encoded.update(self._encode_uncached_records(records[midpoint:]))
            return encoded

    def _encode_records_once(self, records: list[dict]) -> dict[Path, object]:
        tensors = []
        paths = []
        for record in records:
            image = record.get("pil_image")
            if image is None:
                continue
            tensors.append(self._preprocess(ImageOps.exif_transpose(image).convert("RGB")))
            paths.append(record["path"])
        if not tensors:
            return {}
        batch = self._torch.stack(tensors).to(self._device, non_blocking=self.config.pin_memory)
        if self._use_fp16:
            batch = batch.half()
        with self._torch.no_grad():
            with self._autocast_context():
                image_features = self._model.encode_image(batch)
            image_features = image_features.float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return {path: feature.detach() for path, feature in zip(paths, image_features)}

    def _classify_feature(self, path: Path, feature) -> SemanticResult:
        with self._torch.no_grad():
            feature = feature.to(device=self._device, dtype=self._torch.float32)
            label_features = self._label_features.to(device=self._device, dtype=self._torch.float32)
            logits = 100.0 * feature @ label_features.T
            probabilities = logits.softmax(dim=-1).squeeze(0)
            top_values, top_indexes = probabilities.topk(min(3, len(self._label_keys)))
        top3 = []
        for value, index in zip(top_values.detach().cpu().tolist(), top_indexes.detach().cpu().tolist()):
            primary, secondary = self._label_keys[index]
            label = f"{primary}/{secondary}" if secondary else primary
            top3.append(f"{label}:{value:.2f}")
        best_index = int(top_indexes[0].detach().cpu().item())
        primary, secondary = self._label_keys[best_index]
        confidence = float(top_values[0].detach().cpu().item())
        return SemanticResult(
            primary=primary,
            secondary=secondary,
            confidence=confidence,
            score=confidence,
            top3=top3,
        )

    def smoke_test(self, image_path: Path) -> dict:
        try:
            image = Image.open(image_path)
            tensor = self._preprocess(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0)
            tensor = tensor.to(self._device, non_blocking=self.config.pin_memory)
            if self._use_fp16:
                tensor = tensor.half()

            prompts = ["single person portrait", "wedding photography", "landscape photo without people"]
            with self._torch.no_grad():
                with self._autocast_context():
                    image_features = self._model.encode_image(tensor)
                    tokens = self._tokenizer(prompts).to(self._device)
                    text_features = self._model.encode_text(tokens)
                image_features = image_features.float()
                text_features = text_features.float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits = image_features @ text_features.T
        except RuntimeError as exc:
            message = str(exc).lower()
            if self._use_fp16 and ("dtype" in message or "half" in message or "float" in message):
                if self._retry_fp32("smoke test dtype 不一致"):
                    return self.smoke_test(image_path)
            raise
        return {
            "ok": True,
            "status": self.status,
            "use_fp16": self._use_fp16,
            "fp32_fallback": self.fp32_fallback,
            "image_tensor_dtype": str(tensor.dtype),
            "image_feature_dtype": str(image_features.dtype),
            "text_feature_dtype": str(text_features.dtype),
            "similarity_dtype": str(logits.dtype),
            "similarity_shape": tuple(logits.shape),
            "last_error": self.last_error,
        }


class _YoloDetectorBackend:
    def __init__(self, config: AppConfig, manager: ModelManager, device_label: str):
        self.config = config
        self.manager = manager
        self.enabled = False
        self.status = "fallback"
        self._model = None
        self._device = 0 if config.use_gpu and device_label.startswith("GPU: CUDA") else "cpu"
        self._spec = get_yolo_spec(config.detector_model, config.model_profile)
        model_path = manager.detector_path(self._spec)
        if not model_path.exists():
            self.status = "model_missing"
            return
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(model_path))
            self.enabled = True
            self.status = "loaded"
        except Exception:
            get_logger().exception("YOLO 初始化失败，已回退 fallback")
            self.status = "load_failed"

    def detect_paths(self, paths: list[Path]) -> dict[Path, DetectionResult]:
        if not self.enabled or not paths:
            return {}
        try:
            predictions = self._model.predict(
                source=[str(path) for path in paths],
                imgsz=self.config.max_image_size,
                batch=max(1, min(len(paths), self.config.batch_size)),
                device=self._device,
                half=bool(self.config.use_fp16 and self._device != "cpu"),
                verbose=False,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self._device != "cpu":
                get_logger().exception("YOLO CUDA 显存不足，回退 CPU")
                self._device = "cpu"
                return self.detect_paths(paths)
            get_logger().exception("YOLO 检测失败")
            return {}
        except Exception:
            get_logger().exception("YOLO 检测失败")
            return {}

        results: dict[Path, DetectionResult] = {}
        for path, prediction in zip(paths, predictions):
            results[path] = _parse_yolo_prediction(prediction)
        return results


def apply_ai_result_to_item(item, result: AIAnalysisResult, confidence_threshold: float = 0.65) -> None:
    item.people_label = result.people_label
    item.people_count = result.people_count
    item.shooting_type = result.shooting_type
    item.appearance_tags = result.appearance_tags
    item.ai_primary_category = result.ai_primary_category
    item.ai_secondary_category = result.ai_secondary_category
    item.ai_quality_tags = result.ai_quality_tags
    item.ai_confidence = result.ai_confidence
    item.ai_source = result.ai_source
    item.similar_group_id = result.similar_group_id
    item.similar_group_size = result.similar_group_size
    item.recommended_keep = result.recommended_keep
    item.absolute_quality_score = result.absolute_quality_score
    item.relative_quality_score = result.relative_quality_score
    item.group_rank = result.group_rank
    item.group_size = result.group_size
    item.recommended_in_group = result.recommended_in_group
    item.ai_reason = result.ai_reason
    item.ai_runtime_ms = result.ai_runtime_ms
    item.ai_device = result.ai_device
    item.ai_batch_size = result.ai_batch_size
    item.semantic_score = result.semantic_score
    item.semantic_confidence = result.semantic_confidence
    item.top3_semantic_matches = result.top3_semantic_matches
    item.final_pick_score = result.final_pick_score
    item.detected_person_count = result.detected_person_count
    item.main_subject_bbox = result.main_subject_bbox
    item.subject_area_ratio = result.subject_area_ratio
    item.subject_center_score = result.subject_center_score
    item.edge_cutoff_risk = result.edge_cutoff_risk
    item.group_photo_score = result.group_photo_score
    item.detection_confidence = result.detection_confidence
    item.embedding_path = result.embedding_path
    item.embedding_model = result.embedding_model
    item.embedding_cached = result.embedding_cached
    item.embedding_created_at = result.embedding_created_at
    item.image_feature_hash = result.image_feature_hash
    item.screening_reason = result.screening_reason or result.ai_reason
    item.style_label = result.style_label
    item.retouch_suggestion = result.retouch_suggestion
    item.crop_suggestion = result.crop_suggestion
    item.portfolio_suggestion = result.portfolio_suggestion
    item.delivery_suggestion = result.delivery_suggestion
    item.final_recommendation = result.final_recommendation
    item.final_reason = result.ai_reason
    item.compute_final_category(confidence_threshold)


def semantic_prompt_groups() -> dict[tuple[str, str], list[str]]:
    return {
        (CAT_SINGLE, "年轻女生"): [
            "single person portrait",
            "studio portrait of a young woman",
            "personal portrait photography of a young woman",
            "年轻女性写真 人像 棚拍",
        ],
        (CAT_SINGLE, "年轻男生"): [
            "studio portrait of a young man",
            "personal portrait photography of a young man",
            "male portrait photography",
            "年轻男性写真 人像",
        ],
        (CAT_SINGLE, "其他单人"): [
            "single person portrait",
            "personal portrait photography",
            "studio portrait of one person",
        ],
        (CAT_DOUBLE, "情侣"): [
            "couple portrait photography",
            "two people romantic portrait",
            "couple studio portrait",
            "情侣写真 双人合影",
        ],
        (CAT_DOUBLE, "闺蜜姐妹"): [
            "two friends portrait",
            "best friends portrait",
            "sisters portrait style",
            "two women studio portrait",
            "闺蜜姐妹写真",
        ],
        (CAT_DOUBLE, "兄弟朋友"): [
            "two male friends portrait",
            "brothers portrait style",
            "two friends studio portrait",
        ],
        (CAT_MULTI, "小组合影"): [
            "small group portrait",
            "friends group photo",
            "group portrait photography",
        ],
        (CAT_WEDDING, "单人婚纱"): [
            "bridal portrait",
            "bride wedding portrait",
            "single bride portrait photography",
            "单人婚纱照 新娘写真",
        ],
        (CAT_WEDDING, "双人婚纱"): [
            "wedding photography",
            "bride and groom wedding portrait",
            "couple wedding photoshoot",
            "双人婚纱照 新郎新娘",
        ],
        (CAT_WEDDING, "中式婚纱"): [
            "Chinese wedding portrait",
            "traditional Chinese wedding photography",
            "中式婚纱照",
        ],
        (CAT_WEDDING, "外景婚纱"): [
            "outdoor wedding photoshoot",
            "wedding portrait in nature",
            "外景婚纱照",
        ],
        (CAT_WEDDING, "棚拍婚纱"): [
            "studio wedding photoshoot",
            "wedding portrait in photography studio",
            "棚拍婚纱照",
        ],
        (CAT_FAMILY, "三口之家"): [
            "three person family photo",
            "family portrait with child",
            "三口之家 全家福",
        ],
        (CAT_FAMILY, "多人家庭"): [
            "family portrait",
            "multi generation family portrait",
            "large family group photo",
            "多人家庭 全家福",
        ],
        (CAT_PARENT_CHILD, "其他亲子"): [
            "parent and child portrait",
            "mother and child photo",
            "father and child photo",
            "family with child portrait",
            "亲子照",
        ],
        (CAT_CHILD, ""): [
            "child portrait photography",
            "studio photo of a child",
            "儿童写真",
        ],
        (CAT_ELDERLY, ""): [
            "elderly portrait",
            "middle aged portrait",
            "mature person portrait",
            "中老年写真",
        ],
        (CAT_BUSINESS, "单人商务形象"): [
            "business portrait",
            "corporate headshot",
            "professional business portrait",
            "商务形象照",
        ],
        (CAT_BUSINESS, "大合影"): [
            "corporate group photo",
            "large business group portrait",
            "company team group photo",
            "商务会议大合影",
        ],
        (CAT_BUSINESS, "会议现场"): [
            "meeting room event photography",
            "conference event photo",
            "business meeting documentary photo",
            "会议现场 活动纪实",
        ],
        (CAT_EVENT, ""): [
            "event photography",
            "documentary event photo",
            "conference event photo",
            "活动纪实 摄影",
        ],
        (CAT_ENV, ""): [
            "landscape photo",
            "cityscape photo",
            "environment photo without people",
            "风景 环境 空镜",
        ],
        (PRIMARY_REJECT, ""): [
            "blurry photo",
            "bad expression photo",
            "overexposed unusable photo",
            "underexposed unusable photo",
            "out of focus photo",
        ],
    }


def _parse_yolo_prediction(prediction) -> DetectionResult:
    try:
        height, width = prediction.orig_shape
        boxes = prediction.boxes
        if boxes is None or len(boxes) == 0:
            return DetectionResult(source="yolo")
        cls = boxes.cls.detach().cpu().tolist()
        conf = boxes.conf.detach().cpu().tolist()
        xyxy = boxes.xyxy.detach().cpu().tolist()
        person_boxes = [
            (box, float(score))
            for box, label, score in zip(xyxy, cls, conf)
            if int(label) == 0 and float(score) >= 0.25
        ]
        if not person_boxes:
            return DetectionResult(source="yolo")
        main_box, main_conf = max(person_boxes, key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]))
        x1, y1, x2, y2 = [float(value) for value in main_box]
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / max(1.0, float(width * height)))
        center_x = (x1 + x2) / 2.0 / max(1.0, float(width))
        center_y = (y1 + y2) / 2.0 / max(1.0, float(height))
        distance = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5
        center_score = max(0.0, 1.0 - distance / 0.707)
        margin = 0.025
        edge_risk = 0.0
        if x1 <= width * margin or y1 <= height * margin or x2 >= width * (1 - margin) or y2 >= height * (1 - margin):
            edge_risk = 0.72
        detection_confidence = max(score for _box, score in person_boxes)
        return DetectionResult(
            detected_person_count=len(person_boxes),
            main_subject_bbox=f"{int(x1)},{int(y1)},{int(x2)},{int(y2)}",
            subject_area_ratio=area_ratio,
            subject_center_score=center_score,
            edge_cutoff_risk=edge_risk,
            group_photo_score=min(1.0, len(person_boxes) / 6.0),
            detection_confidence=detection_confidence,
            source="yolo",
        )
    except Exception:
        get_logger().exception("YOLO 结果解析失败")
        return DetectionResult(source="yolo_parse_failed")


def _read_bgr(path: Path, cv2, np_module):
    data = np_module.fromfile(str(path), dtype=np_module.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _resize_max_edge(image, max_edge: int, cv2):
    height, width = image.shape[:2]
    edge = max(height, width)
    if edge <= max_edge:
        return image
    scale = max_edge / edge
    return cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))


def _average_hash_from_gray(gray, cv2, hash_size: int = 8) -> str:
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    bits = small > mean
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    width = hash_size * hash_size // 4
    return f"{value:0{width}x}"


def _hamming_distance(hash_a: str, hash_b: str) -> int:
    if not hash_a or not hash_b:
        return 64
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


def _people_label(count: int) -> str:
    if count <= 0:
        return "none"
    if count == 1:
        return "single_person"
    if count == 2:
        return "two_people"
    if count == 3:
        return "three_people"
    return "group_people"


def _quality_score(result: AIAnalysisResult) -> float:
    score = result.final_pick_score or result.absolute_quality_score
    if QUALITY_CLEAR in result.ai_quality_tags:
        score += 6.0
    if QUALITY_BLURRY in result.ai_quality_tags or QUALITY_SEVERE_BLURRY in result.ai_quality_tags:
        score -= 18.0
    if QUALITY_SEVERE_OVEREXPOSED in result.ai_quality_tags or QUALITY_SEVERE_UNDEREXPOSED in result.ai_quality_tags:
        score -= 14.0
    return score


def _absolute_quality_score(record: dict, quality_tags: list[str], detection: DetectionResult) -> float:
    score = 66.0
    score += min(22.0, max(0.0, record["laplacian_variance"] / 900.0 * 22.0))
    score += min(8.0, record["contrast"] / 64.0 * 8.0)
    score += detection.subject_center_score * 6.0 if detection.subject_center_score else 0.0
    score -= 42.0 if QUALITY_CLEAR_REJECT in quality_tags else 0.0
    score -= 10.0 if QUALITY_USABLE_RETOUCH in quality_tags else 0.0
    score -= 7.0 if QUALITY_USABLE_CROP in quality_tags else 0.0
    score += 6.0 if QUALITY_KEEP in quality_tags else 0.0
    return max(0.0, min(100.0, score))


def _final_pick_score(config: AppConfig, visual_score: float, semantic_score: float, relative_score: float, preference_score: float) -> float:
    weights = (config.scoring_weights or {}).get(config.screening_strategy) or (config.scoring_weights or {}).get("delivery") or {}
    return (
        visual_score * float(weights.get("visual_quality", 0.35))
        + semantic_score * float(weights.get("semantic", 0.25))
        + relative_score * float(weights.get("relative", 0.25))
        + preference_score * 100.0 * float(weights.get("preference", 0.15))
    )


def _combined_confidence(rule_confidence: float, semantic_confidence: float, detection_confidence: float) -> float:
    if semantic_confidence <= 0 and detection_confidence <= 0:
        return max(0.0, min(0.96, rule_confidence))
    return max(
        0.0,
        min(
            0.98,
            rule_confidence * 0.35
            + (semantic_confidence or rule_confidence) * 0.45
            + (detection_confidence or rule_confidence) * 0.20,
        ),
    )


def _build_ai_reason(
    primary: str,
    secondary: str,
    tags: list[str],
    confidence: float,
    score: float,
    semantic: SemanticResult,
    detection: DetectionResult,
    final_pick_score: float,
) -> str:
    category = f"{primary}/{secondary}" if secondary else primary
    semantic_text = "；OpenCLIP候选：" + "，".join(semantic.top3) if semantic.top3 else "；语义模型未加载，使用轻量fallback"
    detection_text = (
        f"；YOLO检测人数 {detection.detected_person_count}，主体面积 {detection.subject_area_ratio:.2f}，居中分 {detection.subject_center_score:.2f}"
        if detection.detection_confidence
        else "；检测模型未加载或未检测到可靠人物"
    )
    if QUALITY_CLEAR_REJECT in tags:
        return (
            f"绝对质量分 {score:.1f}，最终分 {final_pick_score:.1f}，存在严重不可修复风险，"
            f"建议进入“{category}”。{semantic_text}{detection_text}"
        )
    if QUALITY_USABLE_CROP in tags:
        return (
            f"绝对质量分 {score:.1f}，最终分 {final_pick_score:.1f}，主体仍有交付价值，但边缘或构图存在风险，"
            f"建议可用待裁切 / 人工复核。{semantic_text}{detection_text}"
        )
    if QUALITY_USABLE_RETOUCH in tags:
        return (
            f"绝对质量分 {score:.1f}，最终分 {final_pick_score:.1f}，画面存在轻微模糊或曝光问题，"
            f"但不直接判废，建议可用待修。{semantic_text}{detection_text}"
        )
    return (
        f"绝对质量分 {score:.1f}，最终分 {final_pick_score:.1f}，置信度 {confidence:.2f}，"
        f"建议分类为“{category}”。{semantic_text}{detection_text}"
    )


def _build_photo_suggestions(
    primary: str,
    secondary: str,
    tags: list[str],
    semantic: SemanticResult,
    detection: DetectionResult,
    final_pick_score: float,
) -> dict[str, str]:
    category = f"{primary}/{secondary}" if secondary else primary
    if primary == CAT_WEDDING:
        style = "偏婚纱正式照"
    elif primary == CAT_BUSINESS:
        style = "偏商业形象照"
    elif primary == CAT_EVENT:
        style = "偏纪实风"
    elif primary == CAT_ENV:
        style = "偏风景环境"
    elif primary in {CAT_SINGLE, CAT_DOUBLE, CAT_CHILD, CAT_ELDERLY}:
        style = "偏棚拍写真" if detection.subject_area_ratio >= 0.18 else "偏生活感"
    else:
        style = "偏生活感"

    if QUALITY_SEVERE_OVEREXPOSED in tags:
        retouch = "高光损失较重，建议人工判断是否仍可恢复；可尝试降低高光和压暗背景。"
    elif QUALITY_SEVERE_UNDEREXPOSED in tags:
        retouch = "暗部损失较重，建议提升人物面部亮度并检查噪点。"
    elif QUALITY_USABLE_RETOUCH in tags:
        retouch = "建议基础修图：校正曝光、色温和面部亮度，必要时弱化背景杂物。"
    else:
        retouch = "画面基础质量较稳定，可做常规肤色、对比度和细节微调。"

    if QUALITY_USABLE_CROP in tags or detection.edge_cutoff_risk >= 0.55:
        crop = "边缘存在裁切或穿帮风险，建议二次裁切；优先尝试 4:5 竖版或收紧边缘。"
    elif detection.subject_center_score and detection.subject_center_score < 0.45:
        crop = "主体偏离中心，建议调整裁切让人物视觉重心更稳定。"
    elif primary == CAT_ENV:
        crop = "可尝试 16:9 横版，突出环境层次。"
    else:
        crop = "构图基本成立，保持原比例或按交付尺寸轻微裁切即可。"

    if final_pick_score >= 78 and QUALITY_CLEAR_REJECT not in tags:
        portfolio = "有进入精选或社交媒体发布候选的潜力，建议复核表情和细节。"
        delivery = "可作为客户交付候选。"
        final_recommendation = "推荐保留"
    elif QUALITY_CLEAR_REJECT in tags:
        portfolio = "不建议进入作品集。"
        delivery = "交付风险较高，建议人工复核后再决定。"
        final_recommendation = "废片疑似"
    elif QUALITY_USABLE_CROP in tags:
        portfolio = "不建议直接进作品集，但可作为裁切后备选。"
        delivery = "主体仍有交付价值，建议可用待裁切。"
        final_recommendation = "可用待裁切"
    elif QUALITY_USABLE_RETOUCH in tags:
        portfolio = "需要修图后再判断是否进入作品集。"
        delivery = "可作为可用待修候选。"
        final_recommendation = "可用待修"
    else:
        portfolio = "画面可用，建议结合相似组对比决定是否精选。"
        delivery = "可作为备选交付。"
        final_recommendation = "人工复核"

    if primary == PRIMARY_DUPLICATE:
        final_recommendation = "相似重复待选"
        delivery = "同组照片需对比后保留 1-3 张，不建议直接删除。"

    screening = f"建议：{final_recommendation}。分类倾向为“{category}”，最终分 {final_pick_score:.1f}。"
    if semantic.top3:
        screening += f" 语义候选：{', '.join(semantic.top3[:3])}。"
    return {
        "screening_reason": screening,
        "style_label": style,
        "retouch_suggestion": retouch,
        "crop_suggestion": crop,
        "portfolio_suggestion": portfolio,
        "delivery_suggestion": delivery,
        "final_recommendation": final_recommendation,
    }


def _refine_semantic_category(
    semantic: SemanticResult,
    people_label: str,
    detection: DetectionResult,
    appearance_tags: list[str],
) -> tuple[str, str]:
    primary, secondary = semantic.primary, semantic.secondary
    person_count = detection.detected_person_count
    if primary == CAT_SINGLE and people_label == "two_people":
        return CAT_DOUBLE, "情侣"
    if primary == CAT_DOUBLE and people_label == "single_person":
        return CAT_SINGLE, secondary if secondary in {"年轻女生", "年轻男生"} else "其他单人"
    if primary == CAT_WEDDING:
        if people_label == "single_person":
            return CAT_WEDDING, "单人婚纱"
        if people_label in {"two_people", "three_people", "group_people"}:
            return CAT_WEDDING, "双人婚纱"
    if primary == CAT_FAMILY and person_count >= 3:
        return CAT_FAMILY, "多人家庭"
    if primary == CAT_BUSINESS and person_count >= 6:
        return CAT_BUSINESS, "大合影"
    if primary == CAT_DOUBLE and secondary == "":
        if "女性外观" in appearance_tags:
            return CAT_DOUBLE, "闺蜜姐妹"
        if "男性外观" in appearance_tags:
            return CAT_DOUBLE, "兄弟朋友"
        return CAT_DOUBLE, "情侣"
    return primary, secondary


def _dedupe_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        if all(_box_iou(box, existing) < 0.35 for existing in kept):
            kept.append(box)
    return kept


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def _has_keyword(value: str, keywords: list[str]) -> bool:
    return any(keyword in value for keyword in keywords)


def _append_once(tags: list[str], value: str) -> None:
    if value and value not in tags:
        tags.append(value)


def _dedupe_tags(tags: list[str]) -> list[str]:
    result = []
    for tag in tags:
        if tag not in result:
            result.append(tag)
    return result
