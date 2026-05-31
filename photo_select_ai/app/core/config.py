from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.json"


@dataclass
class AppConfig:
    use_gpu: bool = True
    batch_size: int = 16
    confidence_threshold: float = 0.58
    thumbnail_size: int = 180
    thumbnail_mode: str = "contain"
    preview_max_size: int = 1600
    export_mode: str = "copy"
    enable_ai: bool = True
    enable_similarity: bool = True
    theme: str = "dark_preview"
    max_compare_images: int = 4
    max_image_size: int = 1024
    use_fp16: bool = True
    import_batch_size: int = 40
    similarity_hamming_threshold: int = 6
    use_clip: bool = True
    autosave_interval_seconds: int = 60
    ask_save_on_exit: bool = True
    restore_last_project: bool = True
    target_pick_ratio: float = 0.20
    max_pending_ratio: float = 0.25
    min_keep_per_group: int = 1
    max_keep_per_group: int = 3
    prefer_relative_ranking: bool = True
    screening_strategy: str = "delivery"
    num_workers: int = 4
    prefetch: bool = True
    pin_memory: bool = True
    model_profile: str = "balanced"
    semantic_model: str = "openclip_vit_l_14"
    detector_model: str = "yolo11x"
    model_cache_dir: str = "models/cache"
    auto_download_models: bool = False
    allow_large_model_download: bool = True
    scoring_weights: dict[str, dict[str, float]] | None = None
    enable_idle_tasks: bool = False
    enable_idle_self_test: bool = False
    enable_idle_embedding: bool = False
    enable_idle_similarity: bool = False
    enable_idle_preference_training: bool = False
    aesthetic_library_folders: list[str] | None = None
    background_task_start_mode: str = "manual"
    background_task_enabled_types: list[str] | None = None
    similarity_mode: str = "standard"
    enable_embedding_grouping: bool = True
    embedding_backend: str = "auto"
    embedding_model_name: str = "openclip_vit_l_14"
    embedding_model_version: str = "v1"
    embedding_device: str = "auto"
    embedding_batch_size: int = 16
    embedding_similarity_threshold: float = 0.86
    embedding_group_min_size: int = 2
    grouping_method: str = "embedding"
    grouping_strategy: str = "complete_linkage"
    group_min_similarity_threshold: float = 0.92
    max_embedding_group_size: int = 25


def load_config() -> AppConfig:
    if not CONFIG_PATH.exists():
        save_config(AppConfig())
        return AppConfig()

    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        logging.getLogger("photoselect").exception("配置读取失败，已使用默认配置")
        return AppConfig()

    config = AppConfig()
    for key, value in raw.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.batch_size = max(1, int(config.batch_size))
    config.thumbnail_size = int(config.thumbnail_size)
    config.preview_max_size = int(config.preview_max_size)
    config.import_batch_size = max(1, int(config.import_batch_size))
    config.max_compare_images = max(2, int(config.max_compare_images))
    config.confidence_threshold = float(config.confidence_threshold)
    config.autosave_interval_seconds = max(10, int(config.autosave_interval_seconds))
    config.target_pick_ratio = max(0.01, min(1.0, float(config.target_pick_ratio)))
    config.max_pending_ratio = max(0.0, min(1.0, float(config.max_pending_ratio)))
    config.min_keep_per_group = max(1, int(config.min_keep_per_group))
    config.max_keep_per_group = max(config.min_keep_per_group, int(config.max_keep_per_group))
    config.num_workers = max(0, int(config.num_workers))
    if config.model_profile not in {"fast", "balanced", "accurate", "ultra"}:
        config.model_profile = "balanced"
    if not isinstance(config.aesthetic_library_folders, list):
        config.aesthetic_library_folders = []
    if config.background_task_start_mode not in {"manual", "ask_when_idle", "auto_low_priority"}:
        config.background_task_start_mode = "manual"
    if not isinstance(config.background_task_enabled_types, list):
        config.background_task_enabled_types = ["thumbnail_cache", "similar_photos", "export_preview"]
    if config.similarity_mode not in {"strict", "standard", "loose"}:
        config.similarity_mode = "standard"
    if config.embedding_backend not in {"auto", "openclip", "torch", "onnx", "mock", "fallback"}:
        config.embedding_backend = "auto"
    if config.embedding_device not in {"auto", "cuda", "cpu"}:
        config.embedding_device = "auto"
    config.embedding_batch_size = max(1, int(config.embedding_batch_size))
    config.embedding_similarity_threshold = max(0.01, min(0.99, float(config.embedding_similarity_threshold)))
    config.embedding_group_min_size = max(2, int(config.embedding_group_min_size))
    if config.grouping_method not in {"embedding", "embedding_openclip", "ahash_fallback", "mock_embedding"}:
        config.grouping_method = "embedding"
    if config.grouping_strategy not in {
        "connected_components",
        "complete_linkage",
        "average_linkage",
        "sequence_constrained",
    }:
        config.grouping_strategy = "complete_linkage"
    config.group_min_similarity_threshold = max(0.01, min(0.99, float(config.group_min_similarity_threshold)))
    config.max_embedding_group_size = max(2, int(config.max_embedding_group_size))
    config.batch_size = _profile_batch_size(config.model_profile, config.batch_size)
    defaults = default_scoring_weights()
    if not isinstance(config.scoring_weights, dict):
        config.scoring_weights = defaults
    else:
        merged = defaults
        for strategy, values in config.scoring_weights.items():
            if isinstance(values, dict):
                merged[strategy] = {**merged.get(strategy, {}), **values}
        config.scoring_weights = merged
    return config


def save_config(config: AppConfig) -> None:
    CONFIG_PATH.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")


def update_config(values: dict[str, Any]) -> AppConfig:
    config = load_config()
    for key, value in values.items():
        if hasattr(config, key):
            setattr(config, key, value)
    save_config(config)
    return config


def default_scoring_weights() -> dict[str, dict[str, float]]:
    return {
        "delivery": {
            "visual_quality": 0.35,
            "semantic": 0.25,
            "relative": 0.25,
            "preference": 0.15,
        },
        "learned": {
            "visual_quality": 0.25,
            "semantic": 0.20,
            "relative": 0.20,
            "preference": 0.35,
        },
        "portfolio": {
            "visual_quality": 0.30,
            "semantic": 0.30,
            "relative": 0.25,
            "preference": 0.15,
        },
        "conservative": {
            "visual_quality": 0.30,
            "semantic": 0.25,
            "relative": 0.20,
            "preference": 0.25,
        },
        "strict": {
            "visual_quality": 0.40,
            "semantic": 0.25,
            "relative": 0.25,
            "preference": 0.10,
        },
    }


def _profile_batch_size(profile: str, current: int) -> int:
    current = max(1, int(current))
    if profile == "fast":
        return max(current, 32)
    if profile == "accurate":
        return min(current, 12)
    if profile == "ultra":
        return min(current, 8)
    return current
