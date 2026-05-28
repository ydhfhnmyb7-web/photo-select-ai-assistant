from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.core.app_logging import get_logger
from app.core.config import AppConfig, PROJECT_ROOT, save_config
from app.core.gpu import detect_ai_environment


MODELS_DIR = PROJECT_ROOT / "models"
OPENCLIP_DIR = MODELS_DIR / "openclip"
YOLO_DIR = MODELS_DIR / "yolo"
MODEL_CACHE_DIR = MODELS_DIR / "cache"
EMBEDDING_CACHE_DIR = PROJECT_ROOT / "cache" / "embeddings"


ProgressCallback = Callable[[str, int, str], None]
CancelCallback = Callable[[], bool]


@dataclass(frozen=True)
class OpenClipSpec:
    id: str
    model_name: str
    pretrained: str
    display_name: str
    profile: str


@dataclass(frozen=True)
class YoloSpec:
    id: str
    filename: str
    display_name: str
    profile: str


@dataclass
class ModelRuntimeStatus:
    semantic_model: str
    detector_model: str
    semantic_downloaded: bool
    detector_downloaded: bool
    semantic_status: str
    detector_status: str
    device: str
    gpu_name: str
    batch_size: int
    use_fp16: bool
    model_profile: str
    model_size_text: str
    cache_dir: str
    embedding_cache_count: int
    message: str


OPENCLIP_SPECS: dict[str, OpenClipSpec] = {
    "openclip_vit_b_32": OpenClipSpec(
        id="openclip_vit_b_32",
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        display_name="OpenCLIP ViT-B/32",
        profile="fast",
    ),
    "openclip_vit_l_14": OpenClipSpec(
        id="openclip_vit_l_14",
        model_name="ViT-L-14",
        pretrained="laion2b_s32b_b82k",
        display_name="OpenCLIP ViT-L/14",
        profile="balanced",
    ),
    "openclip_vit_h_14": OpenClipSpec(
        id="openclip_vit_h_14",
        model_name="ViT-H-14",
        pretrained="laion2b_s32b_b79k",
        display_name="OpenCLIP ViT-H/14",
        profile="accurate",
    ),
}


YOLO_SPECS: dict[str, YoloSpec] = {
    "yolo11n": YoloSpec("yolo11n", "yolo11n.pt", "YOLO11n", "fast"),
    "yolo11s": YoloSpec("yolo11s", "yolo11s.pt", "YOLO11s", "fast"),
    "yolo11m": YoloSpec("yolo11m", "yolo11m.pt", "YOLO11m", "balanced"),
    "yolo11l": YoloSpec("yolo11l", "yolo11l.pt", "YOLO11l", "balanced"),
    "yolo11x": YoloSpec("yolo11x", "yolo11x.pt", "YOLO11x", "accurate"),
}


PROFILE_DEFAULTS = {
    "fast": ("openclip_vit_b_32", "yolo11n", 32),
    "balanced": ("openclip_vit_l_14", "yolo11m", 16),
    "accurate": ("openclip_vit_h_14", "yolo11x", 8),
    "ultra": ("openclip_vit_h_14", "yolo11x", 6),
}


class ModelManager:
    """Model directory, download, runtime status, and embedding cache helpers."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger()
        ensure_model_directories()

    def status(self) -> ModelRuntimeStatus:
        env = detect_ai_environment(self.config.use_gpu)
        semantic_spec = get_openclip_spec(self.config.semantic_model, self.config.model_profile)
        detector_spec = get_yolo_spec(self.config.detector_model, self.config.model_profile)
        semantic_downloaded = self.semantic_marker(semantic_spec).exists()
        detector_downloaded = self.detector_path(detector_spec).exists()
        model_size = _folder_size_text(MODELS_DIR)
        embedding_count = len(list(EMBEDDING_CACHE_DIR.glob("*.pkl"))) if EMBEDDING_CACHE_DIR.exists() else 0
        semantic_status = "已下载" if semantic_downloaded else "模型未下载"
        detector_status = "已下载" if detector_downloaded else "模型未下载"
        if not env.clip_installed:
            semantic_status = "依赖缺失：open_clip_torch"
        if not _module_available("ultralytics"):
            detector_status = "依赖缺失：ultralytics"
        return ModelRuntimeStatus(
            semantic_model=semantic_spec.display_name,
            detector_model=detector_spec.display_name,
            semantic_downloaded=semantic_downloaded,
            detector_downloaded=detector_downloaded,
            semantic_status=semantic_status,
            detector_status=detector_status,
            device=env.source,
            gpu_name=env.gpu_name,
            batch_size=self.config.batch_size,
            use_fp16=self.config.use_fp16,
            model_profile=self.config.model_profile,
            model_size_text=model_size,
            cache_dir=str(MODEL_CACHE_DIR),
            embedding_cache_count=embedding_count,
            message=env.message,
        )

    def apply_profile(self, profile: str) -> AppConfig:
        semantic_id, detector_id, batch_size = PROFILE_DEFAULTS.get(profile, PROFILE_DEFAULTS["balanced"])
        self.config.model_profile = profile
        self.config.semantic_model = semantic_id
        self.config.detector_model = detector_id
        self.config.batch_size = batch_size
        save_config(self.config)
        return self.config

    def semantic_marker(self, spec: OpenClipSpec | None = None) -> Path:
        spec = spec or get_openclip_spec(self.config.semantic_model, self.config.model_profile)
        return OPENCLIP_DIR / spec.id / "downloaded.json"

    def detector_path(self, spec: YoloSpec | None = None) -> Path:
        spec = spec or get_yolo_spec(self.config.detector_model, self.config.model_profile)
        return YOLO_DIR / spec.filename

    def semantic_cache_dir(self, spec: OpenClipSpec | None = None) -> Path:
        spec = spec or get_openclip_spec(self.config.semantic_model, self.config.model_profile)
        path = OPENCLIP_DIR / spec.id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def download_recommended_models(
        self,
        progress_callback: ProgressCallback | None = None,
        cancel_callback: CancelCallback | None = None,
    ) -> ModelRuntimeStatus:
        ensure_model_directories()
        semantic_spec = get_openclip_spec(self.config.semantic_model, self.config.model_profile)
        detector_spec = get_yolo_spec(self.config.detector_model, self.config.model_profile)
        if cancel_callback and cancel_callback():
            raise RuntimeError("模型下载已取消")

        self._emit(progress_callback, "准备下载模型", 5, semantic_spec.display_name)
        self._download_openclip(semantic_spec, progress_callback, cancel_callback)
        if cancel_callback and cancel_callback():
            raise RuntimeError("模型下载已取消")
        self._download_yolo(detector_spec, progress_callback, cancel_callback)
        self.config.auto_download_models = False
        save_config(self.config)
        self._emit(progress_callback, "模型下载完成", 100, "OpenCLIP / YOLO")
        return self.status()

    def _download_openclip(
        self,
        spec: OpenClipSpec,
        progress_callback: ProgressCallback | None,
        cancel_callback: CancelCallback | None,
    ) -> None:
        marker = self.semantic_marker(spec)
        if marker.exists():
            self._emit(progress_callback, "语义模型已存在", 40, spec.display_name)
            return
        if not _module_available("open_clip"):
            raise RuntimeError("未安装 open_clip_torch，请先运行 python -m pip install open_clip_torch")
        try:
            self._emit(progress_callback, "下载语义模型", 20, spec.display_name)
            import open_clip

            cache_dir = self.semantic_cache_dir(spec)
            model, _, _preprocess = open_clip.create_model_and_transforms(
                spec.model_name,
                pretrained=spec.pretrained,
                cache_dir=str(cache_dir),
            )
            del model
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                json.dumps(
                    {
                        "id": spec.id,
                        "model_name": spec.model_name,
                        "pretrained": spec.pretrained,
                        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            self._emit(progress_callback, "语义模型下载完成", 50, spec.display_name)
        except Exception as exc:
            self.logger.exception("OpenCLIP 模型下载失败")
            raise RuntimeError("模型下载失败，请检查网络或代理。") from exc

    def _download_yolo(
        self,
        spec: YoloSpec,
        progress_callback: ProgressCallback | None,
        cancel_callback: CancelCallback | None,
    ) -> None:
        target = self.detector_path(spec)
        if target.exists():
            self._emit(progress_callback, "检测模型已存在", 85, spec.display_name)
            return
        if not _module_available("ultralytics"):
            raise RuntimeError("未安装 ultralytics，请先运行 python -m pip install -U ultralytics")
        try:
            self._emit(progress_callback, "下载检测模型", 60, spec.display_name)
            from ultralytics import YOLO

            YOLO_DIR.mkdir(parents=True, exist_ok=True)
            old_cwd = Path.cwd()
            try:
                os.chdir(YOLO_DIR)
                YOLO(spec.filename)
            finally:
                os.chdir(old_cwd)
            if not target.exists():
                found = next(YOLO_DIR.rglob(spec.filename), None)
                if found and found != target:
                    shutil.copy2(found, target)
            if not target.exists():
                raise FileNotFoundError(target)
            self._emit(progress_callback, "检测模型下载完成", 90, spec.display_name)
        except Exception as exc:
            self.logger.exception("YOLO 模型下载失败")
            raise RuntimeError("模型下载失败，请检查网络或代理。") from exc

    def clear_model_cache(self) -> None:
        _clear_directory(MODEL_CACHE_DIR)

    def clear_embedding_cache(self) -> None:
        _clear_directory(EMBEDDING_CACHE_DIR)

    def _emit(self, callback: ProgressCallback | None, stage: str, percent: int, detail: str) -> None:
        if callback:
            callback(stage, percent, detail)


def ensure_model_directories() -> None:
    for directory in [MODELS_DIR, OPENCLIP_DIR, YOLO_DIR, MODEL_CACHE_DIR, EMBEDDING_CACHE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_openclip_spec(model_id: str, profile: str = "balanced") -> OpenClipSpec:
    if model_id in OPENCLIP_SPECS:
        return OPENCLIP_SPECS[model_id]
    fallback_id = PROFILE_DEFAULTS.get(profile, PROFILE_DEFAULTS["balanced"])[0]
    return OPENCLIP_SPECS[fallback_id]


def get_yolo_spec(model_id: str, profile: str = "balanced") -> YoloSpec:
    if model_id in YOLO_SPECS:
        return YOLO_SPECS[model_id]
    fallback_id = PROFILE_DEFAULTS.get(profile, PROFILE_DEFAULTS["balanced"])[1]
    return YOLO_SPECS[fallback_id]


def embedding_cache_path(image_path: Path, config: AppConfig) -> tuple[Path, str]:
    spec = get_openclip_spec(config.semantic_model, config.model_profile)
    stat = image_path.stat()
    key_source = (
        f"{image_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:"
        f"{spec.id}:{spec.model_name}:{spec.pretrained}"
    )
    digest = hashlib.sha1(key_source.encode("utf-8", "ignore")).hexdigest()
    return EMBEDDING_CACHE_DIR / f"{digest}.pkl", digest


def load_embedding_cache(image_path: Path, config: AppConfig) -> tuple[list[float] | None, Path, str]:
    path, digest = embedding_cache_path(image_path, config)
    if not path.exists():
        return None, path, digest
    try:
        with path.open("rb") as file:
            payload = pickle.load(file)
        vector = payload.get("embedding")
        if isinstance(vector, list) and vector:
            return [float(value) for value in vector], path, digest
    except Exception:
        get_logger().exception("embedding 缓存读取失败，已准备重算：%s", path)
    return None, path, digest


def save_embedding_cache(image_path: Path, config: AppConfig, embedding: list[float]) -> tuple[Path, str]:
    ensure_model_directories()
    path, digest = embedding_cache_path(image_path, config)
    spec = get_openclip_spec(config.semantic_model, config.model_profile)
    payload = {
        "embedding": [float(value) for value in embedding],
        "image_path": str(image_path),
        "model_id": spec.id,
        "model_name": spec.model_name,
        "pretrained": spec.pretrained,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_feature_hash": digest,
    }
    try:
        with path.open("wb") as file:
            pickle.dump(payload, file)
    except Exception:
        get_logger().exception("embedding 缓存写入失败：%s", path)
    return path, digest


def gpu_memory_text() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "CPU"
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        used = (total_bytes - free_bytes) / (1024**3)
        total = total_bytes / (1024**3)
        return f"{used:.1f} / {total:.1f} GB"
    except Exception:
        return "-"


def auto_batch_size_for_profile(config: AppConfig) -> int:
    if not config.use_gpu:
        return min(config.batch_size, 8)
    try:
        import torch

        if not torch.cuda.is_available():
            return min(config.batch_size, 8)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        total_gb = 0
    if config.model_profile == "fast":
        wanted = 32
    elif config.model_profile == "accurate":
        wanted = 8 if total_gb < 16 else 12
    elif config.model_profile == "ultra":
        wanted = 4 if total_gb < 16 else 8
    else:
        wanted = 8 if total_gb and total_gb < 10 else 32
    return max(1, min(config.batch_size, wanted))


def _folder_size_text(path: Path) -> str:
    if not path.exists():
        return "0 MB"
    size = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            try:
                size += file_path.stat().st_size
            except OSError:
                continue
    if size >= 1024**3:
        return f"{size / (1024**3):.2f} GB"
    return f"{size / (1024**2):.1f} MB"


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _clear_directory(path: Path) -> None:
    path = Path(path)
    root = PROJECT_ROOT.resolve()
    resolved = path.resolve()
    if root not in [resolved, *resolved.parents]:
        raise ValueError(f"拒绝清理项目外目录：{path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
