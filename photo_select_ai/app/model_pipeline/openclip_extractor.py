from __future__ import annotations

from contextlib import nullcontext
from importlib.util import find_spec
from pathlib import Path

from PIL import Image, ImageOps

from app.core.config import AppConfig, load_config
from app.core.model_manager import ModelManager, get_openclip_spec
from app.model_pipeline.embedding_extractor import EmbeddingExtractor, normalize_vector
from app.model_pipeline.embedding_types import EmbeddingResult
from app.model_pipeline.runtime_detector import RuntimeInfo, detect_model_runtime


class OpenClipExtractorUnavailable(RuntimeError):
    pass


class OpenClipEmbeddingExtractor(EmbeddingExtractor):
    method = "embedding_openclip"
    preprocess_version = "openclip_preprocess_v1"

    def __init__(
        self,
        config: AppConfig | None = None,
        runtime: RuntimeInfo | None = None,
        manager: ModelManager | None = None,
    ):
        self.config = config or load_config()
        self.runtime = runtime or detect_model_runtime(self.config.use_gpu)
        self.manager = manager or ModelManager(self.config)
        self._torch = None
        self._open_clip = None
        self._model = None
        self._preprocess = None
        self._device = "cpu"
        self._use_fp16 = False
        self._spec = get_openclip_spec(_openclip_model_id(self.config), self.config.model_profile)
        self.model_name = f"openclip_{self._spec.model_name}_{self._spec.pretrained}"
        self.model_version = f"{self._spec.id}:{self._spec.pretrained}:{self.preprocess_version}"
        self.batch_size = max(1, int(getattr(self.config, "embedding_batch_size", 0) or self.config.batch_size))
        self._initialize()

    def extract_batch(self, image_paths: list[Path]) -> list[EmbeddingResult]:
        if not image_paths:
            return []
        results: list[EmbeddingResult] = []
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = [Path(path) for path in image_paths[start : start + self.batch_size]]
            results.extend(self._extract_chunk(batch_paths))
        return results

    def _initialize(self) -> None:
        if find_spec("torch") is None:
            raise OpenClipExtractorUnavailable("torch is not installed")
        if find_spec("open_clip") is None:
            raise OpenClipExtractorUnavailable("open_clip_torch is not installed")
        marker = self.manager.semantic_marker(self._spec)
        if not marker.exists():
            raise OpenClipExtractorUnavailable(f"OpenCLIP model marker is missing: {marker}")

        import open_clip
        import torch

        self._torch = torch
        self._open_clip = open_clip
        requested = getattr(self.config, "embedding_device", "auto")
        if requested == "cuda" and not torch.cuda.is_available():
            raise OpenClipExtractorUnavailable("embedding_device=cuda but CUDA is unavailable")
        if requested == "cpu":
            self._device = torch.device("cpu")
        elif torch.cuda.is_available() and self.config.use_gpu:
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        self._use_fp16 = bool(self.config.use_fp16 and getattr(self._device, "type", "") == "cuda")
        try:
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self._spec.model_name,
                pretrained=self._spec.pretrained,
                device=self._device,
                cache_dir=str(self.manager.semantic_cache_dir(self._spec)),
            )
            self._model.eval()
            if self._use_fp16:
                self._model = self._model.half()
        except Exception as exc:
            raise OpenClipExtractorUnavailable(f"OpenCLIP model load failed: {exc}") from exc

    def _extract_chunk(self, paths: list[Path]) -> list[EmbeddingResult]:
        tensors = []
        for path in paths:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                tensors.append(self._preprocess(image))
        batch = self._torch.stack(tensors).to(self._device, non_blocking=bool(self.config.pin_memory))
        if self._use_fp16:
            batch = batch.half()
        with self._torch.no_grad():
            with self._autocast_context():
                features = self._model.encode_image(batch)
            features = features.float()
            features = features / features.norm(dim=-1, keepdim=True)

        results: list[EmbeddingResult] = []
        for path, feature in zip(paths, features):
            vector = normalize_vector(feature.detach().cpu().numpy()).tolist()
            results.append(
                EmbeddingResult(
                    image_path=path,
                    vector=vector,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    method=self.method,
                    dim=len(vector),
                    source="openclip",
                )
            )
        return results

    def _autocast_context(self):
        if self._use_fp16 and getattr(self._device, "type", "") == "cuda" and hasattr(self._torch, "autocast"):
            return self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
        return nullcontext()


def _openclip_model_id(config: AppConfig) -> str:
    configured = str(getattr(config, "embedding_model_name", "") or "")
    if configured.startswith("openclip_"):
        return configured
    return str(config.semantic_model)
