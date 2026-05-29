from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app.model_pipeline.embedding_types import EmbeddingResult


class EmbeddingExtractor(ABC):
    model_name = "base"
    model_version = "v1"
    method = "embedding"

    @abstractmethod
    def extract_batch(self, image_paths: list[Path]) -> list[EmbeddingResult]:
        raise NotImplementedError


class MockEmbeddingExtractor(EmbeddingExtractor):
    model_name = "mock_embedding"
    model_version = "v1"
    method = "mock_embedding"

    def __init__(self, dim: int = 16):
        self.dim = max(4, int(dim))

    def extract_batch(self, image_paths: list[Path]) -> list[EmbeddingResult]:
        return [self._result_for(Path(path)) for path in image_paths]

    def _result_for(self, path: Path) -> EmbeddingResult:
        stem = path.stem.lower()
        vector = np.zeros(self.dim, dtype="float32")
        if stem.startswith("same_") or stem.startswith("near_"):
            vector[0] = 1.0
            vector[2] = _small_jitter(stem)
        elif stem.startswith("pair_"):
            vector[0] = 0.96
            vector[1] = 0.12
            vector[2] = _small_jitter(stem)
        else:
            vector[1] = 1.0
            vector[3] = _small_jitter(stem)
        return EmbeddingResult(
            image_path=path,
            vector=normalize_vector(vector).tolist(),
            model_name=self.model_name,
            model_version=self.model_version,
            method=self.method,
            source="mock",
        )


class FallbackEmbeddingExtractor(EmbeddingExtractor):
    model_name = "ahash_color_fallback"
    model_version = "v1"
    method = "ahash_fallback"

    def extract_batch(self, image_paths: list[Path]) -> list[EmbeddingResult]:
        results: list[EmbeddingResult] = []
        for path in image_paths:
            vector = fallback_image_vector(path)
            results.append(
                EmbeddingResult(
                    image_path=Path(path),
                    vector=vector.tolist(),
                    model_name=self.model_name,
                    model_version=self.model_version,
                    method=self.method,
                    source="fallback",
                )
            )
        return results


def normalize_vector(vector: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(vector, dtype="float32").reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        return array
    return array / norm


def fallback_image_vector(path: Path | str) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        gray = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        gray_values = np.asarray(gray, dtype="float32").reshape(-1)
        bits = np.where(gray_values >= float(gray_values.mean()), 1.0, -1.0)
        small_rgb = np.asarray(image.resize((64, 64), Image.Resampling.BILINEAR), dtype="float32") / 255.0
        hist_parts = []
        for channel in range(3):
            hist, _ = np.histogram(small_rgb[:, :, channel], bins=8, range=(0.0, 1.0), density=False)
            hist = hist.astype("float32")
            hist_parts.append(hist / max(1.0, float(hist.sum())))
        aspect = np.asarray([image.width / max(1.0, image.height), image.height / max(1.0, image.width)], dtype="float32")
    return normalize_vector(np.concatenate([bits, *hist_parts, aspect]))


def _small_jitter(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8", "ignore")).digest()
    return (digest[0] / 255.0 - 0.5) * 0.02
