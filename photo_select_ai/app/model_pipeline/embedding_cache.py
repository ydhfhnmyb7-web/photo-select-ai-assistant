from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from app.core.config import PROJECT_ROOT
from app.model_pipeline.embedding_types import EmbeddingResult


DEFAULT_EMBEDDING_CACHE_DIR = PROJECT_ROOT / "cache" / "embeddings"


@dataclass
class CachedEmbedding:
    vector: list[float]
    cache_key: str
    cached_at: str
    path: Path
    dim: int


class EmbeddingCache:
    def __init__(self, root: Path | str = DEFAULT_EMBEDDING_CACHE_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def cache_key(self, image_path: Path | str, model_name: str, model_version: str, preprocess_version: str = "v1") -> str:
        path = Path(image_path)
        stat = path.stat()
        source = (
            f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:"
            f"{model_name}:{model_version}:{preprocess_version}"
        )
        return hashlib.sha1(source.encode("utf-8", "ignore")).hexdigest()

    def load(
        self,
        image_path: Path | str,
        model_name: str,
        model_version: str,
        preprocess_version: str = "v1",
    ) -> CachedEmbedding | None:
        key = self.cache_key(image_path, model_name, model_version, preprocess_version)
        vector_path, meta_path = self._paths_for_key(model_name, key)
        if not vector_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            vector = np.load(vector_path).astype("float32")
            if vector.ndim != 1 or vector.size == 0:
                return None
            return CachedEmbedding(
                vector=[float(value) for value in vector.tolist()],
                cache_key=key,
                cached_at=str(meta.get("cached_at") or ""),
                path=vector_path,
                dim=int(vector.size),
            )
        except Exception:
            return None

    def save(
        self,
        embedding: EmbeddingResult,
        preprocess_version: str = "v1",
    ) -> EmbeddingResult:
        key = embedding.cache_key or self.cache_key(
            embedding.image_path,
            embedding.model_name,
            embedding.model_version,
            preprocess_version,
        )
        vector_path, meta_path = self._paths_for_key(embedding.model_name, key)
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        cached_at = datetime.now().isoformat(timespec="seconds")
        np.save(vector_path, np.asarray(embedding.vector, dtype="float32"))
        meta_path.write_text(
            json.dumps(
                {
                    "image_path": str(embedding.image_path),
                    "model_name": embedding.model_name,
                    "model_version": embedding.model_version,
                    "method": embedding.method,
                    "cache_key": key,
                    "cached_at": cached_at,
                    "dim": len(embedding.vector),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        embedding.cache_key = key
        embedding.cached_at = cached_at
        embedding.dim = len(embedding.vector)
        return embedding

    def delete(
        self,
        image_path: Path | str,
        model_name: str,
        model_version: str,
        preprocess_version: str = "v1",
    ) -> None:
        key = self.cache_key(image_path, model_name, model_version, preprocess_version)
        vector_path, meta_path = self._paths_for_key(model_name, key)
        for path in [vector_path, meta_path]:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                continue

    def _paths_for_key(self, model_name: str, key: str) -> tuple[Path, Path]:
        model_dir = self.root / _safe_segment(model_name)
        return model_dir / f"{key}.npy", model_dir / f"{key}.json"


def _safe_segment(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    return safe or "unknown_model"
