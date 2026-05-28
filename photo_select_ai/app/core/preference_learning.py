from __future__ import annotations

import json
import pickle
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.core.config import PROJECT_ROOT
from app.core.mvp_models import PhotoItem


DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLES_PATH = DATA_DIR / "user_preference_samples.jsonl"
MODEL_PATH = MODELS_DIR / "user_preference_model.pkl"
MODEL_VERSION = "local-preference-v1"


@dataclass
class PreferencePrediction:
    label: str
    score: float
    reason: str
    similar_sample_count: int


def record_preference_sample(item: PhotoItem, user_label: str, reason: str = "") -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "path": str(item.path),
        "filename": item.filename,
        "user_label": user_label,
        "reason": reason,
        "features": item_feature_vector(item),
        "ai_primary_category": item.ai_primary_category,
        "ai_secondary_category": item.ai_secondary_category,
        "ai_quality_tags": item.ai_quality_tags,
        "manual_category": item.manual_category,
        "final_category": item.final_category,
        "similar_group_id": item.similar_group_id,
    }
    with SAMPLES_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(sample, ensure_ascii=False) + "\n")


def load_samples() -> list[dict]:
    if not SAMPLES_PATH.exists():
        return []
    samples = []
    for line in SAMPLES_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return samples


def preference_summary() -> dict:
    samples = load_samples()
    labels = [sample.get("user_label") or sample.get("manual_category") or "待定" for sample in samples]
    counts = Counter(labels)
    warnings = []
    if len(samples) < 50:
        warnings.append(f"样本太少：还差 {50 - len(samples)} 张可训练。")
    if len(counts) < 2 and samples:
        warnings.append("样本类别不均衡：至少需要两个以上类别。")
    for label in ["精选", "废片", "可用待修", "可用待裁切", "人工复核"]:
        if counts.get(label, 0) < 5:
            warnings.append(f"{label}样本偏少。")
    return {
        "total": len(samples),
        "counts": dict(counts),
        "can_train": len(samples) >= 50 and len(counts) >= 2,
        "warnings": warnings,
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "model_version": MODEL_VERSION,
    }


def preference_report() -> str:
    summary = preference_summary()
    lines = [
        "我的筛片风格报告",
        "",
        f"样本数量：{summary['total']}",
        f"是否达到训练条件：{'是' if summary['can_train'] else '否'}",
        f"模型版本：{summary['model_version']}",
        f"模型路径：{summary['model_path']}",
        "",
        "样本分布：",
    ]
    for label, count in sorted(summary["counts"].items()):
        lines.append(f"- {label}: {count} 张")
    if summary["warnings"]:
        lines.extend(["", "样本质量提示："])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    else:
        lines.extend(["", "样本质量提示：分布基本可用。"])
    lines.extend(
        [
            "",
            "当前学习方式：本地轻量偏好模型，不训练大模型。",
            "偏好模型只提供建议，不覆盖人工分类。",
        ]
    )
    return "\n".join(lines)


def train_preference_model() -> tuple[bool, str]:
    samples = load_samples()
    if len(samples) < 50:
        return False, f"样本不足：当前 {len(samples)} 张，至少需要 50 张人工标记样本。"

    labels = [sample.get("user_label") or sample.get("manual_category") or "待定" for sample in samples]
    label_counts = Counter(labels)
    vectors = [sample.get("features") or [] for sample in samples]
    sklearn_model = _train_sklearn_logistic(vectors, labels)
    if sklearn_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with MODEL_PATH.open("wb") as file:
            pickle.dump(
                {
                    "kind": "sklearn_logistic_regression",
                    "version": MODEL_VERSION,
                    "sample_count": len(samples),
                    "label_counts": dict(label_counts),
                    "model": sklearn_model["model"],
                    "accuracy": sklearn_model["accuracy"],
                    "trained_at": datetime.now().isoformat(timespec="seconds"),
                },
                file,
            )
        accuracy_text = f"，验证准确率 {sklearn_model['accuracy']:.2f}" if sklearn_model["accuracy"] >= 0 else ""
        return True, f"偏好模型已训练：{len(samples)} 个样本{accuracy_text}，保存到 {MODEL_PATH}"

    feature_sums: dict[str, list[float]] = {}
    feature_counts: dict[str, int] = {}
    for sample, label in zip(samples, labels):
        vector = sample.get("features") or []
        if not vector:
            continue
        feature_sums.setdefault(label, [0.0] * len(vector))
        feature_counts[label] = feature_counts.get(label, 0) + 1
        for index, value in enumerate(vector):
            feature_sums[label][index] += float(value)

    centroids = {
        label: [value / max(1, feature_counts[label]) for value in values]
        for label, values in feature_sums.items()
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as file:
        pickle.dump(
            {
                "version": MODEL_VERSION,
                "sample_count": len(samples),
                "label_counts": dict(label_counts),
                "kind": "centroid_knn_fallback",
                "centroids": centroids,
                "trained_at": datetime.now().isoformat(timespec="seconds"),
            },
            file,
        )
    return True, f"偏好模型已训练：{len(samples)} 个样本，保存到 {MODEL_PATH}"


def predict_preference(item: PhotoItem) -> PreferencePrediction:
    if not MODEL_PATH.exists():
        return PreferencePrediction("", 0.0, "尚未训练偏好模型。", 0)

    try:
        with MODEL_PATH.open("rb") as file:
            model = pickle.load(file)
    except Exception:
        return PreferencePrediction("", 0.0, "偏好模型读取失败。", 0)

    vector = item_feature_vector(item)
    if model.get("kind") == "sklearn_logistic_regression":
        try:
            estimator = model["model"]
            label = str(estimator.predict([vector])[0])
            if hasattr(estimator, "predict_proba"):
                score = float(max(estimator.predict_proba([vector])[0]))
            else:
                score = 0.65
            reason = f"根据你已标记的 {model.get('sample_count', 0)} 张样本，本图更接近“{label}”，偏好分 {score:.2f}。"
            return PreferencePrediction(label, score, reason, model.get("sample_count", 0))
        except Exception:
            return PreferencePrediction("", 0.0, "偏好模型推理失败。", model.get("sample_count", 0))

    best_label = ""
    best_distance = float("inf")
    for label, centroid in (model.get("centroids") or {}).items():
        distance = sum((float(a) - float(b)) ** 2 for a, b in zip(vector, centroid)) ** 0.5
        if distance < best_distance:
            best_distance = distance
            best_label = label

    if not best_label:
        return PreferencePrediction("", 0.0, "偏好模型暂无可用类别。", model.get("sample_count", 0))
    score = max(0.0, min(1.0, 1.0 / (1.0 + best_distance)))
    reason = f"根据你已标记的 {model.get('sample_count', 0)} 张样本，本图更接近“{best_label}”。"
    return PreferencePrediction(best_label, score, reason, model.get("sample_count", 0))


def clear_preference_model() -> None:
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()


def item_feature_vector(item: PhotoItem) -> list[float]:
    base = [
        item.width / 10000.0,
        item.height / 10000.0,
        item.file_size_mb / 100.0,
        item.ai_confidence,
        item.absolute_quality_score / 100.0,
        item.relative_quality_score / 100.0,
        item.semantic_score / 100.0,
        item.final_pick_score / 100.0,
        item.subject_area_ratio,
        item.subject_center_score,
        item.edge_cutoff_risk,
        item.detection_confidence,
        1.0 if item.similar_group_id else 0.0,
        1.0 if item.recommended_keep or item.recommended_in_group else 0.0,
        len(item.ai_quality_tags) / 20.0,
    ]
    return base + _embedding_feature_vector(item.embedding_path)


def _embedding_feature_vector(embedding_path: str, size: int = 32) -> list[float]:
    if not embedding_path:
        return [0.0] * size
    try:
        with Path(embedding_path).open("rb") as file:
            payload = pickle.load(file)
        vector = payload.get("embedding") or []
        values = [float(value) for value in vector[:size]]
        if len(values) < size:
            values.extend([0.0] * (size - len(values)))
        return values
    except Exception:
        return [0.0] * size


def _train_sklearn_logistic(vectors: list[list[float]], labels: list[str]) -> dict | None:
    if len(set(labels)) < 2:
        return None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
    except Exception:
        return None

    try:
        if len(vectors) >= 80:
            x_train, x_test, y_train, y_test = train_test_split(
                vectors,
                labels,
                test_size=0.2,
                random_state=42,
                stratify=labels if min(Counter(labels).values()) >= 2 else None,
            )
        else:
            x_train, x_test, y_train, y_test = vectors, [], labels, []
        model = LogisticRegression(max_iter=800, class_weight="balanced")
        model.fit(x_train, y_train)
        accuracy = float(accuracy_score(y_test, model.predict(x_test))) if x_test else -1.0
        return {"model": model, "accuracy": accuracy}
    except Exception:
        return None
