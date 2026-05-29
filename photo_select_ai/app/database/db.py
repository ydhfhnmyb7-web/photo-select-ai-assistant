from __future__ import annotations

import sqlite3
from pathlib import Path

from app.core.app_logging import get_logger
from app.core.mvp_models import PhotoItem


class PhotoRepository:
    """SQLite persistence with separated manual, AI, and final classification fields."""

    COLUMNS: dict[str, str] = {
        "path": "TEXT PRIMARY KEY",
        "filename": "TEXT NOT NULL DEFAULT ''",
        "thumbnail_path": "TEXT DEFAULT ''",
        "width": "INTEGER NOT NULL DEFAULT 0",
        "height": "INTEGER NOT NULL DEFAULT 0",
        "file_size_mb": "REAL NOT NULL DEFAULT 0",
        "taken_at": "TEXT DEFAULT ''",
        "manual_category": "TEXT DEFAULT '备选'",
        "manual_override": "INTEGER DEFAULT 0",
        "final_category": "TEXT DEFAULT ''",
        "user_note": "TEXT DEFAULT ''",
        "ai_primary_category": "TEXT DEFAULT ''",
        "ai_secondary_category": "TEXT DEFAULT ''",
        "ai_quality_tags": "TEXT DEFAULT '[]'",
        "ai_confidence": "REAL DEFAULT 0",
        "ai_source": "TEXT DEFAULT ''",
        "people_label": "TEXT DEFAULT ''",
        "people_count": "INTEGER DEFAULT 0",
        "shooting_type": "TEXT DEFAULT ''",
        "appearance_tags": "TEXT DEFAULT '[]'",
        "similar_group_id": "TEXT DEFAULT ''",
        "similar_group_size": "INTEGER DEFAULT 0",
        "recommended_keep": "INTEGER DEFAULT 0",
        "user_label": "TEXT DEFAULT ''",
        "user_label_time": "TEXT DEFAULT ''",
        "user_preference_score": "REAL DEFAULT 0",
        "user_preference_reason": "TEXT DEFAULT ''",
        "preference_model_version": "TEXT DEFAULT ''",
        "absolute_quality_score": "REAL DEFAULT 0",
        "relative_quality_score": "REAL DEFAULT 0",
        "group_rank": "INTEGER DEFAULT 0",
        "group_size": "INTEGER DEFAULT 0",
        "scene_group_id": "TEXT DEFAULT ''",
        "outfit_group_id": "TEXT DEFAULT ''",
        "pose_sequence_id": "TEXT DEFAULT ''",
        "recommended_in_group": "INTEGER DEFAULT 0",
        "ai_reason": "TEXT DEFAULT ''",
        "preference_reason": "TEXT DEFAULT ''",
        "final_reason": "TEXT DEFAULT ''",
        "embedding_path": "TEXT DEFAULT ''",
        "embedding_cached": "INTEGER DEFAULT 0",
        "ai_runtime_ms": "REAL DEFAULT 0",
        "ai_device": "TEXT DEFAULT ''",
        "ai_batch_size": "INTEGER DEFAULT 0",
        "semantic_score": "REAL DEFAULT 0",
        "semantic_confidence": "REAL DEFAULT 0",
        "top3_semantic_matches": "TEXT DEFAULT '[]'",
        "final_pick_score": "REAL DEFAULT 0",
        "detected_person_count": "INTEGER DEFAULT 0",
        "main_subject_bbox": "TEXT DEFAULT ''",
        "subject_area_ratio": "REAL DEFAULT 0",
        "subject_center_score": "REAL DEFAULT 0",
        "edge_cutoff_risk": "REAL DEFAULT 0",
        "group_photo_score": "REAL DEFAULT 0",
        "detection_confidence": "REAL DEFAULT 0",
        "embedding_model": "TEXT DEFAULT ''",
        "embedding_created_at": "TEXT DEFAULT ''",
        "image_feature_hash": "TEXT DEFAULT ''",
        "screening_reason": "TEXT DEFAULT ''",
        "style_label": "TEXT DEFAULT ''",
        "retouch_suggestion": "TEXT DEFAULT ''",
        "crop_suggestion": "TEXT DEFAULT ''",
        "portfolio_suggestion": "TEXT DEFAULT ''",
        "delivery_suggestion": "TEXT DEFAULT ''",
        "final_recommendation": "TEXT DEFAULT ''",
        "aesthetic_like_similarity": "REAL DEFAULT 0",
        "aesthetic_dislike_similarity": "REAL DEFAULT 0",
        "similar_reference_count": "INTEGER DEFAULT 0",
        "photo_type": "TEXT DEFAULT ''",
        "subtype": "TEXT DEFAULT ''",
        "quality_rating": "TEXT DEFAULT ''",
        "delivery_use": "TEXT DEFAULT '[]'",
        "issue_tags": "TEXT DEFAULT '[]'",
        "commercial_score": "REAL DEFAULT 0",
        "portfolio_score": "REAL DEFAULT 0",
        "ai_suggestion": "TEXT DEFAULT ''",
        "human_decision": "TEXT DEFAULT ''",
        "review_status": "TEXT DEFAULT '未审'",
        "best_in_group": "INTEGER DEFAULT 0",
        "review_note": "TEXT DEFAULT ''",
        "similar_group_rank": "INTEGER DEFAULT 0",
        "similar_group_status": "TEXT DEFAULT ''",
        "similarity_score": "REAL DEFAULT 0",
        "similar_group_note": "TEXT DEFAULT ''",
        "ai_recommended_best": "INTEGER DEFAULT 0",
        "ai_similarity_reason": "TEXT DEFAULT ''",
        "human_group_decision": "TEXT DEFAULT ''",
        "similarity_hash": "TEXT DEFAULT ''",
        "similarity_hash_mtime": "REAL DEFAULT 0",
        "auto_group_id": "TEXT DEFAULT ''",
        "auto_group_confidence": "REAL DEFAULT 0",
        "auto_group_reason": "TEXT DEFAULT ''",
        "embedding_model_name": "TEXT DEFAULT ''",
        "embedding_cached_at": "TEXT DEFAULT ''",
        "grouping_method": "TEXT DEFAULT ''",
        "embedding_cache_key": "TEXT DEFAULT ''",
        "embedding_dim": "INTEGER DEFAULT 0",
        "embedding_version": "TEXT DEFAULT ''",
        "auto_group_rank": "INTEGER DEFAULT 0",
        "auto_group_size": "INTEGER DEFAULT 0",
        "updated_at": "TEXT DEFAULT CURRENT_TIMESTAMP",
    }

    LEGACY_COLUMN_MAP = {
        "label": "manual_category",
        "notes": "user_note",
        "ai_recommended_category": "final_category",
        "ai_device": "ai_source",
        "ai_backend": "ai_source",
    }

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_or_migrate_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_or_migrate_schema(self) -> None:
        logger = get_logger()
        with self._connect() as connection:
            column_sql = ",\n                    ".join(
                f"{name} {definition}" for name, definition in self.COLUMNS.items()
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS photos (
                    {column_sql}
                )
                """
            )
            existing = {
                row["name"] for row in connection.execute("PRAGMA table_info(photos)").fetchall()
            }
            for name, definition in self.COLUMNS.items():
                if name not in existing:
                    logger.info("SQLite 迁移：新增字段 %s", name)
                    connection.execute(f"ALTER TABLE photos ADD COLUMN {name} {_alter_table_definition(definition)}")

            existing = {
                row["name"] for row in connection.execute("PRAGMA table_info(photos)").fetchall()
            }
            for legacy, current in self.LEGACY_COLUMN_MAP.items():
                if legacy in existing and current in existing:
                    if legacy == "label" and current == "manual_category":
                        connection.execute(
                            """
                            UPDATE photos
                            SET manual_category = label
                            WHERE label IS NOT NULL
                              AND label != ''
                              AND (manual_category IS NULL OR manual_category = '' OR manual_category = '备选')
                            """
                        )
                    else:
                        connection.execute(
                            f"""
                            UPDATE photos
                            SET {current} = COALESCE(NULLIF({current}, ''), {legacy})
                            WHERE {legacy} IS NOT NULL
                            """
                        )

            existing = {
                row["name"] for row in connection.execute("PRAGMA table_info(photos)").fetchall()
            }
            if "review_note" in existing and "user_note" in existing:
                connection.execute(
                    """
                    UPDATE photos
                    SET review_note = user_note
                    WHERE user_note IS NOT NULL
                      AND user_note != ''
                      AND (review_note IS NULL OR review_note = '')
                    """
                )

    def load_item_map(self) -> dict[str, dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM photos").fetchall()
        return {row["path"]: dict(row) for row in rows}

    def save_items(self, items: list[PhotoItem]) -> None:
        if not items:
            return

        columns = [column for column in self.COLUMNS if column != "updated_at"]
        insert_columns = ", ".join(columns + ["updated_at"])
        placeholders = ", ".join([f":{column}" for column in columns] + ["CURRENT_TIMESTAMP"])
        update_sql = ",\n                    ".join(
            f"{column} = excluded.{column}" for column in columns if column != "path"
        )

        with self._connect() as connection:
            connection.executemany(
                f"""
                INSERT INTO photos ({insert_columns})
                VALUES ({placeholders})
                ON CONFLICT(path) DO UPDATE SET
                    {update_sql},
                    updated_at = CURRENT_TIMESTAMP
                """,
                [item.to_db_row() for item in items],
            )

    def save_item(self, item: PhotoItem) -> None:
        self.save_items([item])


def _alter_table_definition(definition: str) -> str:
    """SQLite ALTER TABLE only accepts constant defaults for new columns."""
    if "CURRENT_TIMESTAMP" in definition.upper():
        return "TEXT DEFAULT ''"
    return definition
