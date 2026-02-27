"""Unit tests for deepface_analytics/storage.py."""

from typing import Any, Dict

import pytest

from deepface_analytics.storage import FaceStorage


@pytest.fixture
def storage() -> FaceStorage:
    return FaceStorage()


def test_save_and_load_known_faces_roundtrip(
    storage: FaceStorage, tmp_known_faces_dir: Any
) -> None:
    path = str(tmp_known_faces_dir / "known_faces.json")
    data = {"face_001": {"name": "Alice", "first_seen": "2024-01-01 00:00:00"}}
    storage.save_known_faces(data, path)
    loaded = storage.load_known_faces(path)
    assert loaded == data


def test_load_returns_empty_dict_when_file_missing(
    storage: FaceStorage, tmp_known_faces_dir: Any
) -> None:
    path = str(tmp_known_faces_dir / "nonexistent.json")
    result = storage.load_known_faces(path)
    assert result == {}


def test_prune_keeps_max_entries(storage: FaceStorage) -> None:
    data: Dict[str, Any] = {
        f"face_{i:04d}": {"first_seen": f"2024-01-{(i % 28) + 1:02d} 00:00:00"}
        for i in range(600)
    }
    pruned = storage.prune_known_faces(data, max_entries=500)
    assert len(pruned) == 500


def test_prune_removes_oldest_by_first_seen(storage: FaceStorage) -> None:
    data: Dict[str, Any] = {
        "old_face": {"first_seen": "2020-01-01 00:00:00"},
        "new_face": {"first_seen": "2024-01-01 00:00:00"},
    }
    pruned = storage.prune_known_faces(data, max_entries=1)
    assert "new_face" in pruned
    assert "old_face" not in pruned


def test_generate_session_json_contains_required_keys(
    storage: FaceStorage, sample_known_faces: Dict[str, Any]
) -> None:
    import time

    stats = {"total_detected_faces": 10}
    result = storage.generate_session_json(sample_known_faces, stats, time.time())
    for key in ("timestamp", "date", "time", "session_info", "people"):
        assert key in result
