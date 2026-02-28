"""Unit tests for deepface_analytics/tracker.py."""

import time
from typing import Any

import numpy as np
import pytest

from deepface_analytics.tracker import (
    FACE_TRACKING_THRESHOLD,
    RECOGNITION_THRESHOLD,
    FaceTracker,
)


@pytest.fixture
def tracker() -> FaceTracker:
    return FaceTracker()


def _make_encoding(value: float, dim: int = 128) -> Any:
    return np.full((dim,), value, dtype=np.float64)


def test_is_same_face_returns_face_id_when_distance_below_threshold(
    tracker: FaceTracker,
) -> None:
    enc = _make_encoding(0.0)
    tracker.add_face_encoding("face_001", enc)
    result = tracker.is_same_as_known_face(_make_encoding(0.001))
    assert result == "face_001"


def test_is_same_face_returns_none_when_distance_above_threshold(
    tracker: FaceTracker,
) -> None:
    enc = _make_encoding(0.0)
    tracker.add_face_encoding("face_001", enc)
    # Make distance >> RECOGNITION_THRESHOLD
    result = tracker.is_same_as_known_face(_make_encoding(10.0))
    assert result is None


def test_vectorized_recognition_matches_loop_behavior(
    tracker: FaceTracker,
) -> None:
    """Vectorized result must equal a reference loop implementation for N=10 faces."""
    dim = 64
    encodings = [np.random.rand(dim) for _ in range(10)]
    for i, enc in enumerate(encodings):
        tracker.add_face_encoding(f"face_{i:03d}", enc)

    query = encodings[3].copy()  # exact match expected
    result = tracker.is_same_as_known_face(query)

    # Reference: Python loop
    min_dist = float("inf")
    best_id = None
    for i, enc in enumerate(encodings):
        d = float(np.linalg.norm(enc - query))
        if d < min_dist:
            min_dist = d
            best_id = f"face_{i:03d}"
    expected = best_id if min_dist < RECOGNITION_THRESHOLD else None

    assert result == expected


def test_sliding_window_returns_most_common_emotion(tracker: FaceTracker) -> None:
    for _ in range(20):
        tracker.update_recent_emotion("happy")
    for _ in range(5):
        tracker.update_recent_emotion("sad")
    result = tracker.update_recent_emotion("happy")
    assert result == "happy"


def test_cleanup_removes_inactive_faces(tracker: FaceTracker) -> None:
    old_time = time.time() - FACE_TRACKING_THRESHOLD - 1.0
    tracked = {
        "active_face": {"last_seen": time.time()},
        "inactive_face": {"last_seen": old_time},
    }
    tracker.cleanup_trackers(tracked)
    assert "inactive_face" not in tracked
    assert "active_face" in tracked
