"""Unit tests for deepface_analytics/detector.py."""

from typing import Any, List

import pytest

from deepface_analytics.detector import MIN_FACE_SIZE, BoundingBox, FaceDetector


@pytest.fixture
def detector() -> FaceDetector:
    return FaceDetector()


def test_faces_smaller_than_min_size_are_filtered(detector: FaceDetector) -> None:
    """Bounding boxes smaller than MIN_FACE_SIZE must be excluded."""
    small_faces: List[BoundingBox] = [
        (0, 0, MIN_FACE_SIZE[0] - 1, MIN_FACE_SIZE[1] - 1),
        (10, 10, 10, 10),
    ]
    result = detector.filter_faces(small_faces)
    assert result == []


def test_filter_keeps_faces_at_min_size(detector: FaceDetector) -> None:
    """Bounding boxes exactly at MIN_FACE_SIZE should be kept."""
    ok_face: List[BoundingBox] = [(0, 0, MIN_FACE_SIZE[0], MIN_FACE_SIZE[1])]
    result = detector.filter_faces(ok_face)
    assert result == ok_face


def test_bounding_box_expansion_stays_within_frame_bounds(
    detector: FaceDetector,
) -> None:
    """Expansion near frame edge must clamp to frame boundaries."""
    frame_shape = (480, 640, 3)
    # Box in top-left corner — expansion would go negative
    x, y, w, h = detector.expand_bounding_box(2, 2, 100, 100, frame_shape)
    assert x >= 0
    assert y >= 0
    # Box near bottom-right corner — expansion must not exceed frame
    x2, y2, w2, h2 = detector.expand_bounding_box(600, 440, 40, 40, frame_shape)
    assert x2 + w2 <= 640
    assert y2 + h2 <= 480


def test_detect_faces_returns_empty_list_on_blank_frame(
    detector: FaceDetector, synthetic_frame: Any
) -> None:
    """A blank (all-zero) frame should yield no face detections."""
    result = detector.detect_faces(synthetic_frame)
    assert isinstance(result, list)
    assert result == []
