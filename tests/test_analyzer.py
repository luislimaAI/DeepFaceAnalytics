"""Unit tests for deepface_analytics/analyzer.py."""

from typing import Any
from unittest.mock import patch

import pytest

from deepface_analytics.analyzer import FaceAnalyzer


@pytest.fixture
def analyzer() -> FaceAnalyzer:
    return FaceAnalyzer(ttl=2.0)


def test_cache_hit_does_not_call_deepface(
    analyzer: FaceAnalyzer,
    synthetic_face_crop: Any,
    mock_deepface: Any,
) -> None:
    """Second call within TTL must reuse cache without calling DeepFace again."""
    analyzer.analyze_face(synthetic_face_crop, "face_001")
    analyzer.analyze_face(synthetic_face_crop, "face_001")
    assert mock_deepface.analyze.call_count == 1


def test_cache_miss_after_ttl_calls_deepface(
    analyzer: FaceAnalyzer,
    synthetic_face_crop: Any,
    mock_deepface: Any,
) -> None:
    """After TTL expires, DeepFace must be called again."""
    with patch("deepface_analytics.analyzer.time.time", side_effect=[1000.0, 1003.0, 1003.0]):
        analyzer.analyze_face(synthetic_face_crop, "face_002")
        analyzer.analyze_face(synthetic_face_crop, "face_002")
    assert mock_deepface.analyze.call_count == 2


def test_age_present_in_result(
    analyzer: FaceAnalyzer,
    synthetic_face_crop: Any,
    mock_deepface: Any,
) -> None:
    """Result dict must contain the 'age' key."""
    result = analyzer.analyze_face(synthetic_face_crop, "face_003")
    assert result is not None
    assert "age" in result
    assert result["age"] == 28


def test_returns_none_when_deepface_unavailable(
    synthetic_face_crop: Any,
) -> None:
    """When DEEPFACE_AVAILABLE is False, analyze_face must return None."""
    with patch("deepface_analytics.analyzer.DEEPFACE_AVAILABLE", False):
        analyzer = FaceAnalyzer()
        result = analyzer.analyze_face(synthetic_face_crop, "face_004")
    assert result is None
