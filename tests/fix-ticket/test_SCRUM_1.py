"""Regression test for SCRUM-1: emotion not returned due to invalid 'embedding' action."""
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from deepface_analytics.analyzer import FaceAnalyzer


_FAKE_ANALYZE_RESPONSE = [
    {
        "dominant_emotion": "happy",
        "emotion": {"happy": 95.0, "neutral": 5.0},
        "age": 28,
    }
]
_FAKE_EMBEDDING = [0.1] * 128


def _make_strict_mock() -> MagicMock:
    """Return a mock that raises ValueError when 'embedding' is in actions,
    mirroring real DeepFace ≥0.0.90 behaviour for unknown actions."""

    def analyze_side_effect(img: Any, actions: Any, **kwargs: Any) -> Any:
        if "embedding" in actions:
            raise ValueError(
                "Action 'embedding' is not valid. Valid actions: emotion, age, gender, race"
            )
        return _FAKE_ANALYZE_RESPONSE

    mock_df = MagicMock()
    mock_df.analyze.side_effect = analyze_side_effect
    mock_df.represent.return_value = [{"embedding": _FAKE_EMBEDDING}]
    return mock_df


def test_dominant_emotion_not_none_with_strict_deepface(
    synthetic_face_crop: Any,
) -> None:
    """analyze_face must return a result with dominant_emotion when DeepFace
    rejects 'embedding' as an invalid action.

    FAILS before fix (analyze catches the ValueError and returns None).
    PASSES after fix (actions list no longer includes 'embedding').
    """
    mock_df = _make_strict_mock()

    with patch("deepface_analytics.analyzer.DEEPFACE_AVAILABLE", True), patch(
        "deepface_analytics.analyzer._DeepFace", mock_df, create=True
    ):
        analyzer = FaceAnalyzer()
        result = analyzer.analyze_face(synthetic_face_crop, "face_regression_scrum1")

    assert result is not None, (
        "analyze_face returned None — likely because 'embedding' is still in actions "
        "causing DeepFace to raise and the exception handler to swallow the error."
    )
    assert result.get("dominant_emotion") not in (None, "", "unknown"), (
        f"dominant_emotion should be a valid emotion, got: {result.get('dominant_emotion')!r}"
    )


def test_deepface_analyze_actions_do_not_include_embedding(
    synthetic_face_crop: Any,
) -> None:
    """The actions list passed to DeepFace.analyze must NOT contain 'embedding',
    which is not a valid DeepFace action.

    FAILS before fix. PASSES after fix.
    """
    captured_actions: list = []

    def capture_actions(img: Any, actions: Any, **kwargs: Any) -> Any:
        captured_actions.extend(actions)
        return [
            {
                "dominant_emotion": "happy",
                "emotion": {"happy": 95.0},
                "age": 28,
            }
        ]

    mock_df = MagicMock()
    mock_df.analyze.side_effect = capture_actions
    mock_df.represent.return_value = [{"embedding": _FAKE_EMBEDDING}]

    with patch("deepface_analytics.analyzer.DEEPFACE_AVAILABLE", True), patch(
        "deepface_analytics.analyzer._DeepFace", mock_df, create=True
    ):
        analyzer = FaceAnalyzer()
        analyzer.analyze_face(synthetic_face_crop, "face_regression_actions")

    assert "embedding" not in captured_actions, (
        f"'embedding' must not be passed as an action to DeepFace.analyze. "
        f"Got actions: {captured_actions}. "
        f"Use DeepFace.represent() to obtain embeddings separately."
    )


def test_represent_exception_returns_empty_embedding(
    synthetic_face_crop: Any,
) -> None:
    """When DeepFace.represent raises, analyze_face must still return a valid result
    with an empty embedding — the exception must not propagate."""
    mock_df = _make_strict_mock()
    mock_df.represent.side_effect = RuntimeError("represent model unavailable")

    with patch("deepface_analytics.analyzer.DEEPFACE_AVAILABLE", True), patch(
        "deepface_analytics.analyzer._DeepFace", mock_df, create=True
    ):
        analyzer = FaceAnalyzer()
        result = analyzer.analyze_face(synthetic_face_crop, "face_represent_fallback")

    assert result is not None
    assert result.get("embedding") == []
    assert result.get("dominant_emotion") not in (None, "", "unknown")
