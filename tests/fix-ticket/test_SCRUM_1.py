"""Regression test for SCRUM-1: emotion detection absent from facial analysis results.

This test reproduces the bug by simulating a DeepFace version that raises
ValueError when 'embedding' is passed as an action to DeepFace.analyze.
In current DeepFace releases the valid actions are ['age', 'gender', 'race', 'emotion'];
passing 'embedding' raises ValueError, which is silently caught and causes
analyze_face() to return None — so dominant_emotion is never surfaced.
"""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from deepface_analytics.analyzer import FaceAnalyzer


@pytest.fixture
def face_crop() -> Any:
    return np.zeros((224, 224, 3), dtype=np.uint8)


def _deepface_strict_actions(img: Any, actions: list, **kwargs: Any) -> list:
    """Simulate modern DeepFace that rejects 'embedding' as an action."""
    if "embedding" in actions:
        raise ValueError(
            "Invalid action 'embedding'. Valid actions are: age, gender, race, emotion."
        )
    return [
        {
            "dominant_emotion": "happy",
            "emotion": {"happy": 95.0, "neutral": 5.0},
            "age": 28,
        }
    ]


def test_scrum_1_emotion_returned_when_embedding_is_invalid_action(
    face_crop: Any, mocker: Any
) -> None:
    """SCRUM-1: analyze_face must return dominant_emotion even when DeepFace
    rejects 'embedding' as an action.

    Before fix: 'embedding' in actions causes ValueError → returns None → FAILS.
    After fix:  'embedding' removed from actions → returns emotion data → PASSES.
    """
    mock_df = MagicMock()
    mock_df.analyze.side_effect = _deepface_strict_actions
    mocker.patch("deepface_analytics.analyzer.DEEPFACE_AVAILABLE", True)
    mocker.patch("deepface_analytics.analyzer._DeepFace", mock_df, create=True)

    analyzer = FaceAnalyzer()
    result = analyzer.analyze_face(face_crop, "face_scrum1")

    assert result is not None, (
        "analyze_face returned None — emotion detection is broken. "
        "Likely cause: 'embedding' is being passed as an invalid action to DeepFace.analyze."
    )
    assert "dominant_emotion" in result
    assert result["dominant_emotion"] == "happy"
