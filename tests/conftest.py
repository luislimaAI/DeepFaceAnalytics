from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest


@pytest.fixture
def synthetic_frame() -> np.ndarray:  # type: ignore[type-arg]
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_face_crop() -> np.ndarray:  # type: ignore[type-arg]
    return np.zeros((224, 224, 3), dtype=np.uint8)


@pytest.fixture
def mock_deepface_result() -> Dict[str, Any]:
    return {
        "dominant_emotion": "happy",
        "emotion": {"happy": 95.0, "neutral": 5.0},
        "embedding": [0.1] * 2622,
        "age": 28,
    }


@pytest.fixture
def tmp_known_faces_dir(tmp_path: Any) -> Any:
    d = tmp_path / "known_faces"
    d.mkdir()
    return d


@pytest.fixture
def sample_known_faces() -> Dict[str, Any]:
    return {
        "face_001": {
            "name": "Person 1",
            "first_seen": 1700000000.0,
            "dominant_emotion": "happy",
            "age": 25,
            "image_path": "known_faces/face_001.jpg",
        },
        "face_002": {
            "name": "Person 2",
            "first_seen": 1700000100.0,
            "dominant_emotion": "neutral",
            "age": 30,
            "image_path": "known_faces/face_002.jpg",
        },
    }


@pytest.fixture
def mock_deepface(mocker: Any, mock_deepface_result: Dict[str, Any]) -> Any:
    return mocker.patch(
        "deepface.DeepFace.analyze",
        return_value=[mock_deepface_result],
    )


@pytest.fixture
def embedding_list() -> List[float]:
    return [0.1] * 2622
