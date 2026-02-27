"""Unit tests for deepface_analytics/app.py threading and shutdown."""

import queue
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from deepface_analytics.app import FaceCounterApp


@pytest.fixture
def app() -> FaceCounterApp:
    """Create FaceCounterApp with no_deepface=True to avoid heavy model loading."""
    with patch("cv2.VideoCapture"), patch("deepface_analytics.app.DEEPFACE_AVAILABLE", False):
        return FaceCounterApp(no_deepface=True)


def test_stop_event_terminates_worker_thread(app: FaceCounterApp) -> None:
    """Worker thread must stop within 3s after stop_event is set."""
    app.stop_event.clear()
    app.analysis_thread = threading.Thread(target=app._deepface_worker, daemon=True)
    app.analysis_thread.start()
    assert app.analysis_thread.is_alive()
    app.stop_analysis_thread()
    assert not app.analysis_thread.is_alive()


def test_queue_full_does_not_block_main_thread(app: FaceCounterApp) -> None:
    """put_nowait on a full queue raises Full without blocking."""
    face_img = np.zeros((64, 64, 3), dtype=np.uint8)
    # Fill the queue to maxsize=2
    for _ in range(2):
        app.analysis_queue.put_nowait(("face_001", face_img))
    with pytest.raises(queue.Full):
        app.analysis_queue.put_nowait(("face_002", face_img))


def test_results_dict_updated_by_worker(app: FaceCounterApp) -> None:
    """Worker must write results to results_dict when analysis succeeds."""
    mock_result = {
        "dominant_emotion": "happy",
        "emotion_scores": {"happy": 95.0},
        "embedding": [0.1] * 128,
        "age": 25,
    }
    with patch.object(app.analyzer, "analyze_face", return_value=mock_result):
        with patch("deepface_analytics.app.DEEPFACE_AVAILABLE", True):
            app.stop_event.clear()
            app.analysis_thread = threading.Thread(target=app._deepface_worker, daemon=True)
            app.analysis_thread.start()

            face_img = np.zeros((64, 64, 3), dtype=np.uint8)
            app.analysis_queue.put(("face_x", face_img))
            # Wait for worker to process
            app.analysis_queue.join()

            app.stop_analysis_thread()

    with app.results_lock:
        assert "face_x" in app.results_dict
        assert app.results_dict["face_x"]["dominant_emotion"] == "happy"


def test_warmup_calls_deepface_with_blank_frame(app: FaceCounterApp) -> None:
    """warmup_models must call DeepFace.analyze with a blank 224x224 frame."""
    mock_df = MagicMock()
    mock_df.analyze.return_value = [{}]
    with patch("deepface_analytics.app.DEEPFACE_AVAILABLE", True), patch(
        "deepface_analytics.app._DeepFace", mock_df, create=True
    ):
        app.no_deepface = False
        app.warmup_models()

    assert mock_df.analyze.called
    call_args = mock_df.analyze.call_args
    frame_arg = call_args[0][0] if call_args[0] else call_args[1].get("img_path")
    assert frame_arg is not None
    if hasattr(frame_arg, "shape"):
        assert frame_arg.shape == (224, 224, 3)
