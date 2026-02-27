"""Unit tests for deepface_analytics/app.py threading and shutdown."""

import queue
import threading
from typing import Any
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


@pytest.fixture
def app_with_faces(app: FaceCounterApp) -> FaceCounterApp:
    """App with sample known faces and emotion stats pre-populated."""
    app.emotion_stats = {"happy": 10, "neutral": 5, "sad": 2}
    app.storage.known_faces = {
        "face_001": {
            "name": "Person 1",
            "emotions": {"happy": 8, "neutral": 2},
            "ages": [25, 27, 26],
            "detection_count": 10,
        },
        "face_002": {
            "name": "Person 2",
            "emotions": {},
            "ages": [],
            "detection_count": 3,
        },
    }
    return app


def test_detect_faces_webcam_exits_when_camera_not_opened(app: FaceCounterApp) -> None:
    """detect_faces_webcam must return early when cap.isOpened() is False."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    with patch("deepface_analytics.app.cv2.VideoCapture", return_value=mock_cap):
        app.detect_faces_webcam()  # Should return without raising


def test_show_emotion_statistics_empty(app: FaceCounterApp, capsys: Any) -> None:
    """show_emotion_statistics prints message when no emotions detected."""
    app.emotion_stats = {}
    app.show_emotion_statistics()
    captured = capsys.readouterr()
    assert "Nenhum sentimento" in captured.out


def test_show_emotion_statistics_with_data(
    app_with_faces: FaceCounterApp, capsys: Any
) -> None:
    """show_emotion_statistics prints stats when emotion data exists."""
    with patch("deepface_analytics.app.plt") as mock_plt:
        mock_plt.figure.return_value = MagicMock()
        mock_plt.show.return_value = None
        mock_plt.savefig.return_value = None
        app_with_faces.show_emotion_statistics()
    captured = capsys.readouterr()
    assert "happy" in captured.out
    assert "10" in captured.out


def test_list_people_count_empty(app: FaceCounterApp, capsys: Any) -> None:
    """list_people_count prints message when no people identified."""
    app.storage.known_faces = {}
    app.list_people_count()
    captured = capsys.readouterr()
    assert "Nenhuma pessoa" in captured.out


def test_list_people_count_with_data(
    app_with_faces: FaceCounterApp, capsys: Any
) -> None:
    """list_people_count displays people and emotion distribution."""
    app_with_faces.list_people_count()
    captured = capsys.readouterr()
    assert "Total de pessoas" in captured.out


def test_save_results_to_json(app_with_faces: FaceCounterApp, tmp_path: Any) -> None:
    """save_results_to_json writes a valid JSON file and returns the path."""
    import json

    app_with_faces.output_dir = str(tmp_path)
    result = app_with_faces.save_results_to_json()
    assert result is not None
    assert result.endswith(".json")
    with open(result, encoding="utf-8") as fh:
        data = json.load(fh)
    assert "timestamp" in data
    assert "people" in data


def test_detect_faces_webcam_exits_when_no_frames(app: FaceCounterApp) -> None:
    """detect_faces_webcam must return when first test frame read fails."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # First 10 reads for warmup return False, then the test read returns (False, None)
    mock_cap.read.return_value = (False, None)
    with patch("deepface_analytics.app.cv2.VideoCapture", return_value=mock_cap):
        app.detect_faces_webcam()  # Should return without raising


def test_show_emotion_statistics_plt_exception(
    app_with_faces: FaceCounterApp, capsys: Any
) -> None:
    """show_emotion_statistics must handle matplotlib exception gracefully."""
    with patch("deepface_analytics.app.plt") as mock_plt:
        mock_plt.figure.side_effect = RuntimeError("no display")
        app_with_faces.show_emotion_statistics()
    captured = capsys.readouterr()
    assert "happy" in captured.out  # stats still printed before chart


def test_main_exits_on_option_5(tmp_path: Any) -> None:
    """main() must exit cleanly when user chooses option 5."""
    from deepface_analytics.app import main

    with (
        patch("deepface_analytics.app.DEEPFACE_AVAILABLE", False),
        patch("builtins.input", return_value="5"),
        patch("deepface_analytics.app.logging.basicConfig"),
    ):
        main(no_deepface=True)  # Should exit without raising


def test_main_handles_invalid_option_then_exit(tmp_path: Any) -> None:
    """main() must handle invalid choice and then exit on option 5."""
    from deepface_analytics.app import main

    inputs = iter(["9", "5"])

    with (
        patch("deepface_analytics.app.DEEPFACE_AVAILABLE", False),
        patch("builtins.input", side_effect=lambda _: next(inputs)),
        patch("deepface_analytics.app.logging.basicConfig"),
    ):
        main(no_deepface=True)
