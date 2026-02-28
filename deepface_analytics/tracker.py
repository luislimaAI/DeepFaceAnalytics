"""Face tracking and recognition module with vectorized distance computation."""

import logging
import time
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Tracking constants
RECOGNITION_THRESHOLD: float = 0.3
FACE_TRACKING_THRESHOLD: float = 3.0
EMOTION_WINDOW_SIZE: int = 30


class FaceTracker:
    """Tracks and recognizes faces using vectorized NumPy distance computation.

    Maintains parallel arrays: each row of ``_encoding_matrix`` corresponds to
    the entry at the same index in ``_face_ids``.
    """

    def __init__(self) -> None:
        # Shape (N, embedding_dim); starts empty until first encoding is added
        self._encoding_matrix: npt.NDArray[Any] = np.empty((0, 0))
        self._face_ids: List[str] = []
        self._emotion_window: Deque[str] = deque(maxlen=EMOTION_WINDOW_SIZE)

    def is_same_as_known_face(
        self,
        encoding: npt.NDArray[Any],
    ) -> Optional[str]:
        """Return the face_id of the closest known face if within RECOGNITION_THRESHOLD.

        Computes ``distances = np.linalg.norm(matrix - encoding, axis=1)`` in a
        single vectorized NumPy operation (O(N) in NumPy, not a Python loop).
        Returns None if there are no known faces or if the minimum distance
        is >= RECOGNITION_THRESHOLD.
        """
        if not self._face_ids:
            return None

        distances: npt.NDArray[Any] = np.linalg.norm(
            self._encoding_matrix - encoding, axis=1
        )
        min_idx = int(np.argmin(distances))
        if float(distances[min_idx]) < RECOGNITION_THRESHOLD:
            return self._face_ids[min_idx]
        return None

    def add_face_encoding(
        self,
        face_id: str,
        encoding: npt.NDArray[Any],
    ) -> None:
        """Append *encoding* to the matrix and record *face_id* in the parallel list."""
        row = encoding.reshape(1, -1)
        if len(self._face_ids) == 0:
            self._encoding_matrix = row
        else:
            self._encoding_matrix = np.vstack([self._encoding_matrix, row])
        self._face_ids.append(face_id)

    def update_recent_emotion(self, emotion: str) -> str:
        """Append *emotion* to the sliding window and return the most common emotion.

        Window size is bounded by EMOTION_WINDOW_SIZE (oldest entries drop off).
        Uses ``collections.Counter`` to find the mode in O(n).
        """
        self._emotion_window.append(emotion)
        counter: Counter[str] = Counter(self._emotion_window)
        return counter.most_common(1)[0][0]

    def cleanup_trackers(
        self,
        tracked_faces: Dict[str, Any],
    ) -> None:
        """Remove entries from *tracked_faces* that have been inactive too long.

        An entry is considered inactive when
        ``time.time() - state['last_seen'] > FACE_TRACKING_THRESHOLD``.
        Modifies *tracked_faces* in-place.
        """
        now = time.time()
        inactive = [
            face_id
            for face_id, state in tracked_faces.items()
            if now - float(state["last_seen"]) > FACE_TRACKING_THRESHOLD
        ]
        for face_id in inactive:
            del tracked_faces[face_id]
            logger.debug("Removed inactive face_id=%s from tracker", face_id)
