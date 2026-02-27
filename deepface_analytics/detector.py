"""Face detection module using Haar Cascade classifier."""

import logging
from typing import Any, List, Tuple

import cv2
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Detection constants
MIN_FACE_SIZE: Tuple[int, int] = (30, 30)
SCALE_FACTOR: float = 1.2
MIN_NEIGHBORS: int = 6

# Bounding box expansion factor
_EXPAND_RATIO: float = 0.1

BoundingBox = Tuple[int, int, int, int]  # (x, y, w, h)


class FaceDetector:
    """Detects faces in frames using Haar Cascade classifier."""

    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        logger.debug("FaceDetector initialized with Haar Cascade")

    def detect_faces(self, frame: npt.NDArray[Any]) -> List[BoundingBox]:
        """Detect faces in *frame* and return a list of (x, y, w, h) bounding boxes.

        Only faces larger than MIN_FACE_SIZE are returned.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        raw: npt.NDArray[Any] = self._cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE,
        )

        if len(raw) == 0:
            return []

        boxes: List[BoundingBox] = [
            (int(x), int(y), int(w), int(h)) for x, y, w, h in raw
        ]
        return self.filter_faces(boxes)

    def filter_faces(self, faces: List[BoundingBox]) -> List[BoundingBox]:
        """Remove bounding boxes smaller than MIN_FACE_SIZE."""
        return [
            (x, y, w, h)
            for x, y, w, h in faces
            if w >= MIN_FACE_SIZE[0] and h >= MIN_FACE_SIZE[1]
        ]

    def expand_bounding_box(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        frame_shape: Tuple[int, ...],
    ) -> BoundingBox:
        """Expand a bounding box by *_EXPAND_RATIO* while clamping to frame boundaries."""
        frame_h, frame_w = frame_shape[:2]

        new_x = max(0, x - int(w * _EXPAND_RATIO))
        new_y = max(0, y - int(h * _EXPAND_RATIO))
        new_w = min(frame_w - new_x, int(w * (1 + 2 * _EXPAND_RATIO)))
        new_h = min(frame_h - new_y, int(h * (1 + 2 * _EXPAND_RATIO)))

        return new_x, new_y, new_w, new_h
