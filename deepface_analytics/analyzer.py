"""Face analysis module with TTL cache and DeepFace integration."""

import logging
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Cache TTL in seconds
EMOTION_RECHECK_INTERVAL: float = 2.0

# Target size for DeepFace analysis (downscale only, never upscale)
_ANALYSIS_SIZE: Tuple[int, int] = (224, 224)

try:
    from deepface import DeepFace as _DeepFace

    DEEPFACE_AVAILABLE: bool = True
except ImportError:
    DEEPFACE_AVAILABLE = False

AnalysisResult = Dict[str, Any]
_CacheEntry = Tuple[float, AnalysisResult]


class FaceAnalyzer:
    """Analyzes face images using DeepFace with a TTL-based cache."""

    def __init__(self, ttl: float = EMOTION_RECHECK_INTERVAL) -> None:
        self._ttl = ttl
        self._cache: Dict[str, _CacheEntry] = {}

    def analyze_face(
        self,
        face_img: npt.NDArray[Any],
        face_id: str,
    ) -> Optional[AnalysisResult]:
        """Analyze *face_img* for the given *face_id*, using cache when fresh.

        Returns a dict with keys: dominant_emotion, emotion_scores, embedding, age.
        Returns None when DeepFace is unavailable.
        """
        if not DEEPFACE_AVAILABLE:
            return None

        now = time.time()
        cached = self._cache.get(face_id)
        if cached is not None:
            ts, result = cached
            if now - ts < self._ttl:
                return result

        # Downscale to _ANALYSIS_SIZE if larger; do not upscale small crops
        h, w = face_img.shape[:2]
        if h > _ANALYSIS_SIZE[0] or w > _ANALYSIS_SIZE[1]:
            face_img = cv2.resize(
                face_img,
                (_ANALYSIS_SIZE[1], _ANALYSIS_SIZE[0]),
                interpolation=cv2.INTER_AREA,
            )

        try:
            raw = _DeepFace.analyze(
                face_img,
                actions=["emotion", "age", "embedding"],
                enforce_detection=False,
                silent=True,
            )
            data: Dict[str, Any] = raw[0] if isinstance(raw, list) else raw
            result = {
                "dominant_emotion": data.get("dominant_emotion", "unknown"),
                "emotion_scores": data.get("emotion", {}),
                "embedding": data.get("embedding", []),
                "age": data.get("age", 0),
            }
        except Exception:
            logger.exception("DeepFace analysis failed for face_id=%s", face_id)
            return None

        self._cache[face_id] = (now, result)
        return result

    def invalidate_cache(self, face_id: str) -> None:
        """Remove the cached result for *face_id*."""
        self._cache.pop(face_id, None)
