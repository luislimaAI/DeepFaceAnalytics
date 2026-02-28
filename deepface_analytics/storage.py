"""Face storage module for persistence, image saving, and session reporting."""

import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Maximum number of known-face entries before pruning
MAX_KNOWN_FACES: int = 500


class FaceStorage:
    """Handles all persistence logic for known faces and session results.

    Provides methods to load/save known faces from JSON, register new faces
    (with image saving), prune the database when it grows too large, and
    generate a session summary JSON.
    """

    def __init__(self) -> None:
        self.known_faces: Dict[str, Any] = {}
        self._face_counter: int = 0

    # ------------------------------------------------------------------
    # JSON persistence
    # ------------------------------------------------------------------

    def load_known_faces(self, path: str) -> Dict[str, Any]:
        """Load known faces from *path*.

        Returns an empty dict if the file does not exist or cannot be parsed.
        """
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data: Dict[str, Any] = json.load(fh)
            logger.info("Carregadas %d faces conhecidas de %s", len(data), path)
            return data
        except Exception:
            logger.exception("Falha ao carregar faces de %s", path)
            return {}

    def save_known_faces(self, data: Dict[str, Any], path: str) -> None:
        """Write *data* to *path* as indented JSON (indent=4)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=4)
            logger.info("Salvadas %d faces conhecidas em %s", len(data), path)
        except Exception:
            logger.exception("Falha ao salvar faces em %s", path)

    # ------------------------------------------------------------------
    # Face registration
    # ------------------------------------------------------------------

    def register_face(
        self,
        face_img: npt.NDArray[Any],
        emotion: Optional[str],
        age: Optional[int],
        known_faces_dir: str,
    ) -> Tuple[str, str]:
        """Register a new face, save its image, and add it to ``self.known_faces``.

        Args:
            face_img: BGR face crop to save as JPEG.
            emotion: Initial dominant emotion (or None).
            age: Estimated age (or None).
            known_faces_dir: Directory where JPEG images are stored.

        Returns:
            ``(face_id, name)`` tuple for the newly registered face.
        """
        os.makedirs(known_faces_dir, exist_ok=True)
        self._face_counter += 1
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        face_id = f"face_{now_str}_{self._face_counter}"
        name = f"Pessoa {self._face_counter}"

        img_path = os.path.join(known_faces_dir, f"{face_id}.jpg")
        cv2.imwrite(img_path, face_img)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.known_faces[face_id] = {
            "name": name,
            "image_path": img_path,
            "first_seen": timestamp,
            "last_seen": timestamp,
            "detection_count": 1,
            "emotions": {emotion: 1} if emotion else {},
            "ages": [age] if age is not None else [],
        }

        logger.info("Nova face registrada: ID=%s, Nome=%s", face_id, name)
        return face_id, name

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune_known_faces(
        self,
        data: Dict[str, Any],
        max_entries: int = MAX_KNOWN_FACES,
    ) -> Dict[str, Any]:
        """Remove oldest entries by ``first_seen`` when ``len(data) > max_entries``.

        Returns the pruned dict (a new dict; the original is not modified).
        """
        if len(data) <= max_entries:
            return dict(data)

        # Sort by first_seen ascending; oldest entries are removed first
        sorted_ids: List[str] = sorted(
            data.keys(),
            key=lambda fid: data[fid].get("first_seen", ""),
        )
        ids_to_keep = set(sorted_ids[len(sorted_ids) - max_entries :])
        pruned = {fid: entry for fid, entry in data.items() if fid in ids_to_keep}
        removed = len(data) - len(pruned)
        logger.info("Pruned %d old face entries (kept %d)", removed, len(pruned))
        return pruned

    # ------------------------------------------------------------------
    # Session JSON
    # ------------------------------------------------------------------

    def generate_session_json(
        self,
        known_faces: Dict[str, Any],
        stats: Dict[str, Any],
        start_time: float,
    ) -> Dict[str, Any]:
        """Build and return the session summary dict.

        The returned structure matches the documented JSON output format:
        ``timestamp``, ``date``, ``time``, ``session_info``, ``people``.

        Args:
            known_faces: Mapping of face_id -> face data dicts.
            stats: Must contain at least ``total_detected_faces`` (int).
            start_time: Unix timestamp (float) when the session started.
        """
        import time

        now = datetime.datetime.now()
        people: List[Dict[str, Any]] = []

        for face_id, face_data in known_faces.items():
            emotions: Dict[str, int] = face_data.get("emotions", {})
            ages: List[Any] = face_data.get("ages", [])

            # Compute emotion distribution as percentages
            emotion_percentages: Dict[str, float] = {}
            predominant_emotion: Optional[str] = None
            if emotions:
                total = sum(emotions.values())
                emotion_percentages = {
                    em: (cnt / total) * 100 for em, cnt in emotions.items()
                }
                predominant_emotion = max(emotions, key=lambda e: emotions[e])

            avg_age: Optional[float] = (sum(ages) / len(ages)) if ages else None

            people.append(
                {
                    "id": face_id,
                    "name": face_data.get("name", ""),
                    "detection_count": face_data.get("detection_count", 0),
                    "first_seen": face_data.get("first_seen", ""),
                    "last_seen": face_data.get("last_seen", ""),
                    "emotions": {
                        "predominant": predominant_emotion,
                        "distribution": emotion_percentages,
                    },
                    "age": {"average": avg_age, "samples": len(ages)},
                }
            )

        return {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "session_info": {
                "total_detected_faces": stats.get("total_detected_faces", 0),
                "unique_people": len(known_faces),
                "duration_seconds": time.time() - start_time,
            },
            "people": people,
        }
