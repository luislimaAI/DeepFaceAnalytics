"""Profiling script: measures FaceDetector and FaceAnalyzer performance on synthetic frames."""

from __future__ import annotations

import argparse
import cProfile
import datetime
import json
import os
import pstats
import sys
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import psutil

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from deepface_analytics.analyzer import FaceAnalyzer  # noqa: E402
from deepface_analytics.detector import FaceDetector  # noqa: E402

NUM_FRAMES = 300
FRAME_HEIGHT = 480
FRAME_WIDTH = 640


def _make_mock_analyzer_result() -> Dict[str, Any]:
    return {
        "dominant_emotion": "neutral",
        "emotion_scores": {"neutral": 90.0, "happy": 10.0},
        "embedding": [0.1] * 128,
        "age": 30,
    }


def run_profile(no_deepface: bool = False) -> Dict[str, Any]:
    detector = FaceDetector()
    analyzer = FaceAnalyzer(ttl=0.0)  # TTL=0 forces a new "miss" each call

    frames = [
        np.random.randint(0, 256, (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        for _ in range(NUM_FRAMES)
    ]
    face_crop = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    process = psutil.Process()
    ram_before = process.memory_info().rss / 1024 / 1024  # MB

    frame_latencies: list[float] = []
    deepface_calls = 0

    mock_result = _make_mock_analyzer_result()
    mock_df = MagicMock()
    mock_df.analyze.return_value = [mock_result]

    def _run() -> None:
        nonlocal deepface_calls
        for i, frame in enumerate(frames):
            t0 = time.perf_counter()
            detector.detect_faces(frame)
            if not no_deepface:
                with (
                    patch("deepface_analytics.analyzer.DEEPFACE_AVAILABLE", True),
                    patch("deepface_analytics.analyzer._DeepFace", mock_df, create=True),
                ):
                    result = analyzer.analyze_face(face_crop, f"face_{i % 5}")
                if result is not None:
                    deepface_calls += 1
            latency_ms = (time.perf_counter() - t0) * 1000
            frame_latencies.append(latency_ms)

    # Profile with cProfile
    profiler = cProfile.Profile()
    t_start = time.perf_counter()
    profiler.runcall(_run)
    elapsed = time.perf_counter() - t_start

    ram_after = process.memory_info().rss / 1024 / 1024  # MB

    fps_mean = NUM_FRAMES / elapsed if elapsed > 0 else 0
    latencies_sorted = sorted(frame_latencies)
    p50 = latencies_sorted[int(0.50 * len(latencies_sorted))]
    p95 = latencies_sorted[int(0.95 * len(latencies_sorted))]
    deepface_per_min = (deepface_calls / elapsed) * 60 if elapsed > 0 else 0

    # Save cProfile data
    prof_path = os.path.join(_HERE, "baseline_profile.prof")
    profiler.dump_stats(prof_path)

    # Print top 10 functions by cumulative time
    stats = pstats.Stats(profiler, stream=open(os.devnull, "w"))
    stats.sort_stats("cumulative")
    print("\n=== Top 10 functions by cumulative time ===")
    top_stats = pstats.Stats(profiler)
    top_stats.sort_stats("cumulative")
    top_stats.print_stats(10)

    report: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "num_frames": NUM_FRAMES,
        "no_deepface": no_deepface,
        "fps": {
            "mean": round(fps_mean, 2),
            "min": round(min(
                [1000 / max(lat, 0.001) for lat in frame_latencies] or [0]
            ), 2),
        },
        "frame_latency_ms": {
            "p50": round(p50, 3),
            "p95": round(p95, 3),
        },
        "deepface_calls_per_minute": round(deepface_per_min, 2),
        "ram_mb": {
            "before": round(ram_before, 2),
            "after": round(ram_after, 2),
            "delta": round(ram_after - ram_before, 2),
        },
    }

    report_path = os.path.join(_HERE, "baseline_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=4)

    print("\n=== Profiling Summary ===")
    print(f"Frames processed : {NUM_FRAMES}")
    print(f"Elapsed          : {elapsed:.2f}s")
    print(f"FPS (mean)       : {fps_mean:.1f}")
    print(f"Latency p50      : {p50:.2f}ms")
    print(f"Latency p95      : {p95:.2f}ms")
    print(f"DeepFace calls/m : {deepface_per_min:.1f}")
    print(f"RAM delta        : {ram_after - ram_before:.1f} MB")
    print(f"\nReport saved to  : {report_path}")
    print(f"Profile saved to : {prof_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile DeepFaceAnalytics pipeline")
    parser.add_argument(
        "--no-deepface",
        action="store_true",
        help="Profile detection-only path (skip FaceAnalyzer)",
    )
    args = parser.parse_args()
    run_profile(no_deepface=args.no_deepface)


if __name__ == "__main__":
    main()
