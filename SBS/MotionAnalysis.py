"""
SBS/MotionAnalysis.py

Defines the MotionAnalysis dataclass, which stores all motion-related
measurements produced by the VideoAnalyzer for a single video. Motion is
estimated using the Farneback dense optical flow algorithm applied to
consecutive sampled frames.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class MotionAnalysis:
    """
    Motion analysis results for an entire video.

    Attributes:
        motion_timeline:
            NumPy array of per-frame optical-flow magnitudes (mean pixel
            displacement between each pair of consecutive frames).

        motion_times:
            NumPy array of timestamps (in seconds) aligned with
            motion_timeline.

        avg_motion:
            Mean of motion_timeline across all sampled frames.

        max_motion:
            Peak motion value observed in any single frame transition.

        min_motion:
            Lowest motion value observed (often 0 for the first frame).

        static_ratio:
            Fraction of frames (0–1) where optical-flow magnitude is below
            the hard-coded static threshold of 0.5 px/frame.

        low_motion_ratio:
            Fraction of frames whose motion falls between the static
            threshold and the 25th percentile of the motion distribution.

        high_motion_ratio:
            Fraction of frames whose motion exceeds the 75th percentile
            of the motion distribution.

        camera_motion_detected:
            True when the average motion is above 1.0 and the standard
            deviation is less than half the mean, suggesting a consistent
            directional camera movement (pan/tilt/zoom) rather than
            scene-internal action.

        motion_type:
            Qualitative label for the overall motion character:
            'static'   (avg < 0.5),
            'slow'     (avg < 1.5),
            'moderate' (avg < 3.0),
            'dynamic'  (avg < 5.0),
            'chaotic'  (avg ≥ 5.0).
    """

    motion_timeline: np.ndarray
    motion_times: np.ndarray

    avg_motion: float
    max_motion: float
    min_motion: float

    # Proportion of frames in each motion tier
    static_ratio: float
    low_motion_ratio: float
    high_motion_ratio: float

    # Camera-vs-scene motion heuristic
    camera_motion_detected: bool
    motion_type: str
