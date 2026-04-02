"""
SBS/VisualPatternAnalysis.py

Defines the VisualPatternAnalysis dataclass, which holds all visual-rhythm
and repetition metrics computed by the VideoAnalyzer. These measurements
capture how repetitive and rhythmically structured a video's visual content
is, independent of color or motion magnitude.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class VisualPatternAnalysis:
    """
    Visual pattern and rhythm analysis results for an entire video.

    Attributes:
        frame_similarity_matrix:
            A square NumPy array of shape (n_sampled, n_sampled) where
            each cell [i, j] holds the OpenCV histogram correlation
            similarity (0–1) between sampled frame i and sampled frame j.
            Computed on a downsampled subset of frames (max 50) for memory
            efficiency. Diagonal is always 1.

        repetition_score:
            Mean off-diagonal similarity between frames that are at least
            n/5 positions apart, clamped to [0, 1]. Higher values indicate
            that temporally distant frames look visually similar — a hallmark
            of loops, repeated segments, or static backgrounds.

        visual_rhythm:
            A 1-D NumPy array of smoothed frame-difference magnitudes over
            time, capturing the intensity of visual change from frame to frame.
            Peaks in this signal correspond to scene changes or cuts.

        rhythm_tempo:
            Estimated number of visual 'beats' per minute, measured as the
            count of local maxima in visual_rhythm that exceed a threshold,
            divided by the video duration in minutes.

        key_frame_indices:
            List of absolute frame indices (in the source video) that
            correspond to the sharpest local maxima in the frame-difference
            signal. These frames are the most visually distinct moments and
            are good candidates for thumbnail extraction.

        key_frame_times:
            List of timestamps (seconds) aligned with key_frame_indices.
    """

    frame_similarity_matrix: np.ndarray
    repetition_score: float

    # Visual change intensity over time and its derived tempo
    visual_rhythm: np.ndarray
    rhythm_tempo: float

    # Visually distinctive frames
    key_frame_indices: List[int]
    key_frame_times: List[float]
