"""
SBS/ColorAnalysis.py

Defines the ColorAnalysis dataclass, which aggregates all color-related
measurements produced by the VideoAnalyzer for a single video file.
This includes dominant colors, palette, per-frame color data, temporal
transitions, color temperature, and overall mood classification.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict

from SBS.ColorInfo import ColorInfo


@dataclass
class ColorAnalysis:
    """
    Aggregated color analysis results for an entire video.

    Attributes:
        dominant_colors:
            The top colors found across all sampled frames, ordered by
            frequency (most common first). Up to 15 colors are stored.

        color_percentages:
            Percentage of pixels belonging to each dominant color,
            aligned positionally with dominant_colors.

        palette:
            A curated, diverse color palette of 5–8 visually distinct
            colors selected from dominant_colors (near-duplicates removed).
            Accent colors with high saturation may also be appended.

        frame_colors:
            A list-of-lists: for each sampled frame, the top 5 dominant
            ColorInfo objects detected in that frame.

        frame_times:
            Timestamps (in seconds) corresponding to each entry in
            frame_colors.

        color_histogram_timeline:
            A 2-D NumPy array of shape (11, n_frames) tracking the
            presence of the 11 named base colors (Red, Orange, Yellow,
            Green, Cyan, Blue, Purple, Pink, White, Gray, Black) over time.
            A cell is 1 if that color appeared in the frame, 0 otherwise.

        color_transitions:
            A nested dict mapping each color name to a dict of successor
            color names and how many times that transition occurred, e.g.:
            {'Blue': {'Blue': 120, 'Gray': 5}, ...}.

        temperature:
            Qualitative color temperature label: 'warm', 'cool', or 'neutral'.

        temperature_score:
            Continuous temperature score in [-1, 1], where -1 is maximally
            cool and +1 is maximally warm.

        mood:
            Qualitative mood derived from average saturation and brightness:
            'bright', 'dark', 'vibrant', or 'muted'.

        avg_saturation:
            Mean HSV saturation across all sampled frames, expressed as a
            percentage (0–100).

        avg_brightness:
            Mean HSV value (brightness) across all sampled frames, expressed
            as a percentage (0–100).
    """

    dominant_colors: List[ColorInfo]
    color_percentages: List[float]
    palette: List[ColorInfo]

    # Per-frame color data
    frame_colors: List[List[ColorInfo]]
    frame_times: List[float]

    # Color presence over time (shape: 11 × n_frames)
    color_histogram_timeline: np.ndarray

    # Color-to-color transition counts
    color_transitions: Dict[str, Dict[str, int]]

    # Mood / temperature classification
    temperature: str
    temperature_score: float
    mood: str

    # Aggregate statistics
    avg_saturation: float
    avg_brightness: float
