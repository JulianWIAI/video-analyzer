"""
SBS/BrightnessAnalysis.py

Defines the BrightnessAnalysis dataclass, which holds per-frame brightness
and contrast measurements derived from grayscale conversion of sampled video
frames. Brightness is the mean pixel intensity; contrast is its standard
deviation — a common proxy for local tonal range.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class BrightnessAnalysis:
    """
    Brightness and contrast analysis results for an entire video.

    All values are expressed on a 0–100 percentage scale (raw 0–255
    pixel values are divided by 2.55 at construction time inside
    VideoAnalyzer._analyze_brightness).

    Attributes:
        brightness_timeline:
            NumPy array of per-frame mean brightness values (0–100),
            one entry per sampled frame.

        contrast_timeline:
            NumPy array of per-frame contrast values (0–100), where
            contrast is defined as the standard deviation of the
            grayscale pixel intensities within that frame.

        times:
            NumPy array of timestamps (in seconds) aligned with
            brightness_timeline and contrast_timeline.

        avg_brightness:
            Mean brightness across all sampled frames (0–100).

        avg_contrast:
            Mean contrast across all sampled frames (0–100).

        dark_ratio:
            Fraction of frames (0–1) whose raw brightness was below 80
            (≈ 31% of full scale), indicating a predominantly dark frame.

        bright_ratio:
            Fraction of frames (0–1) whose raw brightness exceeded 180
            (≈ 71% of full scale), indicating a predominantly bright frame.

        brightness_category:
            Qualitative label for the overall lighting character:
            'dark'     (avg raw brightness < 80),
            'moderate' (80–170),
            'bright'   (> 170).
    """

    brightness_timeline: np.ndarray
    contrast_timeline: np.ndarray
    times: np.ndarray

    avg_brightness: float
    avg_contrast: float

    # Proportion of frames classified as dark or bright
    dark_ratio: float
    bright_ratio: float

    brightness_category: str
