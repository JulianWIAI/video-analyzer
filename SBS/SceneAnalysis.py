"""
SBS/SceneAnalysis.py

Defines the SceneAnalysis dataclass, which aggregates all scene-detection
results for a video. It holds the list of detected Scene objects together
with summary statistics about duration, pacing, and transition type counts.
"""

from dataclasses import dataclass
from typing import List

from SBS.Scene import Scene


@dataclass
class SceneAnalysis:
    """
    Aggregated scene detection results for an entire video.

    Attributes:
        scenes:
            Ordered list of all Scene objects detected in the video,
            from first to last.

        total_scenes:
            Total number of detected scenes (len(scenes)).

        avg_scene_duration:
            Mean duration of all scenes, in seconds. Zero if no scenes
            were detected.

        min_scene_duration:
            Shortest scene duration, in seconds.

        max_scene_duration:
            Longest scene duration, in seconds.

        cuts_per_minute:
            Rate of scene changes, computed as
            total_scenes / (video_duration_minutes). Used to classify
            overall editing pace.

        pace_category:
            Qualitative editing pace label derived from cuts_per_minute:
            'slow'  (< 5 cuts/min),
            'moderate' (5–14 cuts/min),
            'fast' (15–29 cuts/min),
            'very fast' (≥ 30 cuts/min).

        cut_count:
            Number of scenes whose transition_type is 'cut' (abrupt change).

        fade_count:
            Number of scenes whose transition_type is 'fade' (gradual
            transition to/from a uniform color, typically black or white).

        dissolve_count:
            Number of scenes whose transition_type is 'dissolve' (gradual
            blend between outgoing and incoming shots).
    """

    scenes: List[Scene]
    total_scenes: int

    # Duration statistics
    avg_scene_duration: float
    min_scene_duration: float
    max_scene_duration: float

    # Pacing
    cuts_per_minute: float
    pace_category: str

    # Transition type breakdown
    cut_count: int
    fade_count: int
    dissolve_count: int
