"""
SBS/Scene.py

Defines the Scene dataclass, which represents a single detected shot or
scene within a video. Each Scene stores its temporal boundaries, visual
properties (dominant color, brightness, motion), the detected transition
type that follows it, and the frame index best suited for a thumbnail.
"""

from dataclasses import dataclass
from typing import Tuple, Dict

from SBS.ColorInfo import ColorInfo


@dataclass
class Scene:
    """
    A single scene (shot) detected within a video.

    Scenes are identified by significant frame-difference thresholds.
    Each scene records its temporal span, representative visual attributes
    measured from a mid-point sample frame, and the type of transition that
    ends it (e.g. hard cut vs. gradual fade).

    Attributes:
        index:
            Zero-based position of this scene in the detected sequence.

        start_time:
            Time in seconds at which this scene begins.

        end_time:
            Time in seconds at which this scene ends.

        duration:
            Length of the scene in seconds (end_time - start_time).

        start_frame:
            Absolute frame index in the source video where this scene starts.

        end_frame:
            Absolute frame index in the source video where this scene ends.

        dominant_color:
            The single most prevalent color sampled from the scene's
            mid-point frame.

        avg_brightness:
            Mean HSV value (brightness) of the mid-point frame, expressed
            as a percentage (0–100).

        avg_motion:
            Mean frame-difference magnitude across all frames in this scene,
            used as a proxy for motion intensity.

        transition_type:
            How this scene transitions to the next one.
            One of: 'cut', 'fade', 'dissolve', 'unknown', or 'end'.

        thumbnail_frame:
            Absolute frame index recommended for extracting a thumbnail
            image (typically the scene's mid-point frame).
    """

    index: int
    start_time: float
    end_time: float
    duration: float
    start_frame: int
    end_frame: int

    # Visual properties sampled from the scene's mid-point frame
    dominant_color: ColorInfo
    avg_brightness: float
    avg_motion: float

    # Transition to the next scene
    transition_type: str

    # Best frame index for a representative thumbnail
    thumbnail_frame: int

    def to_dict(self) -> Dict:
        """
        Serialize this Scene to a JSON-compatible dictionary.

        Returns:
            A dict containing all scene fields. Numeric values are rounded
            to two decimal places. The dominant_color is serialized via its
            own to_dict() method.
        """
        return {
            'index': self.index,
            'start_time': round(self.start_time, 2),
            'end_time': round(self.end_time, 2),
            'duration': round(self.duration, 2),
            'dominant_color': self.dominant_color.to_dict(),
            'avg_brightness': round(self.avg_brightness, 2),
            'avg_motion': round(self.avg_motion, 2),
            'transition_type': self.transition_type
        }
