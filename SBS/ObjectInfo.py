"""
SBS/ObjectInfo.py

Defines the ObjectInfo dataclass, which stores information about a single
object detection event produced by the YOLOv8 detector. Each instance
records the class label, detection confidence, bounding box, the dominant
color sampled from the object's image region, and the timestamp of the
frame in which the detection occurred.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

from SBS.ColorInfo import ColorInfo


@dataclass
class ObjectInfo:
    """
    A single object detection event from a video frame.

    Attributes:
        label:
            The YOLO class label string (e.g. 'person', 'car', 'dog').

        confidence:
            Detection confidence score in the range [0, 1].

        bbox:
            Bounding box in YOLO center-format (x_center, y_center, width,
            height), all values in pixels relative to the original frame.

        color:
            The dominant color sampled from the pixel region inside the
            bounding box, or None if the region was too small to sample
            reliably (fewer than 100 pixels, or width/height < 5 px).

        frame_time:
            Timestamp (in seconds) of the video frame in which this
            detection was made.
    """

    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x_center, y_center, width, height
    color: Optional[ColorInfo]
    frame_time: float
