"""
SBS/ObjectAnalysis.py

Defines the ObjectAnalysis dataclass, which aggregates all object-detection
results produced by the optional YOLOv8 pipeline for a single video. It
stores the raw detection list as well as derived summaries (counts, colors,
co-occurrences) to support both JSON export and visualization.
"""

from dataclasses import dataclass
from typing import List, Dict

from SBS.ColorInfo import ColorInfo
from SBS.ObjectInfo import ObjectInfo


@dataclass
class ObjectAnalysis:
    """
    Aggregated object detection results for an entire video.

    This dataclass is populated only when the caller requests object
    detection (detect_objects=True) and the ultralytics package is
    available. When object detection is skipped, the corresponding field
    on VideoAnalysis is set to None.

    Attributes:
        objects_detected:
            Full list of every ObjectInfo event recorded across all
            sampled frames. May contain many entries for the same class
            across different frames.

        object_counts:
            Dict mapping each detected class label to its total detection
            count across the video (e.g. {'person': 42, 'car': 7}).

        object_colors:
            Dict mapping each class label to a list of up to 5 ColorInfo
            objects representing the dominant colors observed inside
            bounding boxes of that class.

        co_occurrences:
            Nested dict counting how often pairs of objects appeared in the
            same frame: co_occurrences[obj_a][obj_b] = n means obj_a and
            obj_b were in the same frame n times. Symmetric (but not
            deduplicated).

        face_count:
            Total number of frames in which at least one 'person' was
            detected — used as a rough proxy for face/human presence count.

        face_times:
            List of timestamps (seconds) for every frame in which a
            'person' was detected.
    """

    objects_detected: List[ObjectInfo]
    object_counts: Dict[str, int]
    object_colors: Dict[str, List[ColorInfo]]
    co_occurrences: Dict[str, Dict[str, int]]

    face_count: int
    face_times: List[float]
