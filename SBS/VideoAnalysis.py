"""
SBS/VideoAnalysis.py

Defines the VideoAnalysis dataclass, which is the top-level result object
returned by VideoAnalyzer.analyze(). It bundles file metadata with all six
analysis sub-results (colors, scenes, motion, brightness, patterns, objects)
and provides a to_dict() method for JSON export.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

from SBS.ColorAnalysis import ColorAnalysis
from SBS.SceneAnalysis import SceneAnalysis
from SBS.MotionAnalysis import MotionAnalysis
from SBS.BrightnessAnalysis import BrightnessAnalysis
from SBS.VisualPatternAnalysis import VisualPatternAnalysis
from SBS.ObjectAnalysis import ObjectAnalysis


@dataclass
class VideoAnalysis:
    """
    Complete analysis result for a single video file.

    This is the root data structure that consumers (GUI, visualizer, CLI)
    interact with after calling VideoAnalyzer.analyze(). It contains both
    video-level metadata and fully populated analysis sub-objects for each
    analytical dimension.

    Attributes:
        filename:
            Base filename of the analyzed video (e.g. 'clip.mp4').

        filepath:
            Absolute path to the video file as a string.

        duration:
            Total video duration in seconds.

        fps:
            Original frame rate of the video (frames per second).

        frame_count:
            Total number of frames reported by the video container.

        resolution:
            (width, height) of the video in pixels.

        colors:
            Color analysis results — dominant colors, palette, timeline,
            temperature, and mood. See ColorAnalysis for details.

        scenes:
            Scene detection results — list of Scene objects plus aggregate
            pacing statistics. See SceneAnalysis for details.

        motion:
            Motion analysis results — optical-flow timeline, motion type,
            and camera-motion detection. See MotionAnalysis for details.

        brightness:
            Brightness and contrast results — per-frame timelines and
            aggregate category. See BrightnessAnalysis for details.

        patterns:
            Visual-pattern and rhythm results — similarity matrix,
            repetition score, visual rhythm, and key frames.
            See VisualPatternAnalysis for details.

        objects:
            Optional object-detection results populated only when
            detect_objects=True is passed to VideoAnalyzer.analyze() and
            the ultralytics package is installed. None otherwise.
    """

    # File metadata
    filename: str
    filepath: str
    duration: float
    fps: float
    frame_count: int
    resolution: Tuple[int, int]

    # Analysis sub-results
    colors: ColorAnalysis
    scenes: SceneAnalysis
    motion: MotionAnalysis
    brightness: BrightnessAnalysis
    patterns: VisualPatternAnalysis
    objects: Optional[ObjectAnalysis]

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the complete analysis to a JSON-compatible dictionary.

        Nested structures (scenes, colors, etc.) are flattened to plain
        Python types. NumPy arrays and ColorInfo objects are converted to
        lists and dicts respectively. Only the most export-relevant fields
        are included (e.g. first 20 scenes, first 10 key-frame times).

        Returns:
            A dict that can be passed directly to json.dump().
        """
        return {
            'file_info': {
                'filename': self.filename,
                'filepath': self.filepath,
                'duration_seconds': round(self.duration, 2),
                'fps': round(self.fps, 2),
                'frame_count': self.frame_count,
                'resolution': {
                    'width': self.resolution[0],
                    'height': self.resolution[1]
                }
            },
            'color_analysis': {
                'dominant_colors': [c.to_dict() for c in self.colors.dominant_colors[:5]],
                'color_percentages': [round(p, 1) for p in self.colors.color_percentages[:5]],
                'palette': [c.to_dict() for c in self.colors.palette],
                'temperature': self.colors.temperature,
                'temperature_score': round(self.colors.temperature_score, 2),
                'mood': self.colors.mood,
                'avg_saturation': round(self.colors.avg_saturation, 1),
                'avg_brightness': round(self.colors.avg_brightness, 1)
            },
            'scene_analysis': {
                'total_scenes': self.scenes.total_scenes,
                'avg_scene_duration': round(self.scenes.avg_scene_duration, 2),
                'min_scene_duration': round(self.scenes.min_scene_duration, 2),
                'max_scene_duration': round(self.scenes.max_scene_duration, 2),
                'cuts_per_minute': round(self.scenes.cuts_per_minute, 1),
                'pace_category': self.scenes.pace_category,
                'transitions': {
                    'cuts': self.scenes.cut_count,
                    'fades': self.scenes.fade_count,
                    'dissolves': self.scenes.dissolve_count
                },
                # Export only the first 20 scenes to keep JSON size manageable.
                'scenes': [s.to_dict() for s in self.scenes.scenes[:20]]
            },
            'motion_analysis': {
                'avg_motion': round(self.motion.avg_motion, 2),
                'max_motion': round(self.motion.max_motion, 2),
                'static_ratio': round(self.motion.static_ratio * 100, 1),
                'low_motion_ratio': round(self.motion.low_motion_ratio * 100, 1),
                'high_motion_ratio': round(self.motion.high_motion_ratio * 100, 1),
                'motion_type': self.motion.motion_type,
                'camera_motion_detected': self.motion.camera_motion_detected
            },
            'brightness_analysis': {
                'avg_brightness': round(self.brightness.avg_brightness, 1),
                'avg_contrast': round(self.brightness.avg_contrast, 1),
                'dark_ratio': round(self.brightness.dark_ratio * 100, 1),
                'bright_ratio': round(self.brightness.bright_ratio * 100, 1),
                'category': self.brightness.brightness_category
            },
            'pattern_analysis': {
                'repetition_score': round(self.patterns.repetition_score, 3),
                'visual_rhythm_tempo': round(self.patterns.rhythm_tempo, 1),
                'key_frame_count': len(self.patterns.key_frame_indices),
                # Export only the first 10 key-frame timestamps.
                'key_frame_times': [round(t, 2) for t in self.patterns.key_frame_times[:10]]
            }
        }
