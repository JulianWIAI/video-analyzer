"""
SBS/__init__.py

Package initializer for the SBS (Scene-By-Scene) module collection.
Re-exports every public class and the Visualizer module so callers can write:

    from SBS import VideoAnalyzer, VideoAnalysis, ColorInfo, ...
    from SBS import Visualizer

instead of importing each symbol from its individual file.
"""

from SBS.ColorInfo import ColorInfo
from SBS.ColorAnalysis import ColorAnalysis
from SBS.Scene import Scene
from SBS.SceneAnalysis import SceneAnalysis
from SBS.MotionAnalysis import MotionAnalysis
from SBS.BrightnessAnalysis import BrightnessAnalysis
from SBS.ObjectInfo import ObjectInfo
from SBS.ObjectAnalysis import ObjectAnalysis
from SBS.VisualPatternAnalysis import VisualPatternAnalysis
from SBS.VideoAnalysis import VideoAnalysis
from SBS.VideoAnalyzer import VideoAnalyzer
from SBS.Style import Style
from SBS.VisualizerStyle import VisualizerStyle
from SBS.VideoAnalyzerGUI import VideoAnalyzerGUI
from SBS import Visualizer

__all__ = [
    "ColorInfo",
    "ColorAnalysis",
    "Scene",
    "SceneAnalysis",
    "MotionAnalysis",
    "BrightnessAnalysis",
    "ObjectInfo",
    "ObjectAnalysis",
    "VisualPatternAnalysis",
    "VideoAnalysis",
    "VideoAnalyzer",
    "Style",
    "VisualizerStyle",
    "VideoAnalyzerGUI",
    "Visualizer",
]
