"""
SBS/VideoAnalyzer.py

Defines the VideoAnalyzer class — the core analysis engine for the
video-analyzer project. Given a path to a video file, it orchestrates
frame extraction followed by six independent analysis passes:
  1. Color analysis  (dominant colors, palette, temperature, mood)
  2. Scene detection (shot boundaries, transition types, pacing)
  3. Motion analysis (optical flow, camera motion, motion type)
  4. Brightness analysis (brightness/contrast timelines)
  5. Visual patterns  (similarity matrix, rhythm, key frames)
  6. Object detection (optional, requires ultralytics/YOLOv8)

Results are returned as a single VideoAnalysis dataclass instance.
"""

import cv2
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional, List, Dict, Tuple, Any
import colorsys
import warnings

from sklearn.cluster import KMeans

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

warnings.filterwarnings('ignore')


class VideoAnalyzer:
    """
    Orchestrates complete video analysis across all analytical dimensions.

    The analyzer samples frames at a configurable rate, then passes the
    frame buffer to each analysis method. An optional progress callback
    allows the GUI (or any other caller) to receive human-readable status
    updates during long-running operations.

    Attributes:
        SUPPORTED_FORMATS: Set of file extensions accepted by analyze().
    """

    SUPPORTED_FORMATS = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.wmv', '.flv', '.m4v'}

    def __init__(self, sample_rate: int = 2):
        """
        Initialize the analyzer.

        Args:
            sample_rate: Number of frames to analyze per second of video.
                         Higher values produce more accurate results at the
                         cost of longer processing time. Typical range: 1–8.
        """
        self.sample_rate = sample_rate
        self.progress_callback = None
        self._yolo_model = None

    def set_progress_callback(self, callback):
        """
        Register a callable to receive progress status messages.

        The callback is invoked with a single string argument whenever a
        meaningful step begins or a percentage update is available. It is
        called from whatever thread analyze() is running in, so GUI callers
        must ensure thread-safe message passing (e.g. via a queue).

        Args:
            callback: A callable that accepts one str argument.
        """
        self.progress_callback = callback

    def _log(self, message: str):
        """
        Emit a progress message to stdout and, if registered, the callback.

        Args:
            message: Human-readable status string to emit.
        """
        print(message)
        if self.progress_callback:
            self.progress_callback(message)

    def analyze(self, filepath: str, detect_objects: bool = False,
                object_model_size: str = 'medium') -> VideoAnalysis:
        """
        Perform complete video analysis and return all results.

        Opens the video, extracts sampled frames, runs each analysis pass
        in sequence, and assembles the results into a VideoAnalysis object.

        Args:
            filepath: Path to the video file. Must have a supported extension.
            detect_objects: Whether to run the optional YOLOv8 object
                            detection pass (significantly slower).
            object_model_size: YOLOv8 model size to use for object detection.
                               One of: 'nano', 'small', 'medium', 'large',
                               'xlarge'. Larger models are more accurate but
                               require more memory and time.

        Returns:
            A fully populated VideoAnalysis dataclass instance.

        Raises:
            ValueError: If the file extension is unsupported, the file cannot
                        be opened by OpenCV, or fewer than 5 frames can be
                        extracted (likely a corrupted file).
        """
        filepath = Path(filepath)

        if filepath.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {filepath.suffix}")

        self._log(f"Opening: {filepath.name}")

        # Open the video capture handle.
        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {filepath}")

        # Read basic container properties.
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        self._log(f"Duration: {duration:.1f}s, FPS: {fps:.1f}, Resolution: {width}x{height}")

        # Extract and preprocess frames for analysis.
        self._log("Extracting frames...")
        frames_data = self._extract_frames(cap, fps, frame_count)

        cap.release()

        # --- Run each analysis pass sequentially ---
        self._log("Analyzing colors (enhanced)...")
        colors = self._analyze_colors(frames_data)

        self._log("Detecting scenes...")
        scenes = self._analyze_scenes(frames_data, fps)

        self._log("Analyzing motion...")
        motion = self._analyze_motion(frames_data)

        self._log("Analyzing brightness...")
        brightness = self._analyze_brightness(frames_data)

        self._log("Analyzing patterns...")
        patterns = self._analyze_patterns(frames_data, fps)

        # Object detection is optional and requires ultralytics.
        objects = None
        if detect_objects:
            self._log(f"Detecting objects (model: {object_model_size})...")
            objects = self._analyze_objects(frames_data, model_size=object_model_size)

        self._log("Analysis complete!")

        return VideoAnalysis(
            filename=filepath.name,
            filepath=str(filepath),
            duration=duration,
            fps=fps,
            frame_count=frame_count,
            resolution=(width, height),
            colors=colors,
            scenes=scenes,
            motion=motion,
            brightness=brightness,
            patterns=patterns,
            objects=objects
        )

    def _extract_frames(self, cap: cv2.VideoCapture, fps: float,
                        frame_count: int) -> Dict[str, Any]:
        """
        Read and downsample frames from an open VideoCapture handle.

        Frames are resized to 320×180 for fast downstream processing.
        Consecutive frames are differenced to produce the diffs array used
        by scene detection and pattern analysis. Corrupted frames are skipped;
        the method aborts if more than 50 consecutive failures occur.

        Args:
            cap: An already-opened cv2.VideoCapture handle.
            fps: Frame rate of the video (used to calculate timestamps).
            frame_count: Total frame count reported by the container
                         (used for progress updates only).

        Returns:
            A dict with keys:
                'frames'  – list of BGR numpy arrays (320×180).
                'indices' – list of absolute frame indices.
                'times'   – numpy array of timestamps (seconds).
                'diffs'   – numpy array of mean absolute frame differences.
                'fps'     – the original fps value (passed through).

        Raises:
            ValueError: If fewer than 5 valid frames were extracted.
        """
        frame_interval = max(1, int(fps / self.sample_rate))

        frames = []
        frame_indices = []
        times = []
        prev_frame = None
        frame_diffs = []

        frame_idx = 0
        failed_reads = 0
        max_failed_reads = 50  # Tolerate short gaps in corrupted files.

        while True:
            ret, frame = cap.read()

            if not ret:
                failed_reads += 1
                if failed_reads > max_failed_reads:
                    break
                frame_idx += 1
                continue

            failed_reads = 0  # Reset on successful read.

            if frame_idx % frame_interval == 0:
                try:
                    if frame is None or frame.size == 0:
                        frame_idx += 1
                        continue

                    small_frame = cv2.resize(frame, (320, 180))
                    frames.append(small_frame)
                    frame_indices.append(frame_idx)
                    times.append(frame_idx / fps)

                    # Compute mean absolute difference for scene/pattern analysis.
                    if prev_frame is not None:
                        diff = cv2.absdiff(small_frame, prev_frame)
                        frame_diffs.append(np.mean(diff))
                    else:
                        frame_diffs.append(0)

                    prev_frame = small_frame.copy()

                except Exception:
                    self._log(f"Skipping corrupted frame {frame_idx}")
                    continue

            frame_idx += 1

            # Emit a progress update every ~10% of total frames.
            if frame_count > 0 and frame_idx % (frame_count // 10 + 1) == 0:
                progress = min(frame_idx / frame_count * 100, 100)
                self._log(f"Extracting frames: {progress:.0f}%")

        if len(frames) < 5:
            raise ValueError(
                f"Could only extract {len(frames)} frames. Video may be corrupted."
            )

        self._log(f"Extracted {len(frames)} frames successfully")

        return {
            'frames': frames,
            'indices': frame_indices,
            'times': np.array(times),
            'diffs': np.array(frame_diffs),
            'fps': fps
        }

    # ------------------------------------------------------------------
    # Color analysis
    # ------------------------------------------------------------------

    def _analyze_colors(self, frames_data: Dict) -> ColorAnalysis:
        """
        Extract and summarize all color information across sampled frames.

        For each frame: computes 5 dominant colors via K-means, collects
        pixel samples for a global clustering pass, and records HSV
        saturation and brightness. Accent colors (high saturation, not in
        the top dominant set) are identified and may be appended to the
        palette. Color transitions (which colors follow each other
        frame-to-frame) and a 11-band color histogram timeline are also built.

        Args:
            frames_data: Dict returned by _extract_frames().

        Returns:
            A populated ColorAnalysis instance.
        """
        frames = frames_data['frames']
        times = frames_data['times']

        all_pixels = []
        frame_colors = []
        saturations = []
        brightnesses = []
        accent_candidates = []

        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Per-frame dominant colors (5 clusters).
            dominant = self._get_dominant_colors(rgb_frame, n_colors=5)
            frame_colors.append(dominant)

            # Collect a random pixel sample for the global clustering pass.
            pixels = rgb_frame.reshape(-1, 3)
            sample_idx = np.random.choice(len(pixels), min(1500, len(pixels)), replace=False)
            all_pixels.extend(pixels[sample_idx])

            # Track HSV statistics for mood classification.
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturations.append(np.mean(hsv_frame[:, :, 1]))
            brightnesses.append(np.mean(hsv_frame[:, :, 2]))

            # Collect vivid accent color candidates.
            for color in dominant:
                h, s, v = color.hsv
                if s > 50 and 25 < v < 90:
                    accent_candidates.append(color)

        # Global dominant colors from all collected pixels (15 clusters).
        all_pixels = np.array(all_pixels)
        dominant_colors, percentages = self._get_dominant_colors_with_percentages(
            all_pixels, n_colors=15
        )

        # Build a diverse palette (up to 8 visually distinct colors).
        palette = self._create_diverse_palette(dominant_colors, percentages, max_colors=8)

        # Find accent colors not already in the top dominant set.
        accent_colors = self._find_accent_colors(accent_candidates, dominant_colors)
        for accent in accent_colors[:3]:
            if not any(self._colors_similar(accent, p) for p in palette):
                palette.append(accent)

        transitions = self._analyze_color_transitions(frame_colors)

        # Color temperature score and qualitative label.
        temperature_score = self._calculate_temperature(dominant_colors, percentages)
        if temperature_score > 0.2:
            temperature = "warm"
        elif temperature_score < -0.2:
            temperature = "cool"
        else:
            temperature = "neutral"

        # Mood derived from average saturation and brightness.
        avg_sat = np.mean(saturations)
        avg_bright = np.mean(brightnesses)

        if avg_bright > 170:
            mood = "bright"
        elif avg_bright < 80:
            mood = "dark"
        elif avg_sat > 120:
            mood = "vibrant"
        else:
            mood = "muted"

        color_histogram = self._create_color_histogram_timeline(frame_colors, times)

        return ColorAnalysis(
            dominant_colors=dominant_colors,
            color_percentages=percentages,
            palette=palette,
            frame_colors=frame_colors,
            frame_times=times.tolist(),
            color_histogram_timeline=color_histogram,
            color_transitions=transitions,
            temperature=temperature,
            temperature_score=temperature_score,
            mood=mood,
            avg_saturation=avg_sat / 2.55,   # Convert 0–255 → 0–100.
            avg_brightness=avg_bright / 2.55
        )

    def _create_diverse_palette(self, colors: List[ColorInfo],
                                percentages: List[float],
                                max_colors: int = 8) -> List[ColorInfo]:
        """
        Select a visually diverse subset of colors from the dominant list.

        Iterates the dominant colors in frequency order and includes a color
        only if it is not too similar to any already-selected palette color,
        ensuring the palette avoids near-duplicates caused by lighting
        variations of the same hue.

        Args:
            colors: Dominant colors sorted by frequency (most common first).
            percentages: Corresponding frequency percentages (unused here, but
                         kept for a consistent signature with callers).
            max_colors: Maximum number of colors to include in the palette.

        Returns:
            A list of ColorInfo objects of length ≤ max_colors.
        """
        palette = []
        for color in colors:
            is_unique = not any(self._colors_similar(color, existing) for existing in palette)
            if is_unique:
                palette.append(color)
            if len(palette) >= max_colors:
                break
        return palette

    def _colors_similar(self, c1: ColorInfo, c2: ColorInfo, threshold: float = 30) -> bool:
        """
        Determine whether two colors are perceptually similar.

        Uses Euclidean distance in RGB space. A threshold of 30 (out of a
        maximum possible distance of ~441) corresponds roughly to colors
        that are indistinguishable in a small color swatch.

        Args:
            c1: First color.
            c2: Second color.
            threshold: Maximum RGB Euclidean distance to consider similar.

        Returns:
            True if the two colors are within threshold distance of each other.
        """
        r1, g1, b1 = c1.rgb
        r2, g2, b2 = c2.rgb
        distance = np.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)
        return distance < threshold

    def _find_accent_colors(self, candidates: List[ColorInfo],
                            dominant: List[ColorInfo]) -> List[ColorInfo]:
        """
        Identify vivid accent colors not already present in the dominant set.

        Accent colors appear less frequently than the primary palette but
        have high saturation and therefore visual impact. They are scored by
        saturation × log(count) and the top 5 are returned.

        Args:
            candidates: ColorInfo objects that passed the saturation/value
                        pre-filter during per-frame analysis.
            dominant: Top dominant colors (first 5 are checked for overlap).

        Returns:
            Up to 5 accent ColorInfo objects sorted by score (highest first).
        """
        if not candidates:
            return []

        accent_counts: Dict[str, dict] = {}
        for color in candidates:
            key = color.name
            if key not in accent_counts:
                accent_counts[key] = {'color': color, 'count': 0, 'saturation': 0}
            accent_counts[key]['count'] += 1
            accent_counts[key]['saturation'] = max(
                accent_counts[key]['saturation'], color.hsv[1]
            )

        scored_accents = []
        for name, data in accent_counts.items():
            # Skip colors that are already among the top dominant colors.
            if any(d.name == name for d in dominant[:5]):
                continue
            score = data['saturation'] * np.log1p(data['count'])
            scored_accents.append((data['color'], score))

        scored_accents.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored_accents[:5]]

    def _get_dominant_colors(self, image: np.ndarray, n_colors: int = 5) -> List[ColorInfo]:
        """
        Extract dominant colors from a single RGB image using K-means.

        Up to 5 000 pixels are sampled randomly before clustering to keep
        per-frame processing fast. The cluster centers are converted to
        ColorInfo instances.

        Args:
            image: An RGB NumPy array (H × W × 3).
            n_colors: Number of K-means clusters (dominant colors) to find.

        Returns:
            A list of n_colors ColorInfo objects (order matches cluster
            centers, not sorted by frequency).
        """
        pixels = image.reshape(-1, 3)
        if len(pixels) > 5000:
            idx = np.random.choice(len(pixels), 5000, replace=False)
            pixels = pixels[idx]

        kmeans = KMeans(n_clusters=n_colors, n_init=10, max_iter=100)
        kmeans.fit(pixels)

        colors = []
        for center in kmeans.cluster_centers_:
            r, g, b = int(center[0]), int(center[1]), int(center[2])
            colors.append(ColorInfo.from_rgb(r, g, b))
        return colors

    def _get_dominant_colors_with_percentages(
        self, pixels: np.ndarray, n_colors: int = 8
    ) -> Tuple[List[ColorInfo], List[float]]:
        """
        Extract dominant colors and their pixel-coverage percentages.

        Similar to _get_dominant_colors() but also counts how many pixels
        each cluster center represents, enabling frequency-based sorting and
        percentage reporting.

        Args:
            pixels: A flat array of shape (N, 3) containing RGB pixel values.
            n_colors: Number of K-means clusters to fit.

        Returns:
            A tuple of (colors, percentages) where both lists are sorted
            from most to least frequent, and percentages sum to 100.
        """
        if len(pixels) > 10000:
            idx = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[idx]

        kmeans = KMeans(n_clusters=n_colors, n_init=10, max_iter=100)
        labels = kmeans.fit_predict(pixels)

        counts = Counter(labels)
        total = sum(counts.values())

        colors = []
        percentages = []
        for label, count in counts.most_common():
            center = kmeans.cluster_centers_[label]
            r, g, b = int(center[0]), int(center[1]), int(center[2])
            colors.append(ColorInfo.from_rgb(r, g, b))
            percentages.append(count / total * 100)

        return colors, percentages

    def _analyze_color_transitions(
        self, frame_colors: List[List[ColorInfo]]
    ) -> Dict[str, Dict[str, int]]:
        """
        Count how often each named color is followed by each other named color.

        Only the primary (index 0) color of each frame is considered.
        Results are used to populate the 'Color Transitions' chart in the
        visualizer.

        Args:
            frame_colors: Per-frame list of dominant ColorInfo objects.

        Returns:
            A dict of the form {from_color: {to_color: count}}.
        """
        transitions: Dict = defaultdict(lambda: defaultdict(int))
        for i in range(len(frame_colors) - 1):
            current = frame_colors[i][0].name if frame_colors[i] else "Unknown"
            nxt = frame_colors[i + 1][0].name if frame_colors[i + 1] else "Unknown"
            transitions[current][nxt] += 1
        return {k: dict(v) for k, v in transitions.items()}

    def _calculate_temperature(self, colors: List[ColorInfo],
                               percentages: List[float]) -> float:
        """
        Compute a color temperature score weighted by pixel coverage.

        Warm colors (Red, Orange, Yellow, Pink) push the score toward +1;
        cool colors (Blue, Cyan, Purple, Green) push it toward -1.
        Neutral or achromatic colors have no effect.

        Args:
            colors: Dominant color list.
            percentages: Corresponding coverage percentages.

        Returns:
            A float in [-1, 1]. Returns 0 if no warm or cool colors are found.
        """
        warm_colors = {'Red', 'Orange', 'Yellow', 'Pink'}
        cool_colors = {'Blue', 'Cyan', 'Purple', 'Green'}

        warm_score = 0.0
        cool_score = 0.0

        for color, pct in zip(colors, percentages):
            if color.name in warm_colors:
                warm_score += pct
            elif color.name in cool_colors:
                cool_score += pct

        total = warm_score + cool_score
        if total == 0:
            return 0.0
        return (warm_score - cool_score) / total

    def _create_color_histogram_timeline(
        self, frame_colors: List[List[ColorInfo]], times: np.ndarray
    ) -> np.ndarray:
        """
        Build a binary 2-D presence matrix for 11 named base colors over time.

        The 11 base colors are: Red, Orange, Yellow, Green, Cyan, Blue,
        Purple, Pink, White, Gray, Black. Each column represents one sampled
        frame; each row represents one base color. A cell is 1 if that color
        appeared among the frame's dominant colors, 0 otherwise.

        Args:
            frame_colors: Per-frame list of dominant ColorInfo objects.
            times: Timestamp array (used only for shape; not stored in the
                   output array).

        Returns:
            A NumPy array of shape (11, n_frames) with dtype float64.
        """
        color_names = [
            'Red', 'Orange', 'Yellow', 'Green', 'Cyan',
            'Blue', 'Purple', 'Pink', 'White', 'Gray', 'Black'
        ]
        histogram = np.zeros((len(color_names), len(times)))
        for i, colors in enumerate(frame_colors):
            for color in colors:
                if color.name in color_names:
                    idx = color_names.index(color.name)
                    histogram[idx, i] = 1
        return histogram

    # ------------------------------------------------------------------
    # Scene detection
    # ------------------------------------------------------------------

    def _analyze_scenes(self, frames_data: Dict, fps: float) -> SceneAnalysis:
        """
        Detect scene boundaries and classify each scene's visual properties.

        Scene changes are identified where the mean absolute frame difference
        exceeds mean + 2 × std. A smoothed version of the difference signal
        is also computed to classify gradual transitions (fades, dissolves)
        versus hard cuts. Each detected scene is sampled at its mid-point for
        dominant color and brightness.

        Args:
            frames_data: Dict returned by _extract_frames().
            fps: Original video frame rate (used to compute cuts_per_minute).

        Returns:
            A fully populated SceneAnalysis instance.
        """
        diffs = frames_data['diffs']
        times = frames_data['times']
        frames = frames_data['frames']
        indices = frames_data['indices']

        # Hard-cut threshold: mean + 2 standard deviations.
        threshold = np.mean(diffs) + 2 * np.std(diffs)

        # Smoothed signal for detecting gradual transitions.
        window_size = 5
        smoothed_diffs = np.convolve(diffs, np.ones(window_size) / window_size, mode='same')
        gradual_threshold = np.mean(smoothed_diffs) + 1.5 * np.std(smoothed_diffs)

        scene_changes = np.where(diffs > threshold)[0]
        scenes = []
        prev_end = 0

        for change_idx in scene_changes:
            if change_idx <= prev_end:
                continue

            # Classify transition type based on the sharpness of the change.
            if diffs[change_idx] > threshold * 1.5:
                transition = "cut"
            elif (change_idx > 0
                  and smoothed_diffs[change_idx] > gradual_threshold):
                if change_idx > 2 and np.mean(diffs[change_idx - 2:change_idx + 1]) > threshold * 0.5:
                    transition = "dissolve"
                else:
                    transition = "fade"
            else:
                transition = "cut"

            start_idx = prev_end
            end_idx = change_idx

            # Ignore very short segments (fewer than 2 sampled frames).
            if end_idx - start_idx < 2:
                prev_end = change_idx
                continue

            # Sample from the scene's mid-point frame.
            mid_idx = (start_idx + end_idx) // 2
            if mid_idx < len(frames):
                scene_frame = frames[mid_idx]
                rgb_frame = cv2.cvtColor(scene_frame, cv2.COLOR_BGR2RGB)
                dominant = self._get_dominant_colors(rgb_frame, n_colors=1)[0]
                hsv = cv2.cvtColor(scene_frame, cv2.COLOR_BGR2HSV)
                brightness = np.mean(hsv[:, :, 2]) / 2.55
            else:
                dominant = ColorInfo.from_rgb(128, 128, 128)
                brightness = 50.0

            scene_motion = (
                np.mean(diffs[start_idx:end_idx]) if end_idx > start_idx else 0
            )

            scene = Scene(
                index=len(scenes),
                start_time=times[start_idx] if start_idx < len(times) else 0,
                end_time=times[end_idx] if end_idx < len(times) else times[-1],
                duration=(
                    times[end_idx] - times[start_idx]
                    if end_idx < len(times) and start_idx < len(times) else 0
                ),
                start_frame=indices[start_idx] if start_idx < len(indices) else 0,
                end_frame=indices[end_idx] if end_idx < len(indices) else indices[-1],
                dominant_color=dominant,
                avg_brightness=brightness,
                avg_motion=scene_motion,
                transition_type=transition,
                thumbnail_frame=indices[mid_idx] if mid_idx < len(indices) else 0
            )
            scenes.append(scene)
            prev_end = change_idx

        # Append the final scene (from the last change to the end of the video).
        if prev_end < len(times) - 1:
            mid_idx = (prev_end + len(times)) // 2
            if mid_idx < len(frames):
                scene_frame = frames[mid_idx]
                rgb_frame = cv2.cvtColor(scene_frame, cv2.COLOR_BGR2RGB)
                dominant = self._get_dominant_colors(rgb_frame, n_colors=1)[0]
                hsv = cv2.cvtColor(scene_frame, cv2.COLOR_BGR2HSV)
                brightness = np.mean(hsv[:, :, 2]) / 2.55
            else:
                dominant = ColorInfo.from_rgb(128, 128, 128)
                brightness = 50.0

            scene = Scene(
                index=len(scenes),
                start_time=times[prev_end],
                end_time=times[-1],
                duration=times[-1] - times[prev_end],
                start_frame=indices[prev_end],
                end_frame=indices[-1],
                dominant_color=dominant,
                avg_brightness=brightness,
                avg_motion=np.mean(diffs[prev_end:]),
                transition_type="end",
                thumbnail_frame=indices[mid_idx] if mid_idx < len(indices) else indices[-1]
            )
            scenes.append(scene)

        # --- Summary statistics ---
        if scenes:
            durations = [s.duration for s in scenes if s.duration > 0]
            avg_duration = np.mean(durations) if durations else 0
            min_duration = np.min(durations) if durations else 0
            max_duration = np.max(durations) if durations else 0
        else:
            avg_duration = min_duration = max_duration = 0

        total_duration = times[-1] if len(times) > 0 else 1
        cuts_per_minute = len(scenes) / (total_duration / 60) if total_duration > 0 else 0

        # Classify editing pace by cut rate.
        if cuts_per_minute < 5:
            pace = "slow"
        elif cuts_per_minute < 15:
            pace = "moderate"
        elif cuts_per_minute < 30:
            pace = "fast"
        else:
            pace = "very fast"

        cut_count = sum(1 for s in scenes if s.transition_type == "cut")
        fade_count = sum(1 for s in scenes if s.transition_type == "fade")
        dissolve_count = sum(1 for s in scenes if s.transition_type == "dissolve")

        return SceneAnalysis(
            scenes=scenes,
            total_scenes=len(scenes),
            avg_scene_duration=avg_duration,
            min_scene_duration=min_duration,
            max_scene_duration=max_duration,
            cuts_per_minute=cuts_per_minute,
            pace_category=pace,
            cut_count=cut_count,
            fade_count=fade_count,
            dissolve_count=dissolve_count
        )

    # ------------------------------------------------------------------
    # Motion analysis
    # ------------------------------------------------------------------

    def _analyze_motion(self, frames_data: Dict) -> MotionAnalysis:
        """
        Estimate motion intensity using Farneback dense optical flow.

        For each consecutive pair of grayscale frames, the method computes
        the optical-flow field and takes the mean pixel displacement as the
        motion value for that frame transition.

        Args:
            frames_data: Dict returned by _extract_frames().

        Returns:
            A fully populated MotionAnalysis instance.
        """
        frames = frames_data['frames']
        times = frames_data['times']

        motion_values = []
        prev_gray = None

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # Farneback optical flow: returns (dx, dy) per pixel.
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                motion_values.append(np.mean(magnitude))
            else:
                motion_values.append(0)

            prev_gray = gray

        motion_values = np.array(motion_values)

        avg_motion = np.mean(motion_values)
        max_motion = np.max(motion_values)
        min_motion = np.min(motion_values)

        # Percentile-based tier thresholds.
        low_threshold = np.percentile(motion_values, 25)
        high_threshold = np.percentile(motion_values, 75)

        static_ratio = np.sum(motion_values < 0.5) / len(motion_values)
        low_motion_ratio = np.sum(
            (motion_values >= 0.5) & (motion_values < low_threshold)
        ) / len(motion_values)
        high_motion_ratio = np.sum(motion_values > high_threshold) / len(motion_values)

        # Heuristic: consistent motion with low variance → likely camera motion.
        camera_motion = avg_motion > 1.0 and np.std(motion_values) < avg_motion * 0.5

        if avg_motion < 0.5:
            motion_type = "static"
        elif avg_motion < 1.5:
            motion_type = "slow"
        elif avg_motion < 3.0:
            motion_type = "moderate"
        elif avg_motion < 5.0:
            motion_type = "dynamic"
        else:
            motion_type = "chaotic"

        return MotionAnalysis(
            motion_timeline=motion_values,
            motion_times=times,
            avg_motion=avg_motion,
            max_motion=max_motion,
            min_motion=min_motion,
            static_ratio=static_ratio,
            low_motion_ratio=low_motion_ratio,
            high_motion_ratio=high_motion_ratio,
            camera_motion_detected=camera_motion,
            motion_type=motion_type
        )

    # ------------------------------------------------------------------
    # Brightness analysis
    # ------------------------------------------------------------------

    def _analyze_brightness(self, frames_data: Dict) -> BrightnessAnalysis:
        """
        Measure per-frame brightness and contrast from grayscale intensity.

        Brightness is the mean pixel value of the grayscale frame.
        Contrast is the standard deviation of the grayscale pixel values —
        a simple but effective proxy for local tonal range.

        Args:
            frames_data: Dict returned by _extract_frames().

        Returns:
            A fully populated BrightnessAnalysis instance. All timeline and
            average values are expressed on a 0–100 percentage scale.
        """
        frames = frames_data['frames']
        times = frames_data['times']

        brightness_values = []
        contrast_values = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_values.append(np.mean(gray))
            contrast_values.append(np.std(gray))

        brightness_values = np.array(brightness_values)
        contrast_values = np.array(contrast_values)

        avg_brightness = np.mean(brightness_values)
        avg_contrast = np.mean(contrast_values)

        # Thresholds are on the raw 0–255 scale.
        dark_ratio = np.sum(brightness_values < 80) / len(brightness_values)
        bright_ratio = np.sum(brightness_values > 180) / len(brightness_values)

        if avg_brightness < 80:
            category = "dark"
        elif avg_brightness > 170:
            category = "bright"
        else:
            category = "moderate"

        return BrightnessAnalysis(
            brightness_timeline=brightness_values / 2.55,  # → 0–100
            contrast_timeline=contrast_values / 2.55,
            times=times,
            avg_brightness=avg_brightness / 2.55,
            avg_contrast=avg_contrast / 2.55,
            dark_ratio=dark_ratio,
            bright_ratio=bright_ratio,
            brightness_category=category
        )

    # ------------------------------------------------------------------
    # Visual pattern analysis
    # ------------------------------------------------------------------

    def _analyze_patterns(self, frames_data: Dict, fps: float) -> VisualPatternAnalysis:
        """
        Compute visual rhythm, repetition, and key-frame detection.

        A downsampled similarity matrix (max 50 frames) is computed using
        OpenCV histogram correlation. Repetition is measured as the mean
        off-diagonal similarity for pairs of frames more than n/5 positions
        apart. Visual rhythm is a smoothed version of the frame-difference
        signal, and its local peaks are counted to estimate visual tempo.

        Args:
            frames_data: Dict returned by _extract_frames().
            fps: Original video frame rate (not currently used but retained
                 for a consistent signature).

        Returns:
            A fully populated VisualPatternAnalysis instance.
        """
        frames = frames_data['frames']
        times = frames_data['times']
        diffs = frames_data['diffs']

        # Downsample frames to at most 50 for the O(n²) similarity matrix.
        sample_step = max(1, len(frames) // 50)
        sampled_frames = frames[::sample_step]

        n = len(sampled_frames)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                hist_i = cv2.calcHist(
                    [sampled_frames[i]], [0, 1, 2], None,
                    [8, 8, 8], [0, 256, 0, 256, 0, 256]
                )
                hist_j = cv2.calcHist(
                    [sampled_frames[j]], [0, 1, 2], None,
                    [8, 8, 8], [0, 256, 0, 256, 0, 256]
                )
                similarity = cv2.compareHist(hist_i, hist_j, cv2.HISTCMP_CORREL)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        # Repetition: mean similarity of temporally distant frame pairs.
        mask = np.abs(np.arange(n)[:, None] - np.arange(n)) > n // 5
        repetition_score = float(np.mean(similarity_matrix[mask])) if np.any(mask) else 0.0

        # Visual rhythm: smoothed frame-difference signal.
        visual_rhythm = diffs.copy()
        window = min(5, len(visual_rhythm) // 10 + 1)
        if window > 1:
            visual_rhythm = np.convolve(
                visual_rhythm, np.ones(window) / window, mode='same'
            )

        # Count peaks above (mean + 0.5 × std) to estimate visual tempo.
        peaks = []
        threshold = np.mean(visual_rhythm) + 0.5 * np.std(visual_rhythm)
        for i in range(1, len(visual_rhythm) - 1):
            if (visual_rhythm[i] > visual_rhythm[i - 1]
                    and visual_rhythm[i] > visual_rhythm[i + 1]
                    and visual_rhythm[i] > threshold):
                peaks.append(i)

        duration_minutes = times[-1] / 60 if len(times) > 0 else 1
        rhythm_tempo = len(peaks) / duration_minutes if duration_minutes > 0 else 0.0

        # Key frames: local maxima of the difference signal above mean + 1 std.
        key_frame_indices = []
        key_frame_times = []
        key_threshold = np.mean(diffs) + np.std(diffs)
        for i in range(1, len(diffs) - 1):
            if (diffs[i] > diffs[i - 1]
                    and diffs[i] > diffs[i + 1]
                    and diffs[i] > key_threshold):
                key_frame_indices.append(frames_data['indices'][i])
                key_frame_times.append(times[i])

        return VisualPatternAnalysis(
            frame_similarity_matrix=similarity_matrix,
            repetition_score=max(0.0, repetition_score),
            visual_rhythm=visual_rhythm,
            rhythm_tempo=rhythm_tempo,
            key_frame_indices=key_frame_indices,
            key_frame_times=key_frame_times
        )

    # ------------------------------------------------------------------
    # Object detection (optional)
    # ------------------------------------------------------------------

    def _analyze_objects(self, frames_data: Dict,
                         model_size: str = 'medium') -> Optional[ObjectAnalysis]:
        """
        Run YOLOv8 object detection on a sample of frames.

        Requires the ultralytics package. If it is not installed, the method
        logs a message and returns None. The YOLO model is cached on the
        instance so repeated calls with the same model size do not reload it.

        Args:
            frames_data: Dict returned by _extract_frames().
            model_size: One of 'nano', 'small', 'medium', 'large', 'xlarge'.
                        Larger models are more accurate but slower.

        Returns:
            A fully populated ObjectAnalysis, or None if ultralytics is
            unavailable or no objects are detected.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            self._log("Object detection requires ultralytics: pip install ultralytics")
            return None

        frames = frames_data['frames']
        times = frames_data['times']

        model_map = {
            'nano':   'yolov8n.pt',   # Fastest, least accurate.
            'small':  'yolov8s.pt',   # Fast, better accuracy.
            'medium': 'yolov8m.pt',   # Balanced default.
            'large':  'yolov8l.pt',   # Slower, more accurate.
            'xlarge': 'yolov8x.pt',   # Slowest, most accurate.
        }
        model_file = model_map.get(model_size, 'yolov8m.pt')

        # Load (or reuse) the YOLO model.
        if (self._yolo_model is None
                or not hasattr(self, '_yolo_model_size')
                or self._yolo_model_size != model_size):
            self._log(f"Loading YOLO model ({model_size})...")
            self._yolo_model = YOLO(model_file)
            self._yolo_model_size = model_size

        objects_detected = []
        object_counts: Counter = Counter()
        object_colors: Dict = defaultdict(list)
        co_occurrences: Dict = defaultdict(lambda: defaultdict(int))
        face_times = []

        # Analyze up to 50 evenly spaced frames.
        sample_step = max(1, len(frames) // 50)

        for i in range(0, len(frames), sample_step):
            frame = frames[i]
            time = times[i]

            results = self._yolo_model(frame, verbose=False, conf=0.35)

            frame_objects = []
            for r in results:
                for box in r.boxes:
                    label = r.names[int(box.cls)]
                    confidence = float(box.conf)

                    if confidence > 0.35:
                        x, y, w, h = box.xywh[0].tolist()

                        # Sample the dominant color from the object's bounding box.
                        x1 = int(max(0, x - w / 2))
                        y1 = int(max(0, y - h / 2))
                        x2 = int(min(frame.shape[1], x + w / 2))
                        y2 = int(min(frame.shape[0], y + h / 2))

                        obj_color = None
                        if x2 > x1 + 5 and y2 > y1 + 5:
                            obj_region = frame[y1:y2, x1:x2]
                            if obj_region.size > 100:
                                rgb_region = cv2.cvtColor(
                                    obj_region, cv2.COLOR_BGR2RGB
                                )
                                obj_colors_list = self._get_dominant_colors(
                                    rgb_region, n_colors=2
                                )
                                obj_color = obj_colors_list[0] if obj_colors_list else None

                        obj_info = ObjectInfo(
                            label=label,
                            confidence=confidence,
                            bbox=(int(x), int(y), int(w), int(h)),
                            color=obj_color,
                            frame_time=time
                        )
                        objects_detected.append(obj_info)
                        frame_objects.append(label)
                        object_counts[label] += 1

                        if obj_color:
                            object_colors[label].append(obj_color)

                        if label == 'person':
                            face_times.append(time)

            # Record co-occurrences for all unique object pairs in this frame.
            unique_objects = list(set(frame_objects))
            for obj1 in unique_objects:
                for obj2 in unique_objects:
                    if obj1 != obj2:
                        co_occurrences[obj1][obj2] += 1

            if i % (len(frames) // 5 + 1) == 0:
                progress = i / len(frames) * 100
                self._log(f"Object detection: {progress:.0f}%")

        self._log(
            f"Detected {len(object_counts)} object types, "
            f"{sum(object_counts.values())} total"
        )

        return ObjectAnalysis(
            objects_detected=objects_detected,
            object_counts=dict(object_counts),
            # Store at most 5 colors per object class.
            object_colors={k: v[:5] for k, v in object_colors.items()},
            co_occurrences={k: dict(v) for k, v in co_occurrences.items()},
            face_count=len(face_times),
            face_times=face_times
        )

    # ------------------------------------------------------------------
    # Thumbnail extraction
    # ------------------------------------------------------------------

    def extract_thumbnails(self, filepath: str, output_dir: str,
                           analysis: VideoAnalysis,
                           max_thumbnails: int = 10) -> List[str]:
        """
        Extract and save JPEG thumbnail images for detected scenes.

        Seeks directly to the thumbnail_frame index of each scene using
        cv2.CAP_PROP_POS_FRAMES for efficiency rather than decoding every
        frame in sequence.

        Args:
            filepath: Path to the source video file.
            output_dir: Directory in which to save the JPEG thumbnails.
                        Created if it does not already exist.
            analysis: The VideoAnalysis result for the same video file,
                      used to obtain scene thumbnail frame indices.
            max_thumbnails: Maximum number of thumbnails to extract
                            (taken from the first N scenes).

        Returns:
            A list of absolute file paths to the saved thumbnail images.
            Entries are omitted for scenes where the frame could not be read.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return []

        saved = []
        for scene in analysis.scenes.scenes[:max_thumbnails]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, scene.thumbnail_frame)
            ret, frame = cap.read()
            if ret:
                filename = f"scene_{scene.index:03d}_{scene.start_time:.1f}s.jpg"
                out_filepath = output_path / filename
                cv2.imwrite(str(out_filepath), frame)
                saved.append(str(out_filepath))

        cap.release()
        return saved
