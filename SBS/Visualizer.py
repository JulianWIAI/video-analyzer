"""
SBS/Visualizer.py

Provides all matplotlib-based chart creation functions for the video-analyzer
project. Each public function accepts a VideoAnalysis instance (and an
optional output path) and returns a matplotlib Figure.

Style constants are imported from SBS.VisualizerStyle and aliased locally
as 'Style' so that all existing internal references (Style.ACCENT, etc.)
remain unchanged.

Consumed by:
  - SBS/VideoAnalyzerGUI.py — for in-app chart display and export.
  - main.py — available for scripting via direct import.
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import List, Optional, Dict

from SBS.VisualizerStyle import VisualizerStyle as Style   # aliased for internal use
from SBS.VideoAnalysis import VideoAnalysis
from SBS.ColorInfo import ColorInfo
from SBS.Scene import Scene

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Theme setup
# ---------------------------------------------------------------------------

def setup_style():
    """
    Configure matplotlib's global rcParams for the dark-theme chart style.

    Should be called at the start of every figure-creation function to
    ensure consistent styling regardless of the caller's environment.
    """
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': Style.BG_DARK,
        'axes.facecolor':   Style.BG_PANEL,
        'axes.edgecolor':   Style.GRID,
        'axes.labelcolor':  Style.TEXT,
        'text.color':       Style.TEXT,
        'xtick.color':      Style.TEXT_DIM,
        'ytick.color':      Style.TEXT_DIM,
        'grid.color':       Style.GRID,
        'grid.alpha':       0.3,
        'font.size':        9,
    })


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_color_palette(analysis: VideoAnalysis, ax: plt.Axes):
    """
    Render the video's color palette as a row of colored rectangles.

    Each rectangle shows the color swatch, its name, and approximate pixel
    coverage percentage. Text color is chosen automatically (black on light
    swatches, white on dark) for readability.

    Args:
        analysis: The VideoAnalysis result to visualize.
        ax:       The matplotlib Axes to draw on.
    """
    colors = analysis.colors.palette
    percentages = analysis.colors.color_percentages[:len(colors)]

    n = len(colors)
    for i, (color, pct) in enumerate(zip(colors, percentages)):
        rect = Rectangle((i, 0), 0.9, 1, facecolor=color.hex,
                          edgecolor='white', linewidth=1)
        ax.add_patch(rect)

        # Choose text color for contrast against the swatch background.
        text_color = 'white' if color.hsv[2] < 50 else 'black'
        ax.text(i + 0.45, 0.5, f"{color.name}\n{pct:.0f}%",
                ha='center', va='center', fontsize=8,
                color=text_color, fontweight='bold')

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(
        f'Color Palette | Temperature: {analysis.colors.temperature.capitalize()} '
        f'| Mood: {analysis.colors.mood.capitalize()}',
        fontweight='bold'
    )


def plot_color_timeline(analysis: VideoAnalysis, ax: plt.Axes):
    """
    Render a horizontal strip chart showing the top 3 dominant colors per frame
    over the course of the video.

    Each vertical slice of the strip corresponds to one sampled frame; three
    stacked horizontal bands show the first, second, and third dominant color.

    Args:
        analysis: The VideoAnalysis result to visualize.
        ax:       The matplotlib Axes to draw on.
    """
    frame_colors = analysis.colors.frame_colors
    times = analysis.colors.frame_times

    if not frame_colors or not times:
        ax.text(0.5, 0.5, 'No color data', ha='center', va='center')
        return

    strip_height = 1.0 / max(1, max(len(fc) for fc in frame_colors))

    for i, (time, colors) in enumerate(zip(times, frame_colors)):
        width = times[1] - times[0] if len(times) > 1 else 1

        for j, color in enumerate(colors[:3]):   # Top 3 colors per frame.
            y = j * strip_height
            rect = Rectangle((time, y), width, strip_height,
                              facecolor=color.hex, edgecolor='none')
            ax.add_patch(rect)

    ax.set_xlim(0, max(times) if times else 1)
    ax.set_ylim(0, strip_height * 3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Dominant Colors')
    ax.set_title('Color Timeline', fontweight='bold')
    ax.set_yticks([])


def plot_color_transitions(analysis: VideoAnalysis, ax: plt.Axes):
    """
    Render a horizontal bar chart of the most frequent color-to-color transitions.

    Shows the top 10 (from_color → to_color) pairs sorted by occurrence count.

    Args:
        analysis: The VideoAnalysis result to visualize.
        ax:       The matplotlib Axes to draw on.
    """
    transitions = analysis.colors.color_transitions

    if not transitions:
        ax.text(0.5, 0.5, 'No transition data', ha='center', va='center')
        ax.axis('off')
        return

    pairs = []
    for from_color, to_colors in transitions.items():
        for to_color, count in to_colors.items():
            pairs.append((from_color, to_color, count))

    pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = pairs[:10]

    if not top_pairs:
        ax.text(0.5, 0.5, 'No transitions', ha='center', va='center')
        ax.axis('off')
        return

    labels = [f"{p[0]} → {p[1]}" for p in top_pairs]
    counts = [p[2] for p in top_pairs]

    ax.barh(range(len(labels)), counts, color=Style.ACCENT)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Count')
    ax.set_title('Color Transitions', fontweight='bold')
    ax.invert_yaxis()


def plot_scene_timeline(analysis: VideoAnalysis, ax: plt.Axes):
    """
    Render a horizontal timeline with one colored rectangle per scene.

    Rectangle color matches the scene's dominant color. Scene index numbers
    are overlaid on scenes wide enough to accommodate them.

    Args:
        analysis: The VideoAnalysis result to visualize.
        ax:       The matplotlib Axes to draw on.
    """
    scenes = analysis.scenes.scenes

    if not scenes:
        ax.text(0.5, 0.5, 'No scenes detected', ha='center', va='center')
        return

    for scene in scenes:
        rect = Rectangle((scene.start_time, 0), scene.duration, 1,
                          facecolor=scene.dominant_color.hex,
                          edgecolor='white', linewidth=0.5)
        ax.add_patch(rect)

        # Only label scenes wide enough to display a number legibly.
        if scene.duration > analysis.duration / 20:
            text_color = 'white' if scene.dominant_color.hsv[2] < 50 else 'black'
            ax.text(scene.start_time + scene.duration / 2, 0.5,
                    str(scene.index + 1),
                    ha='center', va='center', fontsize=7, color=text_color)

    ax.set_xlim(0, analysis.duration)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time (s)')
    ax.set_title(
        f'Scene Timeline | {analysis.scenes.total_scenes} scenes '
        f'| {analysis.scenes.cuts_per_minute:.1f} cuts/min '
        f'| Pace: {analysis.scenes.pace_category}',
        fontweight='bold'
    )
    ax.set_yticks([])


def plot_scene_durations(analysis: VideoAnalysis, ax: plt.Axes):
    """
    Render a histogram of scene durations with a vertical average line.

    Args:
        analysis: The VideoAnalysis result to visualize.
        ax:       The matplotlib Axes to draw on.
    """
    durations = [s.duration for s in analysis.scenes.scenes if s.duration > 0]

    if not durations:
        ax.text(0.5, 0.5, 'No scene data', ha='center', va='center')
        return

    ax.hist(durations, bins=20, color=Style.ACCENT, edgecolor='white', alpha=0.7)
    ax.axvline(analysis.scenes.avg_scene_duration,
               color=Style.ACCENT2, linestyle='--', linewidth=2,
               label=f'Avg: {analysis.scenes.avg_scene_duration:.1f}s')
    ax.set_xlabel('Duration (s)')
    ax.set_ylabel('Count')
    ax.set_title('Scene Duration Distribution', fontweight='bold')
    ax.legend()


def plot_motion_timeline(analysis: VideoAnalysis, ax: plt.Axes):
    """
    Render a filled line chart of optical-flow motion intensity over time.

    High-motion periods (above the 75th percentile) are highlighted in a
    contrasting accent color.

    Args:
        analysis: The VideoAnalysis result to visualize.
        ax:       The matplotlib Axes to draw on.
    """
    motion = analysis.motion.motion_timeline
    times = analysis.motion.motion_times

    if len(motion) == 0:
        ax.text(0.5, 0.5, 'No motion data', ha='center', va='center')
        return

    ax.fill_between(times, motion, alpha=0.3, color=Style.ACCENT)
    ax.plot(times, motion, color=Style.ACCENT, linewidth=1)

    # Highlight the high-motion tier.
    high_threshold = np.percentile(motion, 75)
    high_motion_mask = motion > high_threshold
    ax.fill_between(times, motion, where=high_motion_mask,
                    alpha=0.5, color=Style.ACCENT2, label='High motion')

    ax.set_xlim(0, max(times) if len(times) > 0 else 1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Motion Intensity')
    ax.set_title(
        f'Motion Analysis | Type: {analysis.motion.motion_type.capitalize()} '
        f'| Avg: {analysis.motion.avg_motion:.2f}',
        fontweight='bold'
    )


def plot_brightness_timeline(analysis: VideoAnalysis, ax: plt.Axes):
    """
    Render overlapping line charts for brightness and contrast over time.

    Args:
        analysis: The VideoAnalysis result to visualize.
        ax:       The matplotlib Axes to draw on.
    """
    brightness = analysis.brightness.brightness_timeline
    contrast = analysis.brightness.contrast_timeline
    times = analysis.brightness.times

    if len(brightness) == 0:
        ax.text(0.5, 0.5, 'No brightness data', ha='center', va='center')
        return

    ax.plot(times, brightness, color=Style.ACCENT, linewidth=1, label='Brightness')
    ax.plot(times, contrast, color=Style.ACCENT2, linewidth=1, alpha=0.7, label='Contrast')

    # Dashed horizontal line at mean brightness for reference.
    ax.axhline(analysis.brightness.avg_brightness,
               color=Style.ACCENT, linestyle='--', alpha=0.5)

    ax.set_xlim(0, max(times) if len(times) > 0 else 1)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Level (%)')
    ax.set_title(
        f'Brightness & Contrast | Category: {analysis.brightness.brightness_category.capitalize()}',
        fontweight='bold'
    )
    ax.legend(loc='upper right')


def plot_visual_rhythm(analysis: VideoAnalysis, ax: plt.Axes):
    """
    Render the smoothed visual-change intensity (visual rhythm) over time.

    Key-frame timestamps are marked as thin vertical lines to show where
    the most visually distinct moments occur.

    Args:
        analysis: The VideoAnalysis result to visualize.
        ax:       The matplotlib Axes to draw on.
    """
    rhythm = analysis.patterns.visual_rhythm

    if len(rhythm) == 0:
        ax.text(0.5, 0.5, 'No rhythm data', ha='center', va='center')
        return

    times = np.linspace(0, analysis.duration, len(rhythm))

    ax.fill_between(times, rhythm, alpha=0.3, color=Style.ACCENT3)
    ax.plot(times, rhythm, color=Style.ACCENT3, linewidth=1)

    # Mark the first 20 key frames.
    for kf_time in analysis.patterns.key_frame_times[:20]:
        ax.axvline(kf_time, color=Style.ACCENT2, alpha=0.5, linewidth=1)

    ax.set_xlim(0, analysis.duration)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Visual Change')
    ax.set_title(
        f'Visual Rhythm | Tempo: {analysis.patterns.rhythm_tempo:.0f} changes/min '
        f'| Key frames: {len(analysis.patterns.key_frame_indices)}',
        fontweight='bold'
    )


def plot_similarity_matrix(analysis: VideoAnalysis, ax: plt.Axes):
    """
    Render the frame-similarity matrix as a heatmap.

    High values (close to 1) indicate frames that look visually similar;
    diagonal entries are always 1 (identical frame compared with itself).

    Args:
        analysis: The VideoAnalysis result to visualize.
        ax:       The matplotlib Axes to draw on.
    """
    matrix = analysis.patterns.frame_similarity_matrix

    if matrix.size == 0:
        ax.text(0.5, 0.5, 'No similarity data', ha='center', va='center')
        return

    im = ax.imshow(matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Frame')
    ax.set_title(
        f'Frame Similarity | Repetition: {analysis.patterns.repetition_score:.1%}',
        fontweight='bold'
    )
    plt.colorbar(im, ax=ax, label='Similarity')


def plot_statistics_radar(analysis: VideoAnalysis, ax: plt.Axes):
    """
    Render a polar radar chart summarizing six key video metrics.

    All metrics are normalized to the range [0, 1] before plotting.

    Args:
        analysis: The VideoAnalysis result to visualize.
        ax:       A polar Axes (created with subplot_kw={'projection': 'polar'}).
    """
    categories = ['Motion', 'Brightness', 'Saturation', 'Pace', 'Repetition', 'Contrast']
    values = [
        min(analysis.motion.avg_motion / 5, 1),
        analysis.brightness.avg_brightness / 100,
        analysis.colors.avg_saturation / 100,
        min(analysis.scenes.cuts_per_minute / 30, 1),
        analysis.patterns.repetition_score,
        analysis.brightness.avg_contrast / 50,
    ]

    # Close the polygon by appending the first value.
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax.plot(angles, values, 'o-', linewidth=2, color=Style.ACCENT)
    ax.fill(angles, values, alpha=0.3, color=Style.ACCENT)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8)
    ax.set_ylim(0, 1)
    ax.set_title('Video Profile', fontweight='bold', pad=15)


def plot_motion_distribution(analysis: VideoAnalysis, ax: plt.Axes):
    """
    Render a pie chart showing the proportions of static, low, and high motion.

    Segments with fewer than 1% of frames are omitted to avoid clutter.

    Args:
        analysis: The VideoAnalysis result to visualize.
        ax:       The matplotlib Axes to draw on.
    """
    labels = ['Static', 'Low', 'High']
    sizes = [
        analysis.motion.static_ratio,
        analysis.motion.low_motion_ratio,
        analysis.motion.high_motion_ratio,
    ]
    colors_list = [Style.TEXT_DIM, Style.ACCENT, Style.ACCENT2]

    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors_list) if s > 0.01]
    if non_zero:
        labels, sizes, colors_list = zip(*non_zero)
        ax.pie(sizes, labels=labels, colors=colors_list, autopct='%1.0f%%',
               startangle=90, textprops={'fontsize': 9})

    ax.set_title('Motion Distribution', fontweight='bold')


def plot_transition_types(analysis: VideoAnalysis, ax: plt.Axes):
    """
    Render a bar chart of scene transition type counts (cuts, fades, dissolves).

    Args:
        analysis: The VideoAnalysis result to visualize.
        ax:       The matplotlib Axes to draw on.
    """
    labels = ['Cuts', 'Fades', 'Dissolves']
    sizes = [
        analysis.scenes.cut_count,
        analysis.scenes.fade_count,
        analysis.scenes.dissolve_count,
    ]
    colors_list = [Style.ACCENT, Style.ACCENT2, Style.ACCENT3]

    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors_list) if s > 0]
    if non_zero:
        labels, sizes, colors_list = zip(*non_zero)
        bars = ax.bar(labels, sizes, color=colors_list, edgecolor='white')

        for bar, size in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    str(size), ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Count')
    ax.set_title('Transition Types', fontweight='bold')


# ---------------------------------------------------------------------------
# Object detection plot functions
# ---------------------------------------------------------------------------

def plot_object_counts(analysis: VideoAnalysis, ax: plt.Axes):
    """
    Render a horizontal bar chart of the top 12 detected object classes by count.

    Shows a placeholder message if object detection was not run or found nothing.

    Args:
        analysis: The VideoAnalysis result to visualize.
        ax:       The matplotlib Axes to draw on.
    """
    if not analysis.objects or not analysis.objects.object_counts:
        ax.text(0.5, 0.5, 'No objects detected\n(Enable object detection)',
                ha='center', va='center', fontsize=10, color=Style.TEXT_DIM)
        ax.axis('off')
        return

    counts = analysis.objects.object_counts
    sorted_objects = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:12]

    if not sorted_objects:
        ax.text(0.5, 0.5, 'No objects found', ha='center', va='center')
        ax.axis('off')
        return

    labels, values = zip(*sorted_objects)
    colors_list = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))
    bars = ax.barh(range(len(labels)), values, color=colors_list, edgecolor='white')

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Count')
    ax.set_title(f'Detected Objects | {len(counts)} types found', fontweight='bold')
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', fontsize=8)


def plot_object_timeline(analysis: VideoAnalysis, ax: plt.Axes):
    """
    Render a scatter plot showing when each object class appears over time.

    Displays the top 8 object classes by appearance count.

    Args:
        analysis: The VideoAnalysis result to visualize.
        ax:       The matplotlib Axes to draw on.
    """
    if not analysis.objects or not analysis.objects.objects_detected:
        ax.text(0.5, 0.5, 'No object timeline data', ha='center', va='center')
        ax.axis('off')
        return

    # Group detection timestamps by object class.
    object_times: Dict[str, list] = {}
    for obj in analysis.objects.objects_detected:
        object_times.setdefault(obj.label, []).append(obj.frame_time)

    top_objects = sorted(object_times.items(), key=lambda x: len(x[1]), reverse=True)[:8]

    if not top_objects:
        ax.text(0.5, 0.5, 'No objects to display', ha='center', va='center')
        return

    colors_list = plt.cm.tab10(np.linspace(0, 1, len(top_objects)))

    for i, (label, times) in enumerate(top_objects):
        y_vals = [i] * len(times)
        ax.scatter(times, y_vals, c=[colors_list[i]], s=30, alpha=0.7, label=label)

    ax.set_yticks(range(len(top_objects)))
    ax.set_yticklabels([obj[0] for obj in top_objects], fontsize=9)
    ax.set_xlabel('Time (s)')
    ax.set_xlim(0, analysis.duration)
    ax.set_title('Object Appearances Over Time', fontweight='bold')


def plot_object_colors(analysis: VideoAnalysis, ax: plt.Axes):
    """
    Render colored swatches showing the dominant colors observed for each
    detected object class (up to 6 classes, 4 swatches each).

    Args:
        analysis: The VideoAnalysis result to visualize.
        ax:       The matplotlib Axes to draw on.
    """
    if not analysis.objects or not analysis.objects.object_colors:
        ax.text(0.5, 0.5, 'No object color data', ha='center', va='center')
        ax.axis('off')
        return

    obj_colors = {k: v for k, v in analysis.objects.object_colors.items() if v}

    if not obj_colors:
        ax.text(0.5, 0.5, 'No color data for objects', ha='center', va='center')
        ax.axis('off')
        return

    top_objects = list(obj_colors.items())[:6]
    ax.axis('off')

    y_pos = 0.9
    for obj_name, colors in top_objects:
        ax.text(0.05, y_pos, f"{obj_name}:", fontsize=9, fontweight='bold',
                transform=ax.transAxes, va='center')

        x_pos = 0.35
        for color in colors[:4]:
            rect = plt.Rectangle((x_pos, y_pos - 0.04), 0.12, 0.08,
                                  facecolor=color.hex, edgecolor='white',
                                  linewidth=1, transform=ax.transAxes)
            ax.add_patch(rect)
            x_pos += 0.15

        y_pos -= 0.15

    ax.set_title('Object Colors', fontweight='bold')


def plot_object_cooccurrence(analysis: VideoAnalysis, ax: plt.Axes):
    """
    Render a co-occurrence heatmap for the top 8 detected object classes.

    Cell [i, j] shows how many frames contained both object i and object j.

    Args:
        analysis: The VideoAnalysis result to visualize.
        ax:       The matplotlib Axes to draw on.
    """
    if not analysis.objects or not analysis.objects.co_occurrences:
        ax.text(0.5, 0.5, 'No co-occurrence data', ha='center', va='center')
        ax.axis('off')
        return

    all_objects = set()
    for obj, others in analysis.objects.co_occurrences.items():
        all_objects.add(obj)
        all_objects.update(others.keys())

    object_counts = analysis.objects.object_counts
    top_objects = sorted(all_objects,
                         key=lambda x: object_counts.get(x, 0),
                         reverse=True)[:8]

    if len(top_objects) < 2:
        ax.text(0.5, 0.5, 'Not enough objects for co-occurrence',
                ha='center', va='center')
        ax.axis('off')
        return

    n = len(top_objects)
    matrix = np.zeros((n, n))

    for i, obj1 in enumerate(top_objects):
        for j, obj2 in enumerate(top_objects):
            if obj1 in analysis.objects.co_occurrences:
                matrix[i, j] = analysis.objects.co_occurrences[obj1].get(obj2, 0)

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(top_objects, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(top_objects, fontsize=8)
    ax.set_title('Object Co-occurrence', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Count')


# ---------------------------------------------------------------------------
# Composite figure creators
# ---------------------------------------------------------------------------

def create_full_analysis_figure(analysis: VideoAnalysis,
                                output_path: Optional[str] = None) -> plt.Figure:
    """
    Create a comprehensive 5-row dashboard combining all analysis dimensions.

    Layout (5 rows × 4 columns):
      Row 1: Color palette | Scene timeline
      Row 2: Color timeline (×3) | Color transitions
      Row 3: Motion timeline (×2) | Brightness timeline (×2)
      Row 4: Visual rhythm (×2) | Similarity matrix (×2)
      Row 5: Radar | Scene durations | Motion distribution | Transition types

    Args:
        analysis:    The VideoAnalysis result to visualize.
        output_path: If given, the figure is saved as a PNG to this path
                     before being returned.

    Returns:
        The matplotlib Figure (caller is responsible for closing it).
    """
    setup_style()

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.4, wspace=0.3,
                           height_ratios=[0.8, 1, 1, 1, 1.2])

    # Row 1
    plot_color_palette(analysis,   fig.add_subplot(gs[0, :2]))
    plot_scene_timeline(analysis,  fig.add_subplot(gs[0, 2:]))

    # Row 2
    plot_color_timeline(analysis,   fig.add_subplot(gs[1, :3]))
    plot_color_transitions(analysis, fig.add_subplot(gs[1, 3]))

    # Row 3
    plot_motion_timeline(analysis,   fig.add_subplot(gs[2, :2]))
    plot_brightness_timeline(analysis, fig.add_subplot(gs[2, 2:]))

    # Row 4
    plot_visual_rhythm(analysis,     fig.add_subplot(gs[3, :2]))
    plot_similarity_matrix(analysis, fig.add_subplot(gs[3, 2:]))

    # Row 5
    plot_statistics_radar(analysis,   fig.add_subplot(gs[4, 0], polar=True))
    plot_scene_durations(analysis,    fig.add_subplot(gs[4, 1]))
    plot_motion_distribution(analysis, fig.add_subplot(gs[4, 2]))
    plot_transition_types(analysis,   fig.add_subplot(gs[4, 3]))

    title = f'🎬 Video Analysis: {analysis.filename}'
    subtitle = (
        f'{analysis.duration:.1f}s | {analysis.resolution[0]}x{analysis.resolution[1]} '
        f'| {analysis.fps:.1f} FPS | {analysis.frame_count} frames'
    )
    fig.suptitle(f'{title}\n{subtitle}', fontsize=14, fontweight='bold', y=0.98)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=Style.BG_DARK, edgecolor='none')
        print(f"Saved: {output_path}")

    return fig


def create_color_analysis_figure(analysis: VideoAnalysis,
                                 output_path: Optional[str] = None) -> plt.Figure:
    """
    Create a dedicated color analysis figure (3 rows × 2 columns).

    Layout:
      Row 1: Color palette (full width)
      Row 2: Color timeline (full width)
      Row 3: Color transitions | Color statistics text box

    Args:
        analysis:    The VideoAnalysis result to visualize.
        output_path: Optional save path for a PNG export.

    Returns:
        The matplotlib Figure.
    """
    setup_style()

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

    plot_color_palette(analysis, fig.add_subplot(gs[0, :]))
    plot_color_timeline(analysis, fig.add_subplot(gs[1, :]))
    plot_color_transitions(analysis, fig.add_subplot(gs[2, 0]))

    # Color statistics text box
    ax_stats = fig.add_subplot(gs[2, 1])
    ax_stats.axis('off')

    stats_text = f"""
    ╔═══════════════════════════════════════╗
    ║         COLOR STATISTICS              ║
    ╠═══════════════════════════════════════╣
    ║                                       ║
    ║  Temperature: {analysis.colors.temperature.capitalize():<20} ║
    ║  Score: {analysis.colors.temperature_score:+.2f}                        ║
    ║                                       ║
    ║  Mood: {analysis.colors.mood.capitalize():<24} ║
    ║                                       ║
    ║  Avg Saturation: {analysis.colors.avg_saturation:.1f}%               ║
    ║  Avg Brightness: {analysis.colors.avg_brightness:.1f}%               ║
    ║                                       ║
    ║  Dominant Colors:                     ║"""

    for i, (color, pct) in enumerate(
        zip(analysis.colors.dominant_colors[:5], analysis.colors.color_percentages[:5])
    ):
        stats_text += f"\n    ║    {i+1}. {color.name:<12} {pct:>5.1f}%          ║"

    stats_text += """
    ║                                       ║
    ╚═══════════════════════════════════════╝
    """

    ax_stats.text(0.1, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor=Style.BG_PANEL,
                            edgecolor=Style.ACCENT))

    fig.suptitle(f'🎨 Color Analysis: {analysis.filename}',
                 fontsize=14, fontweight='bold', y=0.98)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=Style.BG_DARK)
        print(f"Saved: {output_path}")

    return fig


def create_scene_analysis_figure(analysis: VideoAnalysis,
                                 output_path: Optional[str] = None) -> plt.Figure:
    """
    Create a dedicated scene analysis figure (3 rows × 3 columns).

    Layout:
      Row 1: Scene timeline (full width)
      Row 2: Scene durations | Transition types | Brightness distribution
      Row 3: Scene list table (full width, first 15 scenes)

    Args:
        analysis:    The VideoAnalysis result to visualize.
        output_path: Optional save path for a PNG export.

    Returns:
        The matplotlib Figure.
    """
    setup_style()

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    plot_scene_timeline(analysis, fig.add_subplot(gs[0, :]))
    plot_scene_durations(analysis, fig.add_subplot(gs[1, 0]))
    plot_transition_types(analysis, fig.add_subplot(gs[1, 1]))

    # Scene brightness distribution
    ax_bright = fig.add_subplot(gs[1, 2])
    brightnesses = [s.avg_brightness for s in analysis.scenes.scenes]
    if brightnesses:
        ax_bright.hist(brightnesses, bins=15, color=Style.ACCENT,
                       edgecolor='white', alpha=0.7)
        ax_bright.set_xlabel('Brightness')
        ax_bright.set_ylabel('Count')
    ax_bright.set_title('Scene Brightness Distribution', fontweight='bold')

    # Scene list table
    ax_list = fig.add_subplot(gs[2, :])
    ax_list.axis('off')

    scenes_to_show = analysis.scenes.scenes[:15]
    table_data = [
        [
            f"{s.index + 1}",
            f"{s.start_time:.1f}s",
            f"{s.duration:.1f}s",
            s.dominant_color.name,
            f"{s.avg_brightness:.0f}%",
            s.transition_type,
        ]
        for s in scenes_to_show
    ]

    if table_data:
        table = ax_list.table(
            cellText=table_data,
            colLabels=['#', 'Start', 'Duration', 'Color', 'Brightness', 'Transition'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(Style.ACCENT)
                cell.set_text_props(color='white', fontweight='bold')
            else:
                cell.set_facecolor(Style.BG_PANEL)
                cell.set_text_props(color=Style.TEXT)
            cell.set_edgecolor(Style.GRID)

    fig.suptitle(f'🎬 Scene Analysis: {analysis.filename}',
                 fontsize=14, fontweight='bold', y=0.98)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=Style.BG_DARK)
        print(f"Saved: {output_path}")

    return fig


def create_motion_analysis_figure(analysis: VideoAnalysis,
                                  output_path: Optional[str] = None) -> plt.Figure:
    """
    Create a dedicated motion analysis figure (2 rows × 2 columns).

    Layout:
      Row 1: Motion timeline (full width)
      Row 2: Motion distribution pie | Motion statistics text box

    Args:
        analysis:    The VideoAnalysis result to visualize.
        output_path: Optional save path for a PNG export.

    Returns:
        The matplotlib Figure.
    """
    setup_style()

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    plot_motion_timeline(analysis, fig.add_subplot(gs[0, :]))
    plot_motion_distribution(analysis, fig.add_subplot(gs[1, 0]))

    # Motion statistics text box
    ax_stats = fig.add_subplot(gs[1, 1])
    ax_stats.axis('off')

    stats = f"""
    ╔═══════════════════════════════════════╗
    ║         MOTION STATISTICS             ║
    ╠═══════════════════════════════════════╣
    ║                                       ║
    ║  Motion Type: {analysis.motion.motion_type.capitalize():<18} ║
    ║                                       ║
    ║  Average Motion: {analysis.motion.avg_motion:>6.2f}             ║
    ║  Maximum Motion: {analysis.motion.max_motion:>6.2f}             ║
    ║  Minimum Motion: {analysis.motion.min_motion:>6.2f}             ║
    ║                                       ║
    ║  Static Frames:  {analysis.motion.static_ratio*100:>5.1f}%             ║
    ║  Low Motion:     {analysis.motion.low_motion_ratio*100:>5.1f}%             ║
    ║  High Motion:    {analysis.motion.high_motion_ratio*100:>5.1f}%             ║
    ║                                       ║
    ║  Camera Motion: {'Yes' if analysis.motion.camera_motion_detected else 'No':<18} ║
    ║                                       ║
    ╚═══════════════════════════════════════╝
    """

    ax_stats.text(0.1, 0.9, stats, transform=ax_stats.transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor=Style.BG_PANEL,
                            edgecolor=Style.ACCENT))

    fig.suptitle(f'🏃 Motion Analysis: {analysis.filename}',
                 fontsize=14, fontweight='bold', y=0.98)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=Style.BG_DARK)
        print(f"Saved: {output_path}")

    return fig


def create_object_analysis_figure(analysis: VideoAnalysis,
                                  output_path: Optional[str] = None) -> plt.Figure:
    """
    Create a dedicated object detection figure (2 rows × 2 columns).

    Layout:
      Row 1: Object counts | Object timeline
      Row 2: Object colors | Co-occurrence matrix

    Args:
        analysis:    The VideoAnalysis result to visualize.
        output_path: Optional save path for a PNG export.

    Returns:
        The matplotlib Figure.
    """
    setup_style()

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    plot_object_counts(analysis,      fig.add_subplot(gs[0, 0]))
    plot_object_timeline(analysis,    fig.add_subplot(gs[0, 1]))
    plot_object_colors(analysis,      fig.add_subplot(gs[1, 0]))
    plot_object_cooccurrence(analysis, fig.add_subplot(gs[1, 1]))

    obj_count = len(analysis.objects.object_counts) if analysis.objects else 0
    total_detections = (
        sum(analysis.objects.object_counts.values())
        if analysis.objects and analysis.objects.object_counts else 0
    )
    face_count = analysis.objects.face_count if analysis.objects else 0

    fig.suptitle(
        f'🔍 Object Detection: {analysis.filename}\n'
        f'{obj_count} object types | {total_detections} total detections '
        f'| {face_count} face appearances',
        fontsize=14, fontweight='bold', y=0.98
    )

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=Style.BG_DARK)
        print(f"Saved: {output_path}")

    return fig


if __name__ == "__main__":
    print("Visualizer module loaded.")
    print("Use: create_full_analysis_figure(analysis, 'output.png')")
