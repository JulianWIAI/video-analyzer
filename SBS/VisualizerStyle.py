"""
SBS/VisualizerStyle.py

Defines the VisualizerStyle class, which centralises all color constants
used by the matplotlib-based visualization module (visualizer.py). Having
a single source of truth for chart colors makes it easy to adjust the
dark-theme palette without hunting through every plotting function.

The palette is inspired by GitHub's dark-mode color system and is designed
for high contrast against a very dark (#0d1117) figure background.
"""


class VisualizerStyle:
    """
    Static color constants for the matplotlib chart dark theme.

    All values are CSS-style hex color strings compatible with matplotlib's
    color arguments.

    Attributes:
        BG_DARK:   Darkest background, used for the figure face color.
        BG_PANEL:  Panel/axes background, slightly lighter than BG_DARK.
        ACCENT:    Primary data accent (blue) — used for line plots and
                   the dominant bar color.
        ACCENT2:   Secondary data accent (soft red/coral) — used for
                   highlighted regions and secondary series.
        ACCENT3:   Tertiary data accent (green) — used for the visual
                   rhythm plot and success indicators.
        TEXT:      Default axis label and tick text color.
        TEXT_DIM:  Dimmed text used for secondary tick labels and notes.
        GRID:      Grid line and axes edge color.
    """

    BG_DARK = '#0d1117'
    BG_PANEL = '#161b22'
    ACCENT = '#58a6ff'
    ACCENT2 = '#f78166'
    ACCENT3 = '#7ee787'
    TEXT = '#c9d1d9'
    TEXT_DIM = '#8b949e'
    GRID = '#30363d'
