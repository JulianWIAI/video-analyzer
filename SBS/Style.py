"""
SBS/Style.py

Defines the Style class, which centralises all UI color constants used by
the Tkinter-based VideoAnalyzerGUI. Keeping these values in one place makes
global theme changes trivial — update a constant here and every widget that
references it picks up the new value automatically.

The palette is inspired by GitHub's dark-mode color system.
"""


class Style:
    """
    Static color constants for the Tkinter GUI dark theme.

    All values are CSS-style hex color strings. They are used directly as
    the 'background', 'foreground', and similar configuration options for
    Tkinter widgets and ttk styles.

    Attributes:
        BG:           Darkest background — used for the root window and
                      most frame widgets.
        BG_SECONDARY: Slightly lighter background — used for text areas
                      and panel interiors.
        BG_TERTIARY:  Mid-level background — used for hover states and
                      grouped containers.
        ACCENT:       Primary interactive accent (blue) — used for titles,
                      links, and primary action buttons.
        ACCENT2:      Secondary accent (soft red/coral) — used for
                      warnings and secondary highlights.
        SUCCESS:      Success/positive color (green) — used for completed
                      export confirmations.
        TEXT:         Default body text color (light gray).
        TEXT_DIM:     Dimmed/secondary text color — used for labels,
                      placeholders, and status text.
        BORDER:       Border and separator color — used for widget outlines
                      and dividers.
    """

    BG = "#0d1117"
    BG_SECONDARY = "#161b22"
    BG_TERTIARY = "#21262d"
    ACCENT = "#58a6ff"
    ACCENT2 = "#f78166"
    SUCCESS = "#7ee787"
    TEXT = "#c9d1d9"
    TEXT_DIM = "#8b949e"
    BORDER = "#30363d"
