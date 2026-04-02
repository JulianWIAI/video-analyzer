"""
SBS/ColorInfo.py

Defines the ColorInfo dataclass, which represents a single color in multiple
standard formats: RGB tuple, hex string, human-readable English name, and HSV tuple.
This is the fundamental color primitive used throughout the entire analysis pipeline.
"""

import colorsys
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class ColorInfo:
    """
    Represents a color in multiple standard formats simultaneously.

    Rather than storing a color as just an RGB value and converting on demand,
    ColorInfo pre-computes all representations at construction time so that
    downstream consumers (visualizer, export, GUI) can access whichever format
    they need without redundant recalculation.

    Attributes:
        rgb:  The color as an (R, G, B) tuple of integers in the range 0–255.
        hex:  The color as a lowercase CSS-style hex string, e.g. '#ff8800'.
        name: A human-readable English label derived from HSV analysis, such as
              'Vivid Orange' or 'Dark Blue'.
        hsv:  The color in HSV space expressed as (hue°, saturation%, value%),
              all three values as floats.
    """

    rgb: Tuple[int, int, int]
    hex: str
    name: str
    hsv: Tuple[float, float, float]

    @classmethod
    def from_rgb(cls, r: int, g: int, b: int) -> 'ColorInfo':
        """
        Construct a ColorInfo instance from raw RGB component values.

        Converts to hex and HSV automatically, then delegates to
        _get_color_name() to produce a descriptive English label.

        Args:
            r: Red channel, integer 0–255.
            g: Green channel, integer 0–255.
            b: Blue channel, integer 0–255.

        Returns:
            A fully populated ColorInfo with hex, name, and HSV fields
            derived from the supplied RGB values.
        """
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        name = cls._get_color_name(r, g, b)
        return cls(rgb=(r, g, b), hex=hex_color, name=name, hsv=(h * 360, s * 100, v * 100))

    @staticmethod
    def _get_color_name(r: int, g: int, b: int) -> str:
        """
        Map an RGB triplet to a descriptive English color name.

        The method first handles achromatic cases (black, white, grays by value
        and saturation thresholds), then builds a compound name from:
          - a brightness modifier ('Dark ' or 'Light ') based on HSV value, and
          - a saturation modifier ('Pale ' or 'Vivid ') based on HSV saturation,
          - a base hue label ('Red', 'Orange', 'Yellow', …) based on HSV hue.

        When both brightness and saturation modifiers apply, only the more
        distinctive one is prepended to keep the name concise.

        Args:
            r: Red channel, 0–255.
            g: Green channel, 0–255.
            b: Blue channel, 0–255.

        Returns:
            A human-readable color name string (e.g. 'Vivid Orange', 'Dark Blue').
        """
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        h *= 360
        s *= 100
        v *= 100

        # --- Achromatic colors ---
        if v < 10:
            return "Black"
        if v > 90 and s < 10:
            return "White"
        if s < 12:
            if v < 30:
                return "Dark Gray"
            elif v < 60:
                return "Gray"
            else:
                return "Light Gray"

        # --- Brightness modifier ---
        if v < 35:
            brightness = "Dark "
        elif v > 75:
            brightness = "Light "
        else:
            brightness = ""

        # --- Saturation modifier ---
        if s < 40:
            saturation = "Pale "
        elif s > 80:
            saturation = "Vivid "
        else:
            saturation = ""

        # --- Base hue label ---
        if h < 10 or h >= 350:
            hue = "Red"
        elif h < 25:
            hue = "Orange-Red"
        elif h < 40:
            hue = "Orange"
        elif h < 55:
            hue = "Gold"
        elif h < 70:
            hue = "Yellow"
        elif h < 85:
            hue = "Yellow-Green"
        elif h < 150:
            hue = "Green"
        elif h < 175:
            hue = "Teal"
        elif h < 200:
            hue = "Cyan"
        elif h < 230:
            hue = "Blue"
        elif h < 260:
            hue = "Indigo"
        elif h < 290:
            hue = "Purple"
        elif h < 320:
            hue = "Magenta"
        elif h < 350:
            hue = "Pink"
        else:
            hue = "Red"

        # Combine modifiers, favouring the more distinctive one when both apply.
        if brightness and saturation:
            if s > 70:
                return f"{saturation}{hue}"
            else:
                return f"{brightness}{hue}"
        elif brightness:
            return f"{brightness}{hue}"
        elif saturation:
            return f"{saturation}{hue}"
        else:
            return hue

    def to_dict(self) -> Dict:
        """
        Serialize this ColorInfo to a JSON-compatible dictionary.

        Returns:
            A dict with keys 'rgb', 'hex', 'name', and 'hsv'.
            HSV values are rounded to one decimal place.
        """
        return {
            'rgb': self.rgb,
            'hex': self.hex,
            'name': self.name,
            'hsv': [round(x, 1) for x in self.hsv]
        }
