"""
main.py  —  Video Analyzer application entry point.

Launch the GUI with:

    python main.py
"""

import tkinter as tk

from SBS.VideoAnalyzerGUI import VideoAnalyzerGUI


def main():
    """Create the root Tkinter window, instantiate the GUI, and start the event loop."""
    root = tk.Tk()
    app = VideoAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
