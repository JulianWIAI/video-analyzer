"""
SBS/VideoAnalyzerGUI.py

Defines the VideoAnalyzerGUI class, which provides the Tkinter-based
graphical user interface for the video-analyzer application.

The GUI is organized into three notebook tabs:
  - Analyze: file selection, analysis options, quick stats, and a text preview.
  - Results: full text results and buttons to open each chart type.
  - Export: individual and bulk export controls with a live log.

Analysis runs in a background daemon thread; results are passed back to the
main thread via a thread-safe Queue to avoid Tkinter cross-thread violations.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
from pathlib import Path
from typing import Optional
import json
import webbrowser
import os
import gc

# matplotlib must use the non-interactive Agg backend when driven from a
# worker thread — set this BEFORE importing pyplot.
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from SBS.Style import Style
from SBS.VideoAnalysis import VideoAnalysis
from SBS.VideoAnalyzer import VideoAnalyzer


class VideoAnalyzerGUI:
    """
    Main graphical user interface for the Video Analyzer application.

    Wraps a Tkinter root window with a three-tab notebook layout. Analysis
    runs in a background thread and communicates results back through a
    message queue that is polled every 100 ms on the main thread.

    Attributes:
        SUPPORTED_FORMATS: Tuple of accepted video file extensions,
                           used to filter the file-open dialog.
    """

    SUPPORTED_FORMATS = ('.mp4', '.avi', '.mkv', '.mov', '.webm', '.wmv', '.flv', '.m4v')

    def __init__(self, root: tk.Tk):
        """
        Initialise the GUI and build all widgets.

        Args:
            root: The Tkinter root window. The caller is responsible for
                  calling root.mainloop() after construction.
        """
        self.root = root
        self.root.title("🎬 Video Analyzer")
        self.root.geometry("1100x800")
        self.root.minsize(900, 600)
        self.root.configure(bg=Style.BG)

        # Application state
        self.current_analysis: Optional[VideoAnalysis] = None
        self.current_filepath: Optional[str] = None
        self.output_dir = Path("./output")
        self.output_dir.mkdir(exist_ok=True)

        # Thread-safe message queue: worker thread → main thread.
        self.msg_queue: queue.Queue = queue.Queue()

        self.setup_styles()
        self.create_widgets()
        self.process_messages()   # Start the 100 ms polling loop.

    # ------------------------------------------------------------------
    # Style and widget setup
    # ------------------------------------------------------------------

    def setup_styles(self):
        """Configure ttk widget styles to match the dark theme."""
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')

        style.configure("TFrame",         background=Style.BG)
        style.configure("TLabel",         background=Style.BG, foreground=Style.TEXT)
        style.configure("TButton",        padding=8)
        style.configure("Accent.TButton", padding=10)
        style.configure("TNotebook",      background=Style.BG)
        style.configure("TNotebook.Tab",  padding=[15, 8])
        style.configure("TCheckbutton",   background=Style.BG, foreground=Style.TEXT)

    def create_widgets(self):
        """Create and lay out all top-level GUI widgets."""
        main = ttk.Frame(self.root, padding="10")
        main.pack(fill=tk.BOTH, expand=True)

        self.create_header(main)

        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.create_analyze_tab()
        self.create_results_tab()
        self.create_export_tab()

        self.create_status_bar(main)

    def create_header(self, parent: ttk.Frame):
        """
        Build the application header row containing the title and quick-action buttons.

        Args:
            parent: The parent frame to attach the header to.
        """
        header = ttk.Frame(parent)
        header.pack(fill=tk.X, pady=(0, 5))

        title = tk.Label(header, text="🎬 Video Analyzer",
                         font=("Segoe UI", 22, "bold"),
                         bg=Style.BG, fg=Style.ACCENT)
        title.pack(side=tk.LEFT)

        subtitle = tk.Label(header,
                            text="  Color, Scene, Motion & Pattern Analysis",
                            font=("Segoe UI", 11),
                            bg=Style.BG, fg=Style.TEXT_DIM)
        subtitle.pack(side=tk.LEFT, padx=(5, 0))

        btn_frame = ttk.Frame(header)
        btn_frame.pack(side=tk.RIGHT)

        ttk.Button(btn_frame, text="📁 Open Video",
                   command=self.open_video).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📂 Output Folder",
                   command=self.open_output_folder).pack(side=tk.LEFT, padx=2)

    def create_analyze_tab(self):
        """
        Build the Analyze tab with file selection, options, quick stats, and preview.

        Layout: a fixed-width left panel (file selector + options + stats) and
        an expanding right panel (scrolled text preview of the analysis summary).
        """
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="  🔍 Analyze  ")

        # --- Left panel (fixed width) ---
        left = ttk.Frame(tab, width=400)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        # File selection section
        file_frame = ttk.LabelFrame(left, text="Video File", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))

        self.file_label = tk.Label(file_frame, text="No video selected",
                                   font=("Segoe UI", 9),
                                   bg=Style.BG_SECONDARY, fg=Style.TEXT_DIM,
                                   wraplength=350)
        self.file_label.pack(fill=tk.X, pady=5)

        btn_row = ttk.Frame(file_frame)
        btn_row.pack(fill=tk.X)

        ttk.Button(btn_row, text="Browse...",
                   command=self.browse_video).pack(side=tk.LEFT, padx=(0, 5))

        self.analyze_btn = ttk.Button(btn_row, text="🔍 Analyze",
                                      command=self.start_analysis,
                                      style="Accent.TButton")
        self.analyze_btn.pack(side=tk.LEFT)

        formats = tk.Label(file_frame,
                           text="Supports: MP4, AVI, MKV, MOV, WEBM, WMV, FLV",
                           font=("Segoe UI", 8),
                           bg=Style.BG_SECONDARY, fg=Style.TEXT_DIM)
        formats.pack(anchor=tk.W, pady=(5, 0))

        # Analysis options section
        options_frame = ttk.LabelFrame(left, text="Analysis Options", padding="10")
        options_frame.pack(fill=tk.X, pady=(0, 10))

        # Sample rate selector
        sample_frame = ttk.Frame(options_frame)
        sample_frame.pack(fill=tk.X, pady=2)

        ttk.Label(sample_frame, text="Sample Rate:").pack(side=tk.LEFT)
        self.sample_rate_var = tk.StringVar(value="2")
        sample_combo = ttk.Combobox(sample_frame, textvariable=self.sample_rate_var,
                                    values=["1", "2", "4", "8"], width=5, state="readonly")
        sample_combo.pack(side=tk.LEFT, padx=5)
        ttk.Label(sample_frame, text="frames/sec (higher = slower but more accurate)",
                  font=("Segoe UI", 8)).pack(side=tk.LEFT)

        # Object detection toggle
        self.detect_objects_var = tk.BooleanVar(value=False)
        obj_check = ttk.Checkbutton(options_frame, text="Enable Object Detection",
                                    variable=self.detect_objects_var)
        obj_check.pack(anchor=tk.W, pady=5)

        # YOLO model size selector
        model_frame = ttk.Frame(options_frame)
        model_frame.pack(fill=tk.X, pady=2)

        ttk.Label(model_frame, text="   Model Size:").pack(side=tk.LEFT)
        self.model_size_var = tk.StringVar(value="medium")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_size_var,
                                   values=["nano", "small", "medium", "large", "xlarge"],
                                   width=8, state="readonly")
        model_combo.pack(side=tk.LEFT, padx=5)

        tk.Label(model_frame, text="(larger = slower but better)",
                 font=("Segoe UI", 8), bg=Style.BG, fg=Style.TEXT_DIM).pack(side=tk.LEFT)

        # Quick stats section
        stats_frame = ttk.LabelFrame(left, text="Quick Stats", padding="10")
        stats_frame.pack(fill=tk.BOTH, expand=True)

        self.stats_text = tk.Label(stats_frame,
                                   text="Load a video to see analysis",
                                   font=("Consolas", 9),
                                   bg=Style.BG_SECONDARY, fg=Style.TEXT,
                                   justify=tk.LEFT, anchor=tk.NW)
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        # --- Right panel (expanding) ---
        right = ttk.Frame(tab)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        preview_frame = ttk.LabelFrame(right, text="Analysis Preview", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True)

        self.preview_text = scrolledtext.ScrolledText(
            preview_frame,
            font=("Consolas", 10),
            bg=Style.BG_SECONDARY,
            fg=Style.TEXT,
            insertbackground=Style.TEXT,
            wrap=tk.WORD
        )
        self.preview_text.pack(fill=tk.BOTH, expand=True)

    def create_results_tab(self):
        """
        Build the Results tab with a scrolled text area for detailed output
        and a row of buttons to open individual chart windows.
        """
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="  📊 Results  ")

        self.results_text = scrolledtext.ScrolledText(
            tab,
            font=("Consolas", 10),
            bg=Style.BG_SECONDARY,
            fg=Style.TEXT,
            wrap=tk.WORD
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(btn_frame, text="📊 Full Analysis Chart",
                   command=self.show_full_chart).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🎨 Color Analysis",
                   command=self.show_color_chart).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🎬 Scene Analysis",
                   command=self.show_scene_chart).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🏃 Motion Analysis",
                   command=self.show_motion_chart).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🔍 Object Detection",
                   command=self.show_object_chart).pack(side=tk.LEFT, padx=2)

    def create_export_tab(self):
        """
        Build the Export tab with individual export controls for each output
        type (JSON, PNG charts, thumbnails) and a bulk 'Export Everything' button.
        A scrolled log area shows real-time progress of export operations.
        """
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="  💾 Export  ")

        export_frame = ttk.LabelFrame(tab, text="Export Options", padding="15")
        export_frame.pack(fill=tk.X, pady=(0, 10))

        # Helper to add a labeled row with an action button.
        def _row(label: str, btn_text: str, command):
            row = ttk.Frame(export_frame)
            row.pack(fill=tk.X, pady=5)
            ttk.Label(row, text=label).pack(side=tk.LEFT)
            ttk.Button(row, text=btn_text, command=command).pack(side=tk.RIGHT)

        _row("📄 Analysis Data (JSON)",          "Export JSON",         self.export_json)
        _row("📊 Full Analysis Chart (PNG)",      "Export Chart",        self.export_full_chart)
        _row("🎨 Color Analysis Chart (PNG)",     "Export Chart",        self.export_color_chart)
        _row("🎬 Scene Analysis Chart (PNG)",     "Export Chart",        self.export_scene_chart)
        _row("🏃 Motion Analysis Chart (PNG)",    "Export Chart",        self.export_motion_chart)
        _row("🔍 Object Detection Chart (PNG)",   "Export Chart",        self.export_object_chart)
        _row("🖼️ Scene Thumbnails",              "Extract Thumbnails",  self.extract_thumbnails)

        ttk.Button(export_frame, text="📦 Export Everything",
                   command=self.export_all,
                   style="Accent.TButton").pack(pady=15)

        # Export activity log
        log_frame = ttk.LabelFrame(tab, text="Export Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.export_log = scrolledtext.ScrolledText(
            log_frame,
            font=("Consolas", 9),
            bg=Style.BG_SECONDARY,
            fg=Style.TEXT,
            height=10
        )
        self.export_log.pack(fill=tk.BOTH, expand=True)

    def create_status_bar(self, parent: ttk.Frame):
        """
        Build the status bar at the bottom of the window.

        Contains a status label on the left and an indeterminate progress bar
        on the right. The progress bar spins while analysis is running.

        Args:
            parent: The frame to attach the status bar to.
        """
        status = ttk.Frame(parent)
        status.pack(fill=tk.X, pady=(10, 0))

        self.status_label = tk.Label(status, text="Ready",
                                     font=("Segoe UI", 9),
                                     bg=Style.BG, fg=Style.TEXT_DIM)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress = ttk.Progressbar(status, mode='indeterminate', length=150)
        self.progress.pack(side=tk.RIGHT)

    # ------------------------------------------------------------------
    # Status and progress helpers
    # ------------------------------------------------------------------

    def set_status(self, text: str):
        """
        Update the status bar label text.

        Args:
            text: The message to display in the status bar.
        """
        self.status_label.config(text=text)
        self.root.update_idletasks()

    def start_progress(self):
        """Start the indeterminate progress bar animation."""
        self.progress.start(10)

    def stop_progress(self):
        """Stop the progress bar animation."""
        self.progress.stop()

    def process_messages(self):
        """
        Drain the inter-thread message queue on the main Tkinter thread.

        Recognised message types:
          'status'        – update the status bar label.
          'analysis_done' – display the completed VideoAnalysis result.
          'error'         – show an error dialog.
          'done'          – stop the progress animation.
          'log'           – append a line to the export log widget.

        Reschedules itself every 100 ms via root.after() to keep the GUI
        responsive without blocking.
        """
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                if msg['type'] == 'status':
                    self.set_status(msg['text'])
                elif msg['type'] == 'analysis_done':
                    self.display_analysis(msg['data'])
                elif msg['type'] == 'error':
                    messagebox.showerror("Error", msg['text'])
                elif msg['type'] == 'done':
                    self.stop_progress()
                elif msg['type'] == 'log':
                    self.export_log.insert(tk.END, msg['text'] + '\n')
                    self.export_log.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages)

    # ------------------------------------------------------------------
    # File and analysis actions
    # ------------------------------------------------------------------

    def browse_video(self):
        """Open a file-picker dialog and store the selected video path."""
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mkv *.mov *.webm *.wmv *.flv *.m4v"),
                ("All Files", "*.*")
            ]
        )
        if filepath:
            self.current_filepath = filepath
            self.file_label.config(text=Path(filepath).name, fg=Style.TEXT)

    def open_video(self):
        """Open a file-picker dialog and immediately start analysis if a file is chosen."""
        self.browse_video()
        if self.current_filepath:
            self.start_analysis()

    def open_output_folder(self):
        """Open the output folder in the system's file explorer."""
        self.output_dir.mkdir(exist_ok=True)
        if os.name == 'nt':
            os.startfile(str(self.output_dir))
        else:
            webbrowser.open(f'file://{self.output_dir}')

    def start_analysis(self):
        """
        Validate inputs and launch the analysis worker thread.

        Tkinter variables are read on the main thread before spawning the
        worker to avoid the 'main thread is not in main loop' RuntimeError
        that occurs when StringVar/BooleanVar are accessed from other threads.
        """
        # Guard: check that required modules loaded successfully.
        try:
            from SBS.Visualizer import (
                create_full_analysis_figure,
                create_color_analysis_figure,
                create_scene_analysis_figure,
                create_motion_analysis_figure,
                create_object_analysis_figure,
            )
        except ImportError as e:
            messagebox.showerror("Error", f"Required modules not available:\n{e}")
            return

        if not self.current_filepath:
            messagebox.showwarning("Warning", "Please select a video file")
            return

        self.analyze_btn.config(state='disabled')
        self.start_progress()
        self.preview_text.delete(1.0, tk.END)
        self.preview_text.insert(
            tk.END,
            "Analyzing video...\n\nThis may take a while for long videos.\n"
        )

        # Read all Tkinter variable values here, on the main thread.
        sample_rate = int(self.sample_rate_var.get())
        detect_objects = bool(self.detect_objects_var.get())
        model_size = str(self.model_size_var.get())
        filepath = str(self.current_filepath)

        thread = threading.Thread(
            target=self._analyze_worker,
            args=(filepath, sample_rate, detect_objects, model_size),
            daemon=True
        )
        thread.start()

    def _analyze_worker(self, filepath: str, sample_rate: int,
                        detect_objects: bool, model_size: str):
        """
        Background worker: run VideoAnalyzer and post results to the queue.

        This method runs entirely in a daemon thread. All GUI interactions
        go through self.msg_queue to stay on the main thread.

        Args:
            filepath: Absolute path to the video file.
            sample_rate: Frames to analyze per second.
            detect_objects: Whether to enable YOLOv8 object detection.
            model_size: YOLO model size string ('nano' … 'xlarge').
        """
        analyzer = None
        try:
            analyzer = VideoAnalyzer(sample_rate=sample_rate)

            def safe_progress(msg):
                """Thread-safe wrapper for posting progress to the queue."""
                try:
                    self.msg_queue.put({'type': 'status', 'text': msg})
                except Exception:
                    pass

            analyzer.set_progress_callback(safe_progress)

            analysis = analyzer.analyze(
                filepath,
                detect_objects=detect_objects,
                object_model_size=model_size
            )

            self.msg_queue.put({'type': 'analysis_done', 'data': analysis})
            self.msg_queue.put({'type': 'status', 'text': 'Analysis complete!'})

        except Exception as e:
            import traceback
            error_msg = f"{e}\n\n{traceback.format_exc()}"
            self.msg_queue.put({'type': 'error', 'text': error_msg})

        finally:
            # Release the YOLO model and force garbage collection.
            try:
                if analyzer is not None:
                    if hasattr(analyzer, '_yolo_model'):
                        analyzer._yolo_model = None
                    del analyzer
            except Exception:
                pass
            gc.collect()

            self.msg_queue.put({'type': 'done'})

            try:
                self.root.after(100, self._enable_analyze_button)
            except Exception:
                pass

    def _enable_analyze_button(self):
        """Re-enable the Analyze button after a worker thread finishes."""
        try:
            self.analyze_btn.config(state='normal')
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Result display
    # ------------------------------------------------------------------

    def display_analysis(self, analysis: VideoAnalysis):
        """
        Populate all result widgets with data from a completed analysis.

        Updates the quick-stats label, the preview text area, and the full
        results text area, then switches to the Results tab.

        Args:
            analysis: The VideoAnalysis result returned by VideoAnalyzer.analyze().
        """
        self.current_analysis = analysis

        # Quick stats (left panel of the Analyze tab)
        stats = (
            f"\n🎬 {analysis.filename}\n"
            f"⏱️ {analysis.duration:.1f}s | {analysis.fps:.1f} FPS\n"
            f"📐 {analysis.resolution[0]}x{analysis.resolution[1]}\n\n"
            f"🎨 Colors: {analysis.colors.temperature} | {analysis.colors.mood}\n"
            f"🎬 Scenes: {analysis.scenes.total_scenes} | {analysis.scenes.pace_category}\n"
            f"🏃 Motion: {analysis.motion.motion_type}\n"
            f"☀️ Brightness: {analysis.brightness.brightness_category}\n"
        )
        self.stats_text.config(text=stats)

        preview = self._format_preview(analysis)
        self.preview_text.delete(1.0, tk.END)
        self.preview_text.insert(tk.END, preview)

        results = self._format_full_results(analysis)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, results)

        self.notebook.select(1)  # Switch to the Results tab.

    def _format_preview(self, a: VideoAnalysis) -> str:
        """
        Build a concise formatted summary string of the analysis results.

        Args:
            a: A completed VideoAnalysis instance.

        Returns:
            A multi-line string suitable for display in the preview text area.
        """
        colors_str = ", ".join(
            f"{c.name} ({p:.0f}%)"
            for c, p in zip(a.colors.dominant_colors[:3], a.colors.color_percentages[:3])
        )

        preview = f"""
╔══════════════════════════════════════════════════════════════╗
║                    VIDEO ANALYSIS COMPLETE                    ║
╠══════════════════════════════════════════════════════════════╣

📁 File: {a.filename}
⏱️  Duration: {a.duration:.1f} seconds
📐 Resolution: {a.resolution[0]} x {a.resolution[1]}
🎞️  Frames: {a.frame_count} @ {a.fps:.1f} FPS

═══════════════════════════════════════════════════════════════

🎨 COLOR ANALYSIS
   Dominant: {colors_str}
   Temperature: {a.colors.temperature.capitalize()} (score: {a.colors.temperature_score:+.2f})
   Mood: {a.colors.mood.capitalize()}
   Saturation: {a.colors.avg_saturation:.1f}%
   Brightness: {a.colors.avg_brightness:.1f}%

🎬 SCENE ANALYSIS
   Total Scenes: {a.scenes.total_scenes}
   Avg Duration: {a.scenes.avg_scene_duration:.2f}s
   Cuts/Minute: {a.scenes.cuts_per_minute:.1f}
   Pace: {a.scenes.pace_category.capitalize()}
   Transitions: {a.scenes.cut_count} cuts, {a.scenes.fade_count} fades, {a.scenes.dissolve_count} dissolves

🏃 MOTION ANALYSIS
   Type: {a.motion.motion_type.capitalize()}
   Average: {a.motion.avg_motion:.2f}
   Static: {a.motion.static_ratio*100:.1f}%
   High Motion: {a.motion.high_motion_ratio*100:.1f}%
   Camera Motion: {'Detected' if a.motion.camera_motion_detected else 'Not detected'}

☀️ BRIGHTNESS
   Average: {a.brightness.avg_brightness:.1f}%
   Contrast: {a.brightness.avg_contrast:.1f}%
   Category: {a.brightness.brightness_category.capitalize()}
   Dark Scenes: {a.brightness.dark_ratio*100:.1f}%
   Bright Scenes: {a.brightness.bright_ratio*100:.1f}%

🔄 VISUAL PATTERNS
   Repetition Score: {a.patterns.repetition_score:.1%}
   Visual Tempo: {a.patterns.rhythm_tempo:.0f} changes/min
   Key Frames: {len(a.patterns.key_frame_indices)}

╚══════════════════════════════════════════════════════════════╝
"""
        # Append object detection section if data is available.
        if a.objects and a.objects.object_counts:
            top_objects = sorted(
                a.objects.object_counts.items(),
                key=lambda x: x[1], reverse=True
            )[:5]
            objects_str = ", ".join(f"{name} ({count})" for name, count in top_objects)
            preview += f"""
═══════════════════════════════════════════════════════════════

🔍 OBJECT DETECTION
   Objects Found: {len(a.objects.object_counts)} types
   Total Detections: {sum(a.objects.object_counts.values())}
   Face Appearances: {a.objects.face_count}

   Top Objects: {objects_str}

╚══════════════════════════════════════════════════════════════╝
"""
        preview += "\nClick the buttons below to view detailed charts!\n"
        return preview

    def _format_full_results(self, a: VideoAnalysis) -> str:
        """
        Build a detailed formatted results string including the scene list
        and top color-transition pairs.

        Args:
            a: A completed VideoAnalysis instance.

        Returns:
            A multi-line string extending the preview with scene and
            transition tables.
        """
        result = self._format_preview(a)

        result += (
            "\n\n═══════════════════════════════════════════════════════════════\n"
            "                         SCENE LIST\n"
            "═══════════════════════════════════════════════════════════════\n\n"
        )

        for s in a.scenes.scenes[:30]:
            result += (
                f"  Scene {s.index+1:3d} | {s.start_time:6.1f}s - {s.end_time:6.1f}s | "
                f"{s.duration:5.1f}s | {s.dominant_color.name:8s} | "
                f"Brightness: {s.avg_brightness:4.0f}% | {s.transition_type}\n"
            )

        if len(a.scenes.scenes) > 30:
            result += f"\n  ... and {len(a.scenes.scenes) - 30} more scenes\n"

        result += (
            "\n═══════════════════════════════════════════════════════════════\n"
            "                      COLOR TRANSITIONS\n"
            "═══════════════════════════════════════════════════════════════\n\n"
        )

        transitions = [
            (from_c, to_c, count)
            for from_c, to_colors in a.colors.color_transitions.items()
            for to_c, count in to_colors.items()
        ]
        transitions.sort(key=lambda x: x[2], reverse=True)

        for from_c, to_c, count in transitions[:15]:
            result += f"  {from_c:10s} → {to_c:10s}: {count:4d} times\n"

        return result

    # ------------------------------------------------------------------
    # Chart display helpers
    # ------------------------------------------------------------------

    def _require_analysis(self) -> bool:
        """
        Check that a completed analysis is available; show a dialog if not.

        Returns:
            True if self.current_analysis is populated, False otherwise.
        """
        if not self.current_analysis:
            messagebox.showinfo("Info", "No analysis available")
            return False
        return True

    def _open_chart(self, create_fn, suffix: str):
        """
        Generate a chart, save it to the output folder, and open it in the browser.

        Args:
            create_fn: A visualizer function that accepts (analysis, output_path)
                       and returns a matplotlib Figure.
            suffix:    Filename suffix to distinguish chart types
                       (e.g. '_full_analysis.png').
        """
        stem = Path(self.current_analysis.filename).stem
        output_path = self.output_dir / f"{stem}{suffix}"
        fig = create_fn(self.current_analysis, str(output_path))
        plt.close(fig)
        webbrowser.open(str(output_path))

    def show_full_chart(self):
        """Generate and open the full analysis chart."""
        if not self._require_analysis():
            return
        from SBS.Visualizer import create_full_analysis_figure
        self.set_status("Generating chart...")
        self._open_chart(create_full_analysis_figure, "_full_analysis.png")
        self.set_status("Chart opened")

    def show_color_chart(self):
        """Generate and open the color analysis chart."""
        if not self._require_analysis():
            return
        from SBS.Visualizer import create_color_analysis_figure
        self._open_chart(create_color_analysis_figure, "_color_analysis.png")

    def show_scene_chart(self):
        """Generate and open the scene analysis chart."""
        if not self._require_analysis():
            return
        from SBS.Visualizer import create_scene_analysis_figure
        self._open_chart(create_scene_analysis_figure, "_scene_analysis.png")

    def show_motion_chart(self):
        """Generate and open the motion analysis chart."""
        if not self._require_analysis():
            return
        from SBS.Visualizer import create_motion_analysis_figure
        self._open_chart(create_motion_analysis_figure, "_motion_analysis.png")

    def show_object_chart(self):
        """Generate and open the object detection chart, or show a help dialog if unavailable."""
        if not self._require_analysis():
            return
        if not self.current_analysis.objects:
            messagebox.showinfo(
                "Info",
                "No object detection data available.\n\n"
                "To enable object detection:\n"
                "1. Check 'Enable Object Detection' before analyzing\n"
                "2. Make sure ultralytics is installed:\n"
                "   pip install ultralytics"
            )
            return
        from SBS.Visualizer import create_object_analysis_figure
        self._open_chart(create_object_analysis_figure, "_object_analysis.png")

    # ------------------------------------------------------------------
    # Export actions
    # ------------------------------------------------------------------

    def _require_analysis_for_export(self) -> bool:
        """
        Like _require_analysis, but shows 'No analysis to export' message.

        Returns:
            True if an analysis is ready, False otherwise.
        """
        if not self.current_analysis:
            messagebox.showinfo("Info", "No analysis to export")
            return False
        return True

    def export_json(self):
        """Prompt for a save path and write the analysis data as JSON."""
        if not self._require_analysis_for_export():
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile=f"{Path(self.current_analysis.filename).stem}_analysis.json"
        )
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(self.current_analysis.to_dict(), f, indent=2)
            self.export_log.insert(tk.END, f"✓ Exported: {filepath}\n")
            self.set_status(f"Saved: {filepath}")

    def _export_chart(self, create_fn, suffix: str):
        """
        Prompt for a save path and export a single chart PNG.

        Args:
            create_fn: A visualizer figure-creation function.
            suffix:    Default filename suffix (e.g. '_color_analysis.png').
        """
        if not self._require_analysis_for_export():
            return
        stem = Path(self.current_analysis.filename).stem
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")],
            initialfile=f"{stem}{suffix}"
        )
        if filepath:
            fig = create_fn(self.current_analysis, filepath)
            plt.close(fig)
            self.export_log.insert(tk.END, f"✓ Exported: {filepath}\n")

    def export_full_chart(self):
        """Export the full analysis chart to a user-specified path."""
        from SBS.Visualizer import create_full_analysis_figure
        self._export_chart(create_full_analysis_figure, "_full_analysis.png")

    def export_color_chart(self):
        """Export the color analysis chart to a user-specified path."""
        from SBS.Visualizer import create_color_analysis_figure
        self._export_chart(create_color_analysis_figure, "_color_analysis.png")

    def export_scene_chart(self):
        """Export the scene analysis chart to a user-specified path."""
        from SBS.Visualizer import create_scene_analysis_figure
        self._export_chart(create_scene_analysis_figure, "_scene_analysis.png")

    def export_motion_chart(self):
        """Export the motion analysis chart to a user-specified path."""
        from SBS.Visualizer import create_motion_analysis_figure
        self._export_chart(create_motion_analysis_figure, "_motion_analysis.png")

    def export_object_chart(self):
        """Export the object detection chart, or show a dialog if no data exists."""
        if not self._require_analysis_for_export():
            return
        if not self.current_analysis.objects:
            messagebox.showinfo("Info", "No object detection data available")
            return
        from SBS.Visualizer import create_object_analysis_figure
        self._export_chart(create_object_analysis_figure, "_object_analysis.png")

    def extract_thumbnails(self):
        """
        Prompt for an output folder and extract JPEG scene thumbnails.

        Uses VideoAnalyzer.extract_thumbnails() to seek directly to each
        scene's representative frame.
        """
        if not self.current_analysis or not self.current_filepath:
            messagebox.showinfo("Info", "No analysis available")
            return

        folder = filedialog.askdirectory(title="Select folder for thumbnails")
        if folder:
            analyzer = VideoAnalyzer()
            saved = analyzer.extract_thumbnails(
                self.current_filepath, folder, self.current_analysis, max_thumbnails=20
            )
            self.export_log.insert(tk.END, f"✓ Extracted {len(saved)} thumbnails to {folder}\n")
            messagebox.showinfo("Done", f"Extracted {len(saved)} thumbnails")

    def export_all(self):
        """
        Export all outputs (JSON, all charts, thumbnails) to a chosen folder.

        Shows a live log of each file written. If object detection data is
        available its chart is included; otherwise it is skipped silently.
        """
        if not self._require_analysis_for_export():
            return

        folder = filedialog.askdirectory(title="Select export folder")
        if not folder:
            return

        folder = Path(folder)
        base_name = Path(self.current_analysis.filename).stem

        self.export_log.delete(1.0, tk.END)
        self.export_log.insert(tk.END, "Exporting all files...\n\n")

        from SBS.Visualizer import (
            create_full_analysis_figure,
            create_color_analysis_figure,
            create_scene_analysis_figure,
            create_motion_analysis_figure,
            create_object_analysis_figure,
        )

        try:
            # JSON
            json_path = folder / f"{base_name}_analysis.json"
            with open(json_path, 'w') as f:
                json.dump(self.current_analysis.to_dict(), f, indent=2)
            self.export_log.insert(tk.END, f"✓ {json_path.name}\n")

            # Charts — generate and immediately close to free memory.
            for create_fn, suffix in [
                (create_full_analysis_figure,   "_full_analysis.png"),
                (create_color_analysis_figure,  "_color_analysis.png"),
                (create_scene_analysis_figure,  "_scene_analysis.png"),
                (create_motion_analysis_figure, "_motion_analysis.png"),
            ]:
                out = folder / f"{base_name}{suffix}"
                fig = create_fn(self.current_analysis, str(out))
                plt.close(fig)
                self.export_log.insert(tk.END, f"✓ {out.name}\n")

            # Object detection chart (optional)
            if self.current_analysis.objects:
                out = folder / f"{base_name}_object_analysis.png"
                fig = create_object_analysis_figure(self.current_analysis, str(out))
                plt.close(fig)
                self.export_log.insert(tk.END, f"✓ {out.name}\n")

            # Thumbnails
            thumb_folder = folder / "thumbnails"
            thumb_folder.mkdir(exist_ok=True)
            if self.current_filepath:
                analyzer = VideoAnalyzer()
                saved = analyzer.extract_thumbnails(
                    self.current_filepath, str(thumb_folder),
                    self.current_analysis, max_thumbnails=15
                )
                self.export_log.insert(tk.END, f"✓ {len(saved)} thumbnails\n")

            self.export_log.insert(tk.END, f"\n✅ All files exported to {folder}\n")
            messagebox.showinfo("Done", f"Exported all files to {folder}")

        except Exception as e:
            self.export_log.insert(tk.END, f"\n❌ Error: {e}\n")
            messagebox.showerror("Error", str(e))
