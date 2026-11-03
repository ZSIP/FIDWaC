#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIDWAC v2 - GUI (orchestrator)
================================
Actual logic lives in gui/ sub-package:
  gui/widgets.py  — _CanvasProgressBar, _StreamToQueue, ToolTip, TOOLTIPS
  gui/theme.py    — apply_theme()
  gui/panels.py   — GUIPanels mixin (_build_main_tab, _build_advanced_tab, …)
  gui/runner.py   — GUIRunner mixin (_run, _stop, _process_*, _build_config, …)
"""

import os
import sys

import tkinter as tk
from tkinter import ttk
import queue
import threading
from config import load_config
from gui.theme import apply_theme
from gui.panels import GUIPanels
from gui.runner import GUIRunner
from gui.stats import GUIStats
from gui.presets import GUIPresets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))




class FidvacGUI(GUIPanels, GUIRunner, GUIStats, GUIPresets):
    """FIDWAC v2 GUI main class."""

    def __init__(self, root):
        """Initialize the GUI."""
        GUIPanels.__init__(self)
        GUIStats.__init__(self)
        GUIRunner.__init__(self)
        GUIPresets.__init__(self)
        self.root = root
        self.root.title("FIDWAC v2 — DCT Compression")
        self.root.geometry("1200x800")
        self.root.minsize(1050, 600)

        self._log_queue = queue.Queue()
        self._cpu_count = os.cpu_count() or 1
        self._stop_event = threading.Event()

        cfg = load_config(None)

        apply_theme(root)

        outer = ttk.Frame(root, padding="10")
        outer.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        outer.columnconfigure(0, weight=4)  # Left side (Controls / Notebook)
        outer.columnconfigure(1, weight=5)  # Right side (Logs / Stats - wider)
        outer.rowconfigure(0, weight=1)

        left_pane = ttk.Frame(outer)
        left_pane.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_pane.columnconfigure(0, weight=1)
        left_pane.rowconfigure(0, weight=1)  # Notebook expands vertically

        right_pane = ttk.Frame(outer)
        right_pane.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_pane.columnconfigure(0, weight=1)
        right_pane.rowconfigure(0, weight=1)  # Log expands vertically
        right_pane.rowconfigure(1, weight=0)  # Stats remains fixed size below log

        nb = ttk.Notebook(left_pane)
        nb.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        tab_main = ttk.Frame(nb, padding="10")
        tab_adv = ttk.Frame(nb, padding="10")
        nb.add(tab_main, text="  Main  ")
        nb.add(tab_adv, text="  Advanced  ")

        tab_main.columnconfigure(0, weight=1)
        tab_adv.columnconfigure(0, weight=1)
        tab_adv.columnconfigure(1, weight=1)

        self._build_main_tab(tab_main, cfg)
        self._build_advanced_tab(tab_adv, cfg)
        self._on_backend_change()

        self._build_buttons(left_pane, row=1)
        self._build_progress(left_pane, row=2)

        self._build_log(right_pane, row=0)
        self._build_stats_panel(right_pane, row=1, column=0, columnspan=1)

        self.root.after(100, self._poll_log)


def main():
    """Run the FIDWAC v2 GUI application."""
    root = tk.Tk()
    root.withdraw()  # Hide before building UI
    FidvacGUI(root)
    root.update_idletasks()
    
    # Center window on screen and ensure it's not off-screen
    def _center_and_show():
        # Get window dimensions
        window_width = root.winfo_width()
        window_height = root.winfo_height()
        
        # Get screen dimensions
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Calculate center position
        x = max(0, (screen_width - window_width) // 2)
        y = max(0, (screen_height - window_height) // 2)
        
        # Ensure window doesn't go off-screen
        x = min(x, screen_width - window_width)
        y = min(y, screen_height - window_height)
        
        root.geometry(f"+{x}+{y}")
        root.deiconify()
        root.lift()
        root.focus_force()
    
    root.after(50, _center_and_show)
    root.mainloop()


if __name__ == "__main__":
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
