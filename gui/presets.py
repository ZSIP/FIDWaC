"""
FIDWAC v2 GUI — preset configurations
======================================
Mixin class GUIPresets providing:
  _build_presets_panel()
  _apply_preset()
  PRESETS dictionary
"""

import tkinter as tk
from tkinter import ttk

# Preset configurations
PRESETS = {
    "▌ Quick": {
        "accuracy": 0.5,
        "block_size": 8,
        "auto_select": False,
        "rgb_quality": 70,
        "uint8_mode": "quality",
        "files_parallel": 2,
        "description": "Fast compression, moderate quality. Best for large batches.",
    },
    "⚖ Balanced": {
        "accuracy": 0.1,
        "block_size": 16,
        "auto_select": True,
        "rgb_quality": 85,
        "uint8_mode": "accuracy",
        "files_parallel": 1,
        "description": "Balanced speed/quality. Auto-selects best block size.",
    },
    "█ High Quality": {
        "accuracy": 0.01,
        "block_size": 32,
        "auto_select": False,
        "rgb_quality": 95,
        "uint8_mode": "accuracy",
        "files_parallel": 1,
        "description": "Maximum quality, slower compression. Best accuracy.",
    },
    "🔐 Lossless": {
        "accuracy": 0.001,
        "block_size": 32,
        "auto_select": False,
        "rgb_quality": 100,
        "uint8_mode": "lossless",
        "files_parallel": 1,
        "description": "Lossless compression. No data loss.",
    },
}


class GUIPresets:
    """Mixin for preset configurations."""

    def __init__(self):
        """Initialize GUIPresets attributes to None."""
        self.preset_var = None
        self._preset_cb = None
        self._preset_desc = None

    def _build_presets_panel(self, parent):
        """Build presets dropdown on Main tab."""
        f = ttk.LabelFrame(
            parent, text="Presets (Quick Settings)", padding="8", style="Input.TLabelframe"
        )
        f.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=4)
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="Profile:").grid(row=0, column=0, sticky=tk.W)
        self.preset_var = tk.StringVar(value="⚖ Balanced")
        preset_names = list(PRESETS.keys())
        self._preset_cb = ttk.Combobox(
            f,
            textvariable=self.preset_var,
            values=preset_names,
            state="readonly",
            width=20,
        )
        self._preset_cb.grid(row=0, column=1, sticky=tk.W, padx=5)
        self._preset_cb.bind("<<ComboboxSelected>>", lambda e: self._apply_preset())

        # Description label
        self._preset_desc = ttk.Label(
            f, text="", foreground="#888888", font=("TkDefaultFont", 8, "italic"), wraplength=300
        )
        self._preset_desc.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))
        self._update_preset_description()

    def _update_preset_description(self):
        """Update preset description label."""
        preset_name = self.preset_var.get()
        if preset_name in PRESETS:
            desc = PRESETS[preset_name].get("description", "")
            self._preset_desc.config(text=desc)

    def _apply_preset(self):
        """Apply selected preset to GUI controls."""
        preset_name = self.preset_var.get()
        if preset_name not in PRESETS:
            return

        preset = PRESETS[preset_name]
        self._update_preset_description()

        # Apply float settings
        if "accuracy" in preset:
            self.set_accuracy(preset["accuracy"])

        if "block_size" in preset:
            self.block_size.set(str(preset["block_size"]))

        if "auto_select" in preset:
            self.auto_select.set(preset["auto_select"])
            self._on_auto_select_change()

        # Apply uint8 settings
        if "rgb_quality" in preset:
            self.rgb_quality.set(preset["rgb_quality"])
            self._rq_label.config(text=f"{preset['rgb_quality']}%")

        if "uint8_mode" in preset:
            self.uint8_mode.set(preset["uint8_mode"])
            self._on_uint8_mode_change()

        # Apply process settings
        if "files_parallel" in preset:
            self.files_parallel.set(preset["files_parallel"])

        # Log the action
        self._log(f"✓ Preset applied: {preset_name}")


__all__ = ["GUIPresets", "PRESETS"]
