"""
FIDWAC v2 GUI — statistics tracking and display
================================================
Mixin class GUIStats providing:
  _build_stats_panel()
  _init_stats()
  _update_stats()
  _show_final_stats()
"""

import tkinter as tk
from tkinter import ttk
import os


class GUIStats:
    """Mixin for compression statistics tracking."""

    def __init__(self):
        """Initialize GUIStats attributes."""
        self._stats_files_count = 0
        self._stats_files_processed = 0
        self._stats_size_original = 0
        self._stats_size_compressed = 0
        self._stats_time_start = 0
        self._stats_errors = 0
        self._stats_frame = None
        self._stats_labels = {}

    def _build_stats_panel(self, parent, row=4, column=0, columnspan=2):
        """Build statistics display panel."""
        self._stats_frame = ttk.LabelFrame(
            parent, text="Statistics", padding="8", style="Perf.TLabelframe"
        )
        self._stats_frame.grid(
            row=row, column=column, columnspan=columnspan, sticky=(tk.W, tk.E), padx=4, pady=4
        )
        self._stats_frame.columnconfigure(1, weight=1)

        self._stats_labels.clear()
        stats_data = [
            ("Files Processed", "files_processed", "0"),
            ("Original Size", "size_original", "0 B"),
            ("Compressed Size", "size_compressed", "0 B"),
            ("Compression Ratio", "ratio", "—"),
            ("Errors", "errors", "0"),
        ]

        for idx, (label_text, key, initial) in enumerate(stats_data):
            lbl = ttk.Label(self._stats_frame, text=f"{label_text}:")
            lbl.grid(row=idx, column=0, sticky=tk.W, pady=2)
            val = ttk.Label(
                self._stats_frame,
                text=initial,
                foreground="#1565C0",
                font=("TkFixedFont", 9, "bold"),
            )
            val.grid(row=idx, column=1, sticky=tk.W, pady=2, padx=(10, 0))
            self._stats_labels[key] = val

    def _format_size(self, size_bytes: int) -> str:
        """Format bytes to human-readable size."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"

    def _update_stats_file(self, input_path: str, output_path: str = None):
        """Update stats after processing a file."""
        try:
            if os.path.exists(input_path):
                orig_size = os.path.getsize(input_path)
                self._stats_size_original += orig_size

                if output_path and os.path.exists(output_path):
                    comp_size = os.path.getsize(output_path)
                    self._stats_size_compressed += comp_size
                elif output_path:
                    import sys

                    sys.stderr.write(f"Warning: Output path does not exist: {output_path}\n")

                self._stats_files_processed += 1
                self._update_stats_display()
            else:
                import sys

                sys.stderr.write(f"Warning: Input path does not exist: {input_path}\n")
        except OSError as e:
            import sys

            sys.stderr.write(f"Error updating stats for {input_path}: {e}\n")

    def _update_stats_display(self):
        """Update statistics labels in GUI (thread-safe)."""

        def _do():
            if "files_processed" in self._stats_labels:
                self._stats_labels["files_processed"].config(text=str(self._stats_files_processed))

            if "size_original" in self._stats_labels:
                self._stats_labels["size_original"].config(
                    text=self._format_size(self._stats_size_original)
                )

            if "size_compressed" in self._stats_labels:
                self._stats_labels["size_compressed"].config(
                    text=self._format_size(self._stats_size_compressed)
                )

            if "ratio" in self._stats_labels:
                if self._stats_size_original > 0:
                    ratio = self._stats_size_compressed / self._stats_size_original * 100
                    self._stats_labels["ratio"].config(text=f"{ratio:.1f}%")
                else:
                    self._stats_labels["ratio"].config(text="—")

            if "errors" in self._stats_labels:
                self._stats_labels["errors"].config(
                    text=str(self._stats_errors),
                    foreground="#ff6b6b" if self._stats_errors > 0 else "#1565C0",
                )

        self.root.after(0, _do)

    def _reset_stats(self):
        """Reset statistics before new compression run."""
        self._stats_files_count = 0
        self._stats_files_processed = 0
        self._stats_size_original = 0
        self._stats_size_compressed = 0
        self._stats_time_start = 0
        self._stats_errors = 0
        self._update_stats_display()

    def _record_error(self):
        """Record an error in statistics."""
        self._stats_errors += 1
        self._update_stats_display()


__all__ = ["GUIStats"]
