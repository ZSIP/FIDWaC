"""
FIDWAC v2 GUI — Material Design theme
"""

from tkinter import ttk


def apply_theme(root):
    """Apply a colorful Material Design-inspired theme."""
    style = ttk.Style(root)

    if "clam" in style.theme_names():
        style.theme_use("clam")

    C_PRIMARY = "#1565C0"
    C_PRIMARY_DK = "#0D47A1"
    C_ACCENT = "#FF6F00"
    C_SUCCESS = "#2E7D32"
    C_DANGER = "#C62828"
    C_BG = "#F5F5F5"
    C_SURFACE = "#FFFFFF"
    C_TEXT = "#212121"
    C_TEXT_SEC = "#757575"
    C_BORDER = "#BDBDBD"

    style.configure(
        ".",
        background=C_BG,
        foreground=C_TEXT,
        fieldbackground=C_SURFACE,
        bordercolor=C_BORDER,
        focuscolor=C_PRIMARY,
    )
    style.configure("TFrame", background=C_BG)
    style.configure("TLabel", background=C_BG, foreground=C_TEXT)
    style.configure(
        "TLabelframe", background=C_BG, foreground=C_PRIMARY, bordercolor=C_PRIMARY, relief="groove"
    )
    style.configure(
        "TLabelframe.Label",
        background=C_BG,
        foreground=C_PRIMARY,
        font=("TkDefaultFont", 10, "bold"),
    )
    style.configure("TNotebook", background=C_BG, bordercolor=C_BORDER)
    style.configure(
        "TNotebook.Tab",
        background="#E3F2FD",
        foreground=C_PRIMARY_DK,
        padding=[14, 4],
        font=("TkDefaultFont", 10, "bold"),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", C_PRIMARY)],
        foreground=[("selected", "#FFFFFF")],
        expand=[("selected", [0, 0, 0, 2])],
    )

    style.configure(
        "TButton",
        background=C_PRIMARY,
        foreground="#FFFFFF",
        padding=[10, 4],
        font=("TkDefaultFont", 9, "bold"),
        bordercolor=C_PRIMARY_DK,
        focuscolor=C_ACCENT,
    )
    style.map(
        "TButton",
        background=[("active", C_PRIMARY_DK), ("disabled", "#B0BEC5")],
        foreground=[("disabled", "#ECEFF1")],
    )

    style.configure(
        "Run.TButton",
        background=C_SUCCESS,
        foreground="#FFFFFF",
        padding=[14, 6],
        font=("TkDefaultFont", 11, "bold"),
    )
    style.map("Run.TButton", background=[("active", "#1B5E20"), ("disabled", "#A5D6A7")])

    style.configure(
        "Stop.TButton",
        background=C_DANGER,
        foreground="#FFFFFF",
        padding=[10, 4],
        font=("TkDefaultFont", 9, "bold"),
    )
    style.map("Stop.TButton", background=[("active", "#B71C1C"), ("disabled", "#EF9A9A")])

    style.configure(
        "Accent.TButton",
        background=C_ACCENT,
        foreground="#FFFFFF",
        padding=[8, 3],
        font=("TkDefaultFont", 9, "bold"),
    )
    style.map("Accent.TButton", background=[("active", "#E65100")])

    style.configure(
        "TEntry",
        fieldbackground=C_SURFACE,
        foreground=C_TEXT,
        bordercolor=C_BORDER,
        focuscolor=C_PRIMARY,
    )
    style.configure(
        "TCombobox",
        fieldbackground=C_SURFACE,
        foreground=C_TEXT,
        selectbackground=C_PRIMARY,
        selectforeground="#FFFFFF",
    )
    style.configure("TSpinbox", fieldbackground=C_SURFACE, foreground=C_TEXT)
    style.configure("TCheckbutton", background=C_BG, foreground=C_TEXT)
    style.configure("TRadiobutton", background=C_BG, foreground=C_TEXT)
    style.configure("TScale", background=C_BG, troughcolor="#BBDEFB", sliderlength=20)

    for name, fg in [
        ("Input", C_PRIMARY),
        ("Output", "#6A1B9A"),
        ("Compress", "#00695C"),
        ("Float", "#0277BD"),
        ("Uint8", "#E65100"),
        ("Progress", C_SUCCESS),
        ("Model", "#AD1457"),
        ("Perf", "#1A237E"),
    ]:
        style.configure(f"{name}.TLabelframe", foreground=fg, bordercolor=fg)
        style.configure(f"{name}.TLabelframe.Label", foreground=fg)

    style.configure("Log.TLabelframe", foreground=C_TEXT_SEC, bordercolor=C_BORDER)
    style.configure("Log.TLabelframe.Label", foreground=C_TEXT_SEC)

    root.configure(bg=C_BG)


__all__ = ["apply_theme"]
