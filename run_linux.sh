#!/bin/bash
# FIDWAC v2 - Linux Run Script (for WSL2)
# Activates virtual environment and launches the application

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Use VENV_PATH from environment (set by run_windows.bat), or default location
if [ -n "$VENV_PATH" ]; then
    VENV_DIR="$VENV_PATH"
else
    VENV_DIR="$HOME/.fidwac/venv"
fi

cd "$PROJECT_ROOT"

echo "========================================"
echo "Running FIDWAC v2"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "[ERROR] Virtual environment not found at: $VENV_DIR"
    echo ""
    echo "====================================================================="
    echo "WSL ENVIRONMENT ERROR"
    echo "====================================================================="
    echo "Virtual environment is missing or was corrupted."
    echo "To resolve this issue, please:"
    echo "  1. Delete the 'install/wsl_config.txt' file in Windows/Linux"
    echo "  2. Run the installer again: install/setup_windows.bat (from Windows)"
    echo "====================================================================="
    echo ""
    exit 1
fi

echo "[OK] Virtual environment detected: $VENV_DIR"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Check if the entry points exist
for entry in app.py app_tk.py; do
    if [ ! -f "$entry" ]; then
        echo "[ERROR] $entry not found in: $PROJECT_ROOT"
        echo ""
        echo "Make sure you're running this script from the FIDWAC v2 project root directory."
        exit 1
    fi
done

echo "[OK] app.py and app_tk.py detected"
echo ""

# Choose the interface: --gui tk|web on the command line, or FIDWAC_GUI (set by run_windows.bat).
# Without either: a zenity window when a display is available, otherwise a question in the terminal.
GUI_CHOICE="${FIDWAC_GUI:-}"
APP_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --gui) GUI_CHOICE="$2"; shift 2 ;;
        --gui=*) GUI_CHOICE="${1#--gui=}"; shift ;;
        *) APP_ARGS+=("$1"); shift ;;
    esac
done

if [ -z "$GUI_CHOICE" ]; then
    if [ -n "$DISPLAY" ] && command -v zenity >/dev/null 2>&1; then
        GUI_CHOICE=$(zenity --list --title="FIDWAC v2" --text="Which interface should start?" \
            --column=key --column=Interface --hide-column=1 \
            tk "Desktop GUI (Tkinter)" web "Web GUI (browser)" 2>/dev/null) || exit 0
    elif [ -t 0 ]; then
        echo "Which interface should start?"
        echo "  1) Desktop GUI (Tkinter)"
        echo "  2) Web GUI (browser)"
        read -r -p "Selection [2]: " answer
        case "$answer" in
            1) GUI_CHOICE=tk ;;
            *) GUI_CHOICE=web ;;
        esac
    else
        GUI_CHOICE=web
    fi
fi

case "$GUI_CHOICE" in
    tk)
        if ! python3 -c "import tkinter" 2>/dev/null; then
            echo "[WARNING] tkinter is not available in the virtual environment; starting the web GUI"
            GUI_CHOICE=web
        fi
        ;;
    web) ;;
    *)
        echo "[ERROR] unknown interface: $GUI_CHOICE (use tk or web)"
        exit 1
        ;;
esac
echo "Interface: $GUI_CHOICE"
echo ""

# Optional backends: JPEG quality mode (cm=1) and SZ3 (cm=8). Skipped when run_windows.bat has already checked them.
if [ -z "${FIDWAC_BACKENDS_CHECKED:-}" ]; then
    python3 install/check_backends.py
    echo ""
fi

# Launch the application
echo "Launching application..."
echo ""
if [ "$GUI_CHOICE" = "tk" ]; then
    python3 app_tk.py "${APP_ARGS[@]}"
else
    python3 app.py "${APP_ARGS[@]}"
fi
