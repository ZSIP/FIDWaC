@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo ========================================
echo Running FIDWAC v2 WSL2
echo ========================================
echo.

REM Check WSL
wsl --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] WSL2 is not installed
    echo Run first: install\setup_windows.bat
    pause
    exit /b 1
)
echo [OK] WSL2 available
echo.

REM Read configuration from file
set "CONFIG_FILE=%~dp0install\wsl_config.txt"
if not exist "!CONFIG_FILE!" (
    echo [ERROR] Configuration file not found: !CONFIG_FILE!
    echo Run first: install\setup_windows.bat
    pause
    exit /b 1
)

for /f "usebackq tokens=1,* delims==" %%a in ("!CONFIG_FILE!") do (
    if "%%a"=="WSL_DIST_NAME" set "WSL_DIST_NAME=%%b"
    if "%%a"=="WSL_USER" set "WSL_USER=%%b"
    if "%%a"=="VENV_PATH_WSL" set "VENV_PATH_WSL=%%b"
)
set "WSL_DIST_NAME=!WSL_DIST_NAME: =!"
set "WSL_USER=!WSL_USER: =!"
set "VENV_PATH_WSL=!VENV_PATH_WSL: =!"

if "!WSL_DIST_NAME!"=="" (
    echo [ERROR] Distribution name not found in configuration
    echo Run first: install\setup_windows.bat
    pause
    exit /b 1
)

echo Using distribution: !WSL_DIST_NAME!
echo Using user: !WSL_USER!
echo.

REM Check if distribution works
echo Checking distribution !WSL_DIST_NAME!...
wsl -d %WSL_DIST_NAME% echo test >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Distribution !WSL_DIST_NAME! is not running
    echo Run first: install\setup_windows.bat
    pause
    exit /b 1
)
echo [OK] Distribution !WSL_DIST_NAME! is running
echo.

REM Check if virtual environment exists
if not defined VENV_PATH_WSL (
    echo [WARNING] Virtual environment path not found in configuration
    echo Using default: /home/!WSL_USER!/.fidwac/venv
    set "VENV_PATH_WSL=/home/!WSL_USER!/.fidwac/venv"
)

REM Checking virtual environment...
wsl -d %WSL_DIST_NAME% bash -c "[ -d '%VENV_PATH_WSL%' ]" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Virtual environment not found at: %VENV_PATH_WSL%
    echo.
    echo =====================================================================
    echo WSL ENVIRONMENT ERROR
    echo =====================================================================
    echo Virtual environment is missing or was corrupted.
    echo.
    echo To resolve this issue, please:
    echo   1. Delete the 'install\wsl_config.txt' file in this folder.
    echo   2. Run the installer again: install\setup_windows.bat
    echo =====================================================================
    echo.
    pause
    exit /b 1
)
echo [OK] Virtual environment exists
echo.

REM Convert Windows path to WSL path
set "SCRIPT_PATH=%~dp0"
for /f "delims=" %%p in ('wsl -d %WSL_DIST_NAME% wslpath -a "%SCRIPT_PATH:~0,-1%"') do set "SCRIPT_PATH_WSL=%%p"
if not defined SCRIPT_PATH_WSL (
    echo [ERROR] Cannot convert Windows path to WSL
    pause
    exit /b 1
)

REM Check optional backends (JPEG cm=1, SZ3 cm=8) in the WSL venv
echo Checking optional backends...
wsl -d %WSL_DIST_NAME% bash -c "cd '!SCRIPT_PATH_WSL!' && source '%VENV_PATH_WSL%/bin/activate' && python3 install/check_backends.py"
echo.

REM Choose the interface in a window (Yes = desktop Tkinter, No = web, Cancel = quit).
REM Pass tk or web as the first argument to skip the window.
set "FIDWAC_GUI="
if /i "%~1"=="tk" set "FIDWAC_GUI=tk"
if /i "%~1"=="web" set "FIDWAC_GUI=web"
if not defined FIDWAC_GUI (
    for /f "usebackq delims=" %%c in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install\choose_gui.ps1"`) do set "FIDWAC_GUI=%%c"
)
if not defined FIDWAC_GUI set "FIDWAC_GUI=web"
if "%FIDWAC_GUI%"=="quit" exit /b 0
echo Interface: %FIDWAC_GUI%
echo.

REM Run Linux script in WSL2
echo Running application in WSL2 distribution: !WSL_DIST_NAME!...
echo.
wsl -d %WSL_DIST_NAME% bash -c "export VENV_PATH='%VENV_PATH_WSL%' FIDWAC_BACKENDS_CHECKED=1 FIDWAC_GUI='%FIDWAC_GUI%' && cd '!SCRIPT_PATH_WSL!' && bash run_linux.sh"

if errorlevel 1 (
    echo.
    echo [ERROR] Application failed to start
    echo.
    echo =====================================================================
    echo WSL ENVIRONMENT RUNTIME ERROR
    echo =====================================================================
    echo If this is an environment setup or path issue, please:
    echo   1. Delete the 'install\wsl_config.txt' file in this folder.
    echo   2. Run the installer again: install\setup_windows.bat
    echo =====================================================================
    echo.
    pause
    exit /b 1
)
    exit /b 1
)
