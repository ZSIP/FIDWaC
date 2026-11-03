@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo ========================================
echo FIDWAC v2 WSL2 Installer
echo ========================================
echo.

REM Check WSL
wsl --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] WSL2 is not installed
    echo Run: wsl --install
    pause
    exit /b 1
)
echo [OK] WSL2 available
echo.

REM Get list of existing distributions via PowerShell and temporary file to avoid UTF-16, NUL chars and CMD bugs
echo Detecting WSL distributions...
set "DIST_FILE=%TEMP%\fidwac_wsl_dists_%RANDOM%.txt"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$lines = @(& wsl.exe --list --quiet 2>$null | ForEach-Object { ($_ -replace [char]0, '').Trim() } | Where-Object { $_ }); [System.IO.File]::WriteAllText($env:DIST_FILE, [string]::Join([char]10, $lines))"

set DIST_COUNT=0
if exist "%DIST_FILE%" (
    for /f "usebackq delims=" %%d in ("%DIST_FILE%") do (
        set "val=%%d"
        if not "!val!"=="" (
            set /a DIST_COUNT+=1
            set "DIST_!DIST_COUNT!=!val!"
            echo   !DIST_COUNT!. !val!
        )
    )
)

if !DIST_COUNT! equ 0 (
    echo [WARNING] No WSL distributions found
    echo.
    set /p "INSTALL_DEBIAN=Do you want to install Debian? (Y/N): "
    if /i "!INSTALL_DEBIAN!"=="Y" (
        echo Installing Debian...
        wsl --install -d Debian
        echo.
        echo Installation completed. Please restart your computer and run this script again.
        pause
        exit /b 0
    ) else (
        echo Installation cancelled.
        pause
        exit /b 1
    )
)

echo.
echo Options:
echo   1-%DIST_COUNT% - Select existing distribution from the list above
echo   0              - Install new Debian distribution
echo.

set /p "DIST_CHOICE=Enter your choice (default: 1): "
set "DIST_CHOICE=!DIST_CHOICE: =!"
if not defined DIST_CHOICE set "DIST_CHOICE=1"

if "!DIST_CHOICE!"=="0" (
    set /p "WSL_DIST_NAME=Enter name for new distribution (default: fidwac): "
    if not defined WSL_DIST_NAME set "WSL_DIST_NAME=fidwac"
    echo Installing Debian as !WSL_DIST_NAME!...
    wsl --install -d Debian
    if errorlevel 1 (
        echo [ERROR] Failed to install Debian
        pause
        exit /b 1
    )
    echo [OK] Debian installed. Please restart and run this script again.
    pause
    exit /b 0
)

REM Select existing distribution
set "WSL_DIST_NAME="
set SELECT_INDEX=0
if exist "%DIST_FILE%" (
    for /f "usebackq delims=" %%d in ("%DIST_FILE%") do (
        set /a SELECT_INDEX+=1
        if "!SELECT_INDEX!"=="!DIST_CHOICE!" (
            set "WSL_DIST_NAME=%%d"
        )
    )
    del "%DIST_FILE%" >nul 2>&1
)
if "!WSL_DIST_NAME!"=="" (
    echo [ERROR] Invalid choice: !DIST_CHOICE!
    pause
    exit /b 1
)
set "WSL_DIST_NAME=!WSL_DIST_NAME: =!"
echo Using distribution: !WSL_DIST_NAME!
echo.

REM Verify distribution works
echo Checking distribution [!WSL_DIST_NAME!]...
if not defined WSL_DIST_NAME (
    echo [ERROR] Distribution name is empty.
    pause
    exit /b 1
)
wsl -d %WSL_DIST_NAME% echo test >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Distribution [!WSL_DIST_NAME!] is not running or failed to start
    echo.
    echo Try manually: wsl -d %WSL_DIST_NAME% echo test
    pause
    exit /b 1
)
echo [OK] Distribution !WSL_DIST_NAME! is running
echo.

REM List existing users in the distribution via temporary file to avoid pipe and encoding issues
echo Detecting users in !WSL_DIST_NAME!...
set "USERS_FILE=%TEMP%\fidwac_wsl_users_%RANDOM%.txt"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$lines = @(& wsl.exe -d $env:WSL_DIST_NAME bash -c 'getent passwd {1000..65535} | cut -d: -f1' 2>$null | ForEach-Object { ($_ -replace [char]0, '').Trim() } | Where-Object { $_ }); [System.IO.File]::WriteAllText($env:USERS_FILE, [string]::Join([char]10, $lines))"

set USER_COUNT=0
if exist "%USERS_FILE%" (
    for /f "usebackq delims=" %%u in ("%USERS_FILE%") do (
        set "val=%%u"
        if not "!val!"=="" (
            set /a USER_COUNT+=1
            set "USER_!USER_COUNT!=!val!"
            echo   !USER_COUNT!. !val!
        )
    )
)

if !USER_COUNT! gtr 0 (
    echo.
    echo Options:
    echo   1-!USER_COUNT! - Select existing user from the list above
    echo   0              - Create new user
    echo.
    set /p "USER_CHOICE=Enter your choice (default: 1): "
    set "USER_CHOICE=!USER_CHOICE: =!"
    if not defined USER_CHOICE set "USER_CHOICE=1"

    if "!USER_CHOICE!"=="0" (
        set /p "WSL_USER=Enter username (default: fidwac): "
        if not defined WSL_USER set "WSL_USER=fidwac"
        goto create_user
    )

    set "WSL_USER="
    set SELECT_INDEX=0
    if exist "%USERS_FILE%" (
        for /f "usebackq delims=" %%u in ("%USERS_FILE%") do (
            set /a SELECT_INDEX+=1
            if "!SELECT_INDEX!"=="!USER_CHOICE!" (
                set "WSL_USER=%%u"
            )
        )
        del "%USERS_FILE%" >nul 2>&1
    )
    if "!WSL_USER!"=="" (
        echo [ERROR] Invalid choice
        pause
        exit /b 1
    )
    set "WSL_USER=!WSL_USER: =!"
    echo Using user: !WSL_USER!
    goto user_ready
) else (
    if exist "%USERS_FILE%" del "%USERS_FILE%" >nul 2>&1
    echo No regular users found in distribution.
    set /p "WSL_USER=Enter username (default: fidwac): "
    if not defined WSL_USER set "WSL_USER=fidwac"
    set "WSL_USER=!WSL_USER: =!"
)

:create_user
set "WSL_USER=%WSL_USER: =%"
echo Using username: %WSL_USER%
echo.
echo Creating user %WSL_USER%...
wsl -d %WSL_DIST_NAME% bash -c "if id -u %WSL_USER% > /dev/null 2>&1; then echo User already exists; else useradd -m -s /bin/bash %WSL_USER% && echo '%WSL_USER%:%WSL_USER%' | chpasswd && usermod -aG sudo %WSL_USER% && echo User created; fi"
if errorlevel 1 (
    echo [ERROR] Failed to create user
    pause
    exit /b 1
)
echo [OK] User %WSL_USER% ready

:user_ready
echo.

REM Set default user in WSL config
echo Setting default user in WSL config...
wsl -d %WSL_DIST_NAME% -u root -- sh -c "printf '[user]\ndefault=%s\n' '%WSL_USER%' > /etc/wsl.conf"
if errorlevel 1 (
    echo [ERROR] Failed to set default user in WSL config
    pause
    exit /b 1
)
echo [OK] Default user set
echo.

REM Save configuration to file for run_windows.bat
echo Saving configuration...
set "CONFIG_FILE=%~dp0wsl_config.txt"
set "VENV_PATH_WSL=/home/%WSL_USER%/.fidwac/venv"
> "%CONFIG_FILE%" echo WSL_DIST_NAME=%WSL_DIST_NAME%
>> "%CONFIG_FILE%" echo WSL_USER=%WSL_USER%
>> "%CONFIG_FILE%" echo VENV_PATH_WSL=%VENV_PATH_WSL%
echo [OK] Configuration saved to: %CONFIG_FILE%
echo.

REM Convert Windows path to WSL path
set "SCRIPT_PATH=%~dp0.."
for /f "delims=" %%p in ('wsl -d %WSL_DIST_NAME% wslpath -u "%SCRIPT_PATH%"') do set "SCRIPT_PATH_WSL=%%p"
if "%SCRIPT_PATH_WSL%"=="" (
    echo [ERROR] Cannot convert Windows path to WSL
    pause
    exit /b 1
)

echo Project path in WSL: %SCRIPT_PATH_WSL%
echo.

REM Run Linux setup script
echo Running installation in WSL2 distribution: %WSL_DIST_NAME%...
wsl -d %WSL_DIST_NAME% bash -c "cd '%SCRIPT_PATH_WSL%' && bash install/setup_linux.sh"
if errorlevel 1 (
    echo [ERROR] Installation in WSL2 failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo INSTALLATION COMPLETED
echo ========================================
echo.
echo To run the application, execute: run_windows.bat
echo.
pause
