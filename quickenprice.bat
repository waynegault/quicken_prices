@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "VENV_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"

if not exist "%PROJECT_DIR%QuickenPrices.py" (
    echo [ERROR] Could not find QuickenPrices.py at:
    echo         %PROJECT_DIR%
    pause
    exit /b 1
)

pushd "%PROJECT_DIR%" || (
    echo [ERROR] Failed to switch to project directory:
    echo         %PROJECT_DIR%
    pause
    exit /b 1
)

if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" "QuickenPrices.py"
) else (
    py -3 "QuickenPrices.py"
)
set "EXIT_CODE=%ERRORLEVEL%"

popd

if not "%EXIT_CODE%"=="0" (
    echo [ERROR] QuickenPrices exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
