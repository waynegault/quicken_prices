@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "VENV_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"

if not exist "%PROJECT_DIR%quicken_prices.py" (
    echo [ERROR] Could not find quicken_prices.py at:
    echo         %PROJECT_DIR%
    pause
    exit /b 1
)

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Project virtual environment is missing:
    echo         %VENV_PYTHON%
    echo Create it with: py -3.14 -m venv .venv
    pause
    exit /b 1
)

pushd "%PROJECT_DIR%" || (
    echo [ERROR] Failed to switch to project directory:
    echo         %PROJECT_DIR%
    pause
    exit /b 1
)

"%VENV_PYTHON%" "quicken_prices.py"
set "EXIT_CODE=%ERRORLEVEL%"

popd

if not "%EXIT_CODE%"=="0" (
    echo [ERROR] QuickenPrices exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
