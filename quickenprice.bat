@echo off
setlocal

set "PROJECT_DIR=C:\Users\wayne\GitHub\Python\Projects\quicken_prices"
set "PYTHON_EXE=C:\Users\wayne\AppData\Local\Programs\Python\Python314\python.exe"

if not exist "%PROJECT_DIR%\QuickenPrices.py" (
    echo [ERROR] Could not find QuickenPrices.py at:
    echo         %PROJECT_DIR%
    pause
    exit /b 1
)

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python executable not found at:
    echo         %PYTHON_EXE%
    pause
    exit /b 1
)

pushd "%PROJECT_DIR%" || (
    echo [ERROR] Failed to switch to project directory:
    echo         %PROJECT_DIR%
    pause
    exit /b 1
)

"%PYTHON_EXE%" "QuickenPrices.py"
set "EXIT_CODE=%ERRORLEVEL%"

popd

if not "%EXIT_CODE%"=="0" (
    echo [ERROR] QuickenPrices exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
