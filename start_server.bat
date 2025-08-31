@echo off
echo ========================================
echo    ML Model Hosting Script
echo ========================================
echo.

echo Starting Pile Settlement Prediction App...
echo.

cd /d "C:\Users\Ziad\Desktop\Master Thesis\WEB_3"

echo Current directory: %CD%
echo.

echo Option 1: Development Server (Flask)
echo Option 2: Production Server (Gunicorn)
echo Option 3: Network Accessible Server
echo.

set /p choice="Choose option (1-3): "

if "%choice%"=="1" (
    echo Starting Flask development server...
    python app.py
) else if "%choice%"=="2" (
    echo Starting Gunicorn production server...
    gunicorn --bind 0.0.0.0:5000 app:app
) else if "%choice%"=="3" (
    echo Starting network accessible server...
    echo Your app will be accessible to other devices on your network.
    echo Find your IP address and share: http://YOUR_IP:5000
    ipconfig | findstr "IPv4"
    echo.
    python app.py
) else (
    echo Invalid choice. Starting default server...
    python app.py
)

pause
