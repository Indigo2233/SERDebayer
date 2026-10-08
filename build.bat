@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo Installing pack tools...
python -m pip install -q -r requirements.txt pyinstaller
if errorlevel 1 (
  echo pip failed
  pause
  exit /b 1
)

echo Building dist\SERDebayer ...
python -m PyInstaller --noconfirm --clean ser_debayer.spec
if errorlevel 1 (
  echo PyInstaller failed
  pause
  exit /b 1
)

echo.
echo Done. Zip this folder and send it:
echo   %~dp0dist\SERDebayer
echo GUI:  SERDebayer.exe
echo CLI:  SERDebayerCLI.exe info/convert ...
echo Do not ship only the exe; keep the whole folder.
pause
