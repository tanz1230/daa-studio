@echo off
REM DAA Studio launcher (Windows). Opens the app in your browser.
REM   run.bat                              start screen: create or open a project
REM   run.bat --project PATH               straight into a project
REM   run.bat --host 0.0.0.0 --port 8765   let annotators on other machines connect
setlocal
set "HERE=%~dp0"
set "PY=python"

%PY% -c "import numpy, scipy, yaml, cv2" 1>NUL 2>NUL
if errorlevel 1 (
  echo DAA Studio needs numpy, scipy, pyyaml and opencv. Install them with:
  echo     %PY% -m pip install -r "%HERE%requirements.txt"
  pause
  exit /b 1
)
cd /d "%HERE%"
%PY% -m studio serve %*
pause
