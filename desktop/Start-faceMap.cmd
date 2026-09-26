@echo off
setlocal
pushd "%~dp0"
py -3.12 -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
if not errorlevel 1 (
  py -3.12 bootstrap.py %*
  goto finished
)
py -3.11 -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
if not errorlevel 1 (
  py -3.11 bootstrap.py %*
  goto finished
)
python -c "import sys; assert (3,11) <= sys.version_info[:2] <= (3,12) and sys.maxsize > 2**32" >nul 2>&1
if not errorlevel 1 (
  python bootstrap.py %*
  goto finished
)
echo Install 64-bit Python 3.11 or 3.12 from https://www.python.org/downloads/
echo Include Tcl/Tk and the Python launcher, then run this file again.
pause
popd
exit /b 1
:finished
set "FACEMAP_RESULT=%ERRORLEVEL%"
if not "%FACEMAP_RESULT%"=="0" pause
popd
exit /b %FACEMAP_RESULT%
