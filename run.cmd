@echo off
rem ==============================================================
rem  Project launcher - always use the project virtual environment
rem  (E:\venv) because it has PyTorch with CUDA support.
rem
rem  The system Python from the Microsoft Store installs packages
rem  into a very deep folder path, which makes the PyTorch install
rem  fail (Windows 260-character path limit). E:\venv is short.
rem
rem  Usage:
rem     run.cmd cli.py analyze video1.mp4 --out results
rem     run.cmd tools/export_frames.py --start 16 --end 50 --step 1
rem
rem  Install new packages with:
rem     E:\venv\Scripts\python.exe -m pip install <package>
rem
rem  NOTE: keep this file ASCII-only. Non-ASCII characters break
rem        cmd.exe on Chinese Windows (it reads .cmd as GBK).
rem ==============================================================

cd /d "%~dp0"
"E:\venv\Scripts\python.exe" %*
exit /b %ERRORLEVEL%
