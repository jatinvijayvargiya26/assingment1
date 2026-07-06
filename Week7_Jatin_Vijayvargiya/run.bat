@echo off
chcp 65001 >nul
set PYTHONUTF8=1
echo.
echo ============================================================
echo   Week 7 - Document Question Answering System (RAG)
echo ============================================================
echo.
echo Usage:
echo   run.bat                    - Demo mode (built-in document)
echo   run.bat --interactive      - Interactive Q^&A mode
echo   run.bat --txt file.txt     - Use your own TXT file
echo   run.bat --pdf file.pdf     - Use your own PDF
echo   run.bat --hybrid           - Hybrid search mode
echo.

if "%1"=="" (
    echo [INFO] Running demo mode with built-in sample document...
    echo.
    python main.py %*
) else (
    python main.py %*
)
