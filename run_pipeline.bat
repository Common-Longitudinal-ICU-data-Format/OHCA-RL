@echo off
REM OHCA-RL Pipeline Runner (Windows)
REM Runs all pipeline steps sequentially.
REM Usage: run_pipeline.bat
REM
REM Logs: Each script writes its own log to output\final\<script_name>.log
REM        Combined pipeline output is saved to output\final\pipeline_<timestamp>.log

setlocal enabledelayedexpansion

REM Change to the directory where this script lives
cd /d "%~dp0"

REM Create output directory
if not exist "output\final" mkdir "output\final"

REM Generate timestamped log filename (YYYYMMDD_HHMMSS)
set "DATESTAMP=%date:~-4%%date:~-10,2%%date:~-7,2%"
set "HOUR=%time:~0,2%"
if "%HOUR:~0,1%"==" " set "HOUR=0%HOUR:~1,1%"
set "TIMESTAMP=%DATESTAMP%_%HOUR%%time:~3,2%%time:~6,2%"
set "LOG_FILE=output\final\pipeline_%TIMESTAMP%.log"

echo ==========================================
echo   OHCA-RL Pipeline
echo   Log: %LOG_FILE%
echo ==========================================

REM Log header
echo ========================================== > "%LOG_FILE%"
echo   OHCA-RL Pipeline >> "%LOG_FILE%"
echo   Started: %date% %time% >> "%LOG_FILE%"
echo ========================================== >> "%LOG_FILE%"

REM Pipeline steps
set STEPS[0]=code\00_cohort_identification.py
set STEPS[1]=code\01_create_wide_df.py
set STEPS[2]=code\02_sofa_calculator.py
set STEPS[3]=code\03_ffill_and_bucketing.py
set STEPS[4]=code\04_create_tableone.py
set STEPS[5]=code\05_figures.py

for %%i in (0 1 2 3 4 5) do (
    echo.
    echo ------------------------------------------
    echo   Running: !STEPS[%%i]!
    echo ------------------------------------------

    echo. >> "%LOG_FILE%"
    echo ------------------------------------------ >> "%LOG_FILE%"
    echo   Running: !STEPS[%%i]! >> "%LOG_FILE%"
    echo ------------------------------------------ >> "%LOG_FILE%"

    uv run "!STEPS[%%i]!" >> "%LOG_FILE%" 2>&1
    if !ERRORLEVEL! neq 0 (
        echo   FAILED: !STEPS[%%i]! ^(exit code !ERRORLEVEL!^)
        echo   FAILED: !STEPS[%%i]! ^(exit code !ERRORLEVEL!^) >> "%LOG_FILE%"
        echo   Check log: %LOG_FILE%
        echo ========================================== >> "%LOG_FILE%"
        echo   Pipeline FAILED at !STEPS[%%i]! >> "%LOG_FILE%"
        echo ========================================== >> "%LOG_FILE%"
        exit /b 1
    )
    echo   Done: !STEPS[%%i]!
    echo   Done: !STEPS[%%i]! >> "%LOG_FILE%"
)

echo.
echo ==========================================
echo   Pipeline complete!
echo   Log: %LOG_FILE%
echo ==========================================

echo. >> "%LOG_FILE%"
echo ========================================== >> "%LOG_FILE%"
echo   Pipeline complete! >> "%LOG_FILE%"
echo   Finished: %date% %time% >> "%LOG_FILE%"
echo ========================================== >> "%LOG_FILE%"

endlocal
