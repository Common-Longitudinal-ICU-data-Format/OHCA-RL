@echo off
REM OHCA-RL External Validation Pipeline (Windows)
REM For participating sites validating the trained model on local data.
REM Skips step 06 (training) — uses pre-trained model from shared\.
REM
REM Usage: run_external_validation.bat
REM See SITE_INSTRUCTIONS.md for full setup guide.

setlocal enabledelayedexpansion

REM Change to the directory where this script lives
cd /d "%~dp0"

REM Create output directory
if not exist "output\final" mkdir "output\final"

REM Generate timestamped log filename
set "DATESTAMP=%date:~-4%%date:~-10,2%%date:~-7,2%"
set "HOUR=%time:~0,2%"
if "%HOUR:~0,1%"==" " set "HOUR=0%HOUR:~1,1%"
set "TIMESTAMP=%DATESTAMP%_%HOUR%%time:~3,2%%time:~6,2%"
set "LOG_FILE=output\final\pipeline_%TIMESTAMP%.log"

echo ==========================================
echo   OHCA-RL External Validation Pipeline
echo   Log: %LOG_FILE%
echo ==========================================

REM Verify shared\ artifacts exist
for %%f in (
    shared\best_model.pt
    shared\preprocessor.json
    shared\state_features.json
    shared\training_config.json
    shared\action_remap.json
) do (
    if not exist "%%f" (
        echo ERROR: Missing %%f
        echo Download standardization artifacts from Box first.
        echo See SITE_INSTRUCTIONS.md for details.
        exit /b 1
    )
)

echo   Shared artifacts verified.

REM Log header
echo ========================================== > "%LOG_FILE%"
echo   OHCA-RL External Validation Pipeline >> "%LOG_FILE%"
echo   Started: %date% %time% >> "%LOG_FILE%"
echo ========================================== >> "%LOG_FILE%"

REM Pipeline steps (skip step 06 — training)
set STEPS[0]=code\00_cohort_identification.py
set STEPS[1]=code\01_create_wide_df.py
set STEPS[2]=code\02_sofa_calculator.py
set STEPS[3]=code\03_ffill_and_bucketing.py
set STEPS[4]=code\04_create_tableone.py
set STEPS[5]=code\05_figures.py
set STEPS[6]=code\07_external_validation.py

for %%i in (0 1 2 3 4 5 6) do (
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
        exit /b 1
    )
    echo   Done: !STEPS[%%i]!
    echo   Done: !STEPS[%%i]! >> "%LOG_FILE%"
)

echo.
echo ==========================================
echo   External validation complete!
echo   Results: output\final\external_validation\
echo   Log: %LOG_FILE%
echo ==========================================

echo. >> "%LOG_FILE%"
echo ========================================== >> "%LOG_FILE%"
echo   External validation complete! >> "%LOG_FILE%"
echo   Finished: %date% %time% >> "%LOG_FILE%"
echo ========================================== >> "%LOG_FILE%"

endlocal
