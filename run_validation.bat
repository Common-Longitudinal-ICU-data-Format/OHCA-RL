@echo off
REM OHCA-RL external validation pipeline (site use)
REM
REM Runs the site's data through cohort/state/reward construction,
REM then evaluates the coordinating center's pre-trained model.
REM
REM Steps: 01_cohort -> 02_wide -> 03_sofa -> 04_mdp -> 05_reward
REM        -> external_validation -> make_tableone
REM
REM Full stdout/stderr is captured to output\final\validation_log_<timestamp>.txt.
REM
REM Before running:
REM   1. Edit config\config.json with your site's settings
REM   2. Place shared\ddqn_cql_reviewed.pt and shared\feature_metadata.json
REM      (provided by the coordinating center) in this repo

setlocal enabledelayedexpansion

cd /d "%~dp0"

if "%PY%"=="" set PY=python

if not exist "shared\ddqn_cql_reviewed.pt" (
    echo ERROR: shared\ddqn_cql_reviewed.pt not found.
    echo        Obtain this file from the coordinating center.
    exit /b 1
)

if not exist "output\final" mkdir "output\final"

for /f "tokens=2 delims==" %%I in ('wmic OS Get localdatetime /value') do set DT=%%I
set TS=%DT:~0,8%_%DT:~8,6%
set LOG_FILE=output\final\validation_log_%TS%.txt

echo ================================================================ > "%LOG_FILE%"
echo OHCA-RL EXTERNAL VALIDATION PIPELINE -- %date% %time% >> "%LOG_FILE%"
echo Log: %LOG_FILE% >> "%LOG_FILE%"
echo ================================================================ >> "%LOG_FILE%"

echo ================================================================
echo OHCA-RL EXTERNAL VALIDATION PIPELINE
echo Log: %LOG_FILE%
echo ================================================================

for %%S in (01_cohort 02_wide 03_sofa 04_mdp 05_reward external_validation make_tableone) do (
    echo. >> "%LOG_FILE%"
    echo ---- Running code\%%S.py ---- >> "%LOG_FILE%"
    echo.
    echo ---- Running code\%%S.py ----
    %PY% code\%%S.py >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        echo Step %%S failed. See %LOG_FILE%. Aborting.
        exit /b 1
    )
)

echo. >> "%LOG_FILE%"
echo ================================================================ >> "%LOG_FILE%"
echo Validation pipeline complete -- %date% %time% >> "%LOG_FILE%"
echo   Site-specific shareable artifacts -^> output\final\^<site_id^>\ >> "%LOG_FILE%"
echo   Log                               -^> %LOG_FILE% >> "%LOG_FILE%"
echo   Upload that folder + this log to the coordinating center >> "%LOG_FILE%"
echo ================================================================ >> "%LOG_FILE%"

echo.
echo ================================================================
echo Validation pipeline complete.
echo   Site-specific shareable artifacts -^> output\final\^<site_id^>\
echo   Log                               -^> %LOG_FILE%
echo   Upload that folder + this log to the coordinating center
echo ================================================================
