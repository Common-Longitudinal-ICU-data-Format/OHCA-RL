@echo off
REM OHCA-RL training pipeline (coordinating-center / training-site use)
REM
REM Runs: 01_cohort -> 02_wide -> 03_sofa -> 04_mdp -> 05_reward
REM       -> 06_model_cql_fqe -> make_tableone
REM
REM Trained model + feature metadata land in shared/ for distribution.
REM Full stdout/stderr is captured to output\final\training_log_<timestamp>.txt.

setlocal enabledelayedexpansion

cd /d "%~dp0"

if "%PY%"=="" set PY=python

if not exist "output\final" mkdir "output\final"

for /f "tokens=2 delims==" %%I in ('wmic OS Get localdatetime /value') do set DT=%%I
set TS=%DT:~0,8%_%DT:~8,6%
set LOG_FILE=output\final\training_log_%TS%.txt

echo ================================================================ > "%LOG_FILE%"
echo OHCA-RL TRAINING PIPELINE -- %date% %time% >> "%LOG_FILE%"
echo Log: %LOG_FILE% >> "%LOG_FILE%"
echo ================================================================ >> "%LOG_FILE%"

echo ================================================================
echo OHCA-RL TRAINING PIPELINE
echo Log: %LOG_FILE%
echo ================================================================

for %%S in (01_cohort 02_wide 03_sofa 04_mdp 05_reward 06_model_cql_fqe make_tableone) do (
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
echo Training pipeline complete -- %date% %time% >> "%LOG_FILE%"
echo   Shareable artifacts -^> output\final\ >> "%LOG_FILE%"
echo   Shareable model     -^> shared\ddqn_cql_reviewed.pt >> "%LOG_FILE%"
echo   Log                 -^> %LOG_FILE% >> "%LOG_FILE%"
echo ================================================================ >> "%LOG_FILE%"

echo.
echo ================================================================
echo Training pipeline complete.
echo   Shareable artifacts -^> output\final\
echo   Shareable model     -^> shared\ddqn_cql_reviewed.pt
echo   Log                 -^> %LOG_FILE%
echo ================================================================
