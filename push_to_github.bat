@echo off
setlocal enabledelayedexpansion

echo =======================================================================
echo  Healthcare IoT Big Data Analytics - GitHub Push & Streamlit Prep
echo  Supervisor: Ms. G. Hema Sri
echo =======================================================================
echo.

if "%1"=="" (
    echo [ERROR] Please provide your GitHub repository URL!
    echo Usage: push_to_github.bat https://github.com/YOUR_USERNAME/YOUR_REPO.git
    echo.
    echo Example:
    echo   push_to_github.bat https://github.com/vivek/healthcare-iot-clustering.git
    echo.
    pause
    exit /b 1
)

set REPO_URL=%1

echo [1/5] Initializing Git repository...
git init
if errorlevel 1 (
    echo Failed to initialize git. Make sure Git is installed.
    exit /b 1
)

echo.
echo [2/5] Adding project files...
git add .

echo.
echo [3/5] Creating initial commit...
git commit -m "Initial commit: Scalable Clustering of Heterogeneous Healthcare IoT Big Data via Distributed XGBoost (Ms. G. Hema Sri)"

echo.
echo [4/5] Setting main branch and remote origin...
git branch -M main
git remote remove origin 2>nul
git remote add origin %REPO_URL%

echo.
echo [5/5] Pushing to GitHub (%REPO_URL%)...
git push -u origin main

if errorlevel 1 (
    echo.
    echo [!] Push failed or requires authentication.
    echo Please make sure the GitHub repository exists and you have write permissions.
) else (
    echo.
    echo =======================================================================
    echo  SUCCESSFULLY PUSHED TO GITHUB!
    echo =======================================================================
    echo Next step: Deploy on Streamlit Cloud!
    echo 1. Go to https://share.streamlit.io/
    echo 2. Click 'New app'
    echo 3. Select repository: %REPO_URL%
    echo 4. Main file path: app.py
    echo 5. Click 'Deploy!'
    echo =======================================================================
)

pause
