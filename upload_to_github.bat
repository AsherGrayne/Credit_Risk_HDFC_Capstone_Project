@echo off
REM GitHub Upload Script for Credit Card Delinquency Pack
REM Run this script to upload your project to GitHub

echo ========================================
echo GitHub Upload Script
echo ========================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed!
    echo Please install Git from https://git-scm.com/
    pause
    exit /b 1
)

echo [1/4] Checking git status...
git status
echo.

echo [2/4] Adding all files...
git add .
echo.

echo [3/4] Committing changes...
set /p commit_msg="Enter commit message (or press Enter for default): "
if "%commit_msg%"=="" set commit_msg=Update project files
git commit -m "%commit_msg%"
echo.

echo [4/4] Pushing to GitHub...
git push origin main
echo.

echo ========================================
echo Upload Complete!
echo ========================================
echo.
echo Your repository: https://github.com/AsherGrayne/Credit_Risk_HDFC_Capstone_Project.git
echo.
pause

