#!/bin/bash
# GitHub Upload Script for Credit Card Delinquency Pack
# Run this script to upload your project to GitHub

echo "========================================"
echo "GitHub Upload Script"
echo "========================================"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "ERROR: Git is not installed!"
    echo "Please install Git from https://git-scm.com/"
    exit 1
fi

echo "[1/4] Checking git status..."
git status
echo ""

echo "[2/4] Adding all files..."
git add .
echo ""

echo "[3/4] Committing changes..."
read -p "Enter commit message (or press Enter for default): " commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Update project files"
fi
git commit -m "$commit_msg"
echo ""

echo "[4/4] Pushing to GitHub..."
git push origin main
echo ""

echo "========================================"
echo "Upload Complete!"
echo "========================================"
echo ""
echo "Your repository: https://github.com/AsherGrayne/Credit_Risk_HDFC_Capstone_Project.git"
echo ""

