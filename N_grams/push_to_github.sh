#!/bin/bash

# Script to push N_grams folder to GitHub repository
# Repository: https://github.com/ICS-NLP/Prosit1.git

echo "=========================================="
echo "Pushing N_grams folder to GitHub"
echo "Repository: https://github.com/ICS-NLP/Prosit1.git"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
N_GRAMS_DIR="$SCRIPT_DIR"

# Check if we're in a git repository
if [ -d ".git" ]; then
    echo "✓ Already in a git repository"
    REPO_DIR="$(pwd)"
else
    # Try to find or clone the repository
    echo "Not in a git repository. Checking for Prosit1 repository..."
    
    # Check if Prosit1 exists in parent directories
    if [ -d "../Prosit1/.git" ]; then
        REPO_DIR="$(cd ../Prosit1 && pwd)"
        echo "✓ Found Prosit1 repository at: $REPO_DIR"
    elif [ -d "../../Prosit1/.git" ]; then
        REPO_DIR="$(cd ../../Prosit1 && pwd)"
        echo "✓ Found Prosit1 repository at: $REPO_DIR"
    else
        echo "Prosit1 repository not found. Cloning..."
        cd "$(dirname "$N_GRAMS_DIR")"
        git clone https://github.com/ICS-NLP/Prosit1.git
        if [ $? -eq 0 ]; then
            REPO_DIR="$(cd Prosit1 && pwd)"
            echo "✓ Cloned repository to: $REPO_DIR"
        else
            echo "✗ Failed to clone repository"
            echo "Please clone manually: git clone https://github.com/ICS-NLP/Prosit1.git"
            exit 1
        fi
    fi
fi

echo ""
echo "Repository directory: $REPO_DIR"
echo "N_grams directory: $N_GRAMS_DIR"
echo ""

# Check if N_grams already exists in repo
if [ -d "$REPO_DIR/N_grams" ]; then
    echo "⚠ N_grams folder already exists in repository"
    read -p "Do you want to replace it? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    rm -rf "$REPO_DIR/N_grams"
fi

# Copy N_grams folder to repository
echo "Copying N_grams folder to repository..."
cp -r "$N_GRAMS_DIR" "$REPO_DIR/"
if [ $? -eq 0 ]; then
    echo "✓ Copied N_grams folder successfully"
else
    echo "✗ Failed to copy N_grams folder"
    exit 1
fi

# Navigate to repository
cd "$REPO_DIR"

# Check git status
echo ""
echo "Checking git status..."
git status

# Add N_grams folder
echo ""
echo "Adding N_grams folder to git..."
git add N_grams/

# Show what will be committed
echo ""
echo "Files to be committed:"
git status --short

# Commit
echo ""
read -p "Enter commit message (or press Enter for default): " COMMIT_MSG
if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="Add N-gram language model implementation for Akan language"
fi

git commit -m "$COMMIT_MSG"

if [ $? -eq 0 ]; then
    echo "✓ Committed successfully"
else
    echo "✗ Commit failed (maybe no changes to commit?)"
    exit 1
fi

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
echo "Repository: https://github.com/ICS-NLP/Prosit1.git"
echo "Branch: main (or master)"
echo ""

# Try main branch first, then master
git push origin main 2>/dev/null || git push origin master

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Successfully pushed to GitHub!"
    echo "=========================================="
    echo ""
    echo "View your repository at:"
    echo "https://github.com/ICS-NLP/Prosit1"
    echo ""
else
    echo ""
    echo "✗ Push failed. Possible reasons:"
    echo "  1. Authentication required (use SSH or personal access token)"
    echo "  2. No write access to repository"
    echo "  3. Network issues"
    echo ""
    echo "To push manually:"
    echo "  cd $REPO_DIR"
    echo "  git push origin main"
    echo ""
    exit 1
fi
