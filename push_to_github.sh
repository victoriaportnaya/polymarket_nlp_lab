#!/bin/bash
# Creates the repo on GitHub (via browser) and pushes this project.
# Usage: ./push_to_github.sh [GITHUB_USERNAME]
# Repo name: polymarket-sentiment (change REPO_NAME below if you prefer)

set -e
REPO_NAME="polymarket-sentiment"
DESCRIPTION="Polymarket sentiment-to-prediction pipeline (Teichmann-style randomized signatures)"

USER="${1:-}"
if [ -z "$USER" ]; then
  echo "Usage: ./push_to_github.sh YOUR_GITHUB_USERNAME"
  echo "Example: ./push_to_github.sh johndoe"
  exit 1
fi

URL="https://github.com/new"
echo "Opening GitHub. Create a new repo with name: $REPO_NAME"
echo "Leave it empty (no README, no .gitignore). Then come back here."
echo "URL: $URL"
open "$URL" 2>/dev/null || xdg-open "$URL" 2>/dev/null || true

read -p "After you created the repo, press Enter to push..."

cd "$(dirname "$0")"
if git remote get-url origin 2>/dev/null; then
  git remote remove origin
fi
git remote add origin "https://github.com/${USER}/${REPO_NAME}.git"
git branch -M main
git push -u origin main
echo "Done. Repo: https://github.com/${USER}/${REPO_NAME}"
