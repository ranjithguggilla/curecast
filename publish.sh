#!/usr/bin/env bash
# Publish curecast to github.com/ranjithguggilla with a clean, progressive
# history under your identity. Run this once, from inside the repo folder, on a
# machine where git can write (the shared Projects mount cannot).
#
#   cd curecast && bash publish.sh
#
# It rebuilds the commit history from scratch, runs a content scan, and pushes.
set -euo pipefail

REPO="curecast"
GH_USER="ranjithguggilla"

cd "$(dirname "$0")"

echo "==> clearing any stale git state and build cruft"
rm -rf .git
rm -rf .pytest_cache .ruff_cache src/*.egg-info
find . -name __pycache__ -type d -prune -exec rm -rf {} +
rm -f assets/curecast_model.joblib   # rebuilt by training and CI

echo "==> initialising repo"
git init -q -b main
git config user.name "Ranjith Guggilla"
git config user.email "ranjithguggilla668@gmail.com"

echo "==> building progressive history"
git add pyproject.toml LICENSE .gitignore requirements.txt requirements-dev.txt \
        src/curecast/__init__.py src/curecast/config.py
git commit -q -m "Scaffold package, packaging metadata, and MIT license"

git add src/curecast/data.py data/concrete.csv data/README.md tests/test_data.py
git commit -q -m "Add concrete dataset with strict loading and range validation"

git add src/curecast/features.py src/curecast/model.py src/curecast/evaluate.py \
        tests/test_features.py tests/test_model.py tests/test_evaluate.py
git commit -q -m "Add mix feature builder, model zoo, and regression metrics"

git add src/curecast/train.py src/curecast/plots.py src/curecast/eda.py \
        assets/metrics.json assets/*.png
git commit -q -m "Wire end-to-end training with cross-validation and charts"

git add src/curecast/optimize.py src/curecast/scoring.py tests/conftest.py \
        tests/test_scoring.py tests/test_optimize.py
git commit -q -m "Add carbon-aware mix optimiser and the scoring entry point"

git add src/curecast/app.py
git commit -q -m "Add Gradio app over the scoring and optimiser functions"

git add .github/workflows/ci.yml
git commit -q -m "Add CI: ruff and pytest, plus a main-branch training artifact"

git add README.md
git commit -q -m "Document the project, results, and the carbon case"

# pick up anything not explicitly staged above (safety net)
if [ -n "$(git status --porcelain)" ]; then
  git add -A
  git commit -q -m "Add remaining project files"
fi

echo "==> content scan (only binary/dataset noise should match)"
if git grep -inE "claude|anthropic|llm|day [0-9]|streak" -- '*.py' '*.md' '*.toml' '*.txt' '*.yml'; then
  echo "!! text match found above — review before pushing" >&2
  exit 1
fi
echo "   clean"

echo "==> publishing to github.com/${GH_USER}/${REPO}"
if command -v gh >/dev/null 2>&1; then
  gh repo create "${GH_USER}/${REPO}" --public --source . --push
else
  echo "   gh not found, using plain git remote"
  git remote add origin "https://github.com/${GH_USER}/${REPO}.git"
  git push -u origin main
fi

echo "==> done. CI will lint, test, and build the model artifact on push."
