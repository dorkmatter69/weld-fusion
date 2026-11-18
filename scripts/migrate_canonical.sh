#!/usr/bin/env bash
# Run on the user's Mac after reviewing the staged integration.
# Snapshot the unversioned simulator before attaching the existing GitHub history.
set -euo pipefail

staged_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
canonical="${1:-$HOME/Developer/weld-twin}"
branch="feat/virtual-cell-mvp"

if [[ -e "$canonical/.git" ]]; then
  echo "Refusing: $canonical already has Git history; inspect it manually." >&2
  exit 1
fi
for item in README.md pyproject.toml run_demo.py weld_twin tests; do
  [[ -e "$canonical/$item" ]] || { echo "Missing original $canonical/$item" >&2; exit 1; }
done
[[ "$(git -C "$staged_repo" branch --show-current)" == "$branch" ]] || {
  echo "Staged integration is not on $branch" >&2; exit 1;
}
[[ -z "$(git -C "$staged_repo" status --porcelain)" ]] || {
  echo "Commit or inspect staged changes first." >&2; exit 1;
}

git -C "$canonical" init -b legacy/simulator
git -C "$canonical" add README.md pyproject.toml run_demo.py weld_twin tests .gitignore
git -C "$canonical" commit -m "Preserve original unversioned weld-twin simulator"
git -C "$canonical" remote add origin "$(git -C "$staged_repo" remote get-url origin)"
git -C "$canonical" fetch origin "$branch"
if [[ "$(git -C "$canonical" rev-parse "origin/$branch")" != "$(git -C "$staged_repo" rev-parse HEAD^)" ]]; then
  echo "Remote branch has moved; original snapshot is safe on legacy/simulator. Rebase the staged integration before switching." >&2
  exit 1
fi
git -C "$canonical" switch -c "$branch" --track "origin/$branch"
git -C "$staged_repo" format-patch -1 HEAD --stdout | git -C "$canonical" am

echo "Canonical migration complete: $canonical"
git -C "$canonical" status --short --branch
git -C "$canonical" remote -v
echo "Original source remains in branch legacy/simulator; integrated source is under src/weld_twin."
