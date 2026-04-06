#!/usr/bin/env bash
set -euo pipefail

TOP_N="${1:-50}"
ROOT="${2:-.}"

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is not installed."
  exit 1
fi

if ! git -C "$ROOT" rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "Error: '$ROOT' is not inside a git repository."
  exit 1
fi

REPO_ROOT="$(git -C "$ROOT" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "Repository: $REPO_ROOT"
echo "Showing top $TOP_N largest files"
echo

tmp_all="$(mktemp)"
tmp_git="$(mktemp)"
trap 'rm -f "$tmp_all" "$tmp_git"' EXIT

# All files on disk, excluding .git
find . -path './.git' -prune -o -type f -printf '%s\t%p\n' \
  | sort -nr > "$tmp_all"

# Git-tracked files
git ls-files > "$tmp_git"

printf "%-10s %-8s %s\n" "SIZE" "TRACKED" "FILE"
printf "%-10s %-8s %s\n" "--------" "--------" "----"

count=0
while IFS=$'\t' read -r size path; do
  human_size="$(numfmt --to=iec-i --suffix=B "$size" 2>/dev/null || echo "${size}B")"
  clean_path="${path#./}"

  if grep -Fxq "$clean_path" "$tmp_git"; then
    tracked="yes"
  else
    tracked="no"
  fi

  printf "%-10s %-8s %s\n" "$human_size" "$tracked" "$clean_path"

  count=$((count + 1))
  if [ "$count" -ge "$TOP_N" ]; then
    break
  fi
done < "$tmp_all"
