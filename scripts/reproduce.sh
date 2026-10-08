#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/eduvaluebench-matplotlib"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/eduvaluebench-cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python_bin="${PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  printf 'Python interpreter not found: %s\n' "$python_bin" >&2
  exit 127
fi
"$python_bin" src/analyze_results.py \
  data/eduvaluebench_results.json.gz \
  --output-dir outputs \
  --quiet
"$python_bin" tests/verify_outputs.py outputs expected

printf '\nReproduction complete. Outputs are in %s/outputs\n' "$repo_root"
