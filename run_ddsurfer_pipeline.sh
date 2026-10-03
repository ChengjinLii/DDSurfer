#!/usr/bin/env bash

# DWI-to-native-surface pipeline. Cache cleanup is on; postprocess is opt-in.
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

ARGS=()
while (($#)); do
  case "$1" in
    -s) ARGS+=(--subject); shift ;;
    -h) ARGS+=(--help); shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
exec "${PYTHON_BIN:-python3}" "$PROJECT_ROOT/run_ddsurfer_pipeline.py" "${ARGS[@]}"
