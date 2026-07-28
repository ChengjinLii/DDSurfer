#!/usr/bin/env bash

###############################################################################
# DDSurfer end-to-end pipeline (shell wrapper)
#
# This script mirrors the behaviour of run_ddsurfer_pipeline.py and provides a
# purely shell-based interface for integrating into traditional workflows.
###############################################################################

set -euo pipefail
IFS=$'\n\t'

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DATA_TYPE="hcp"
DEVICE="cuda:0"
PREDICT_MODE="all"
INPUT_ROOT="$PROJECT_ROOT/DTI-inputs"
RAW_INPUT_ROOT="$PROJECT_ROOT/raw-dwi-inputs"
OUTPUT_ROOT="$PROJECT_ROOT/data_Reg/test"
PREDICTIONS_DIR="$PROJECT_ROOT/pred_results_DDSurfer"
SKIP_PREPROCESSING=0
SKIP_POSTPROCESSING=0

usage() {
  cat <<'USAGE'
Usage: run_ddsurfer_pipeline.sh --subject <ID> [options]

Required:
  -s, --subject <ID>         Subject identifier.

Options:
      --data-type <name>     Dataset key passed to prediction scripts (default: hcp).
      --device <target>      Torch device, e.g. cuda:0 or cpu (default: cuda:0).
      --raw-input-root <path> Root with raw diffusion inputs under <ID>/T1w/Diffusion.
      --input-root <path>    Directory containing registered DTI inputs (default: ./DTI-inputs).
      --output-root <path>   Directory for preprocessing outputs (default: ./data_Reg/test).
      --predictions-dir <path> Directory for predicted meshes (default: ./pred_results_DDSurfer).
      --predict-mode <mode>  Prediction mode for TANet: wm or all (default: all).
      --skip-preprocessing   Skip Data-Preprocessing.sh.
      --skip-postprocessing  Skip utils/space_MNI2orig.sh.
  -h, --help                 Show this message.
USAGE
}

SUBJECT_ID=""

while (($#)); do
  case "$1" in
    -s|--subject)
      SUBJECT_ID=$2; shift 2 ;;
    --data-type)
      DATA_TYPE=$2; shift 2 ;;
    --device)
      DEVICE=$2; shift 2 ;;
    --raw-input-root)
      RAW_INPUT_ROOT=$2; shift 2 ;;
    --input-root)
      INPUT_ROOT=$2; shift 2 ;;
    --output-root)
      OUTPUT_ROOT=$2; shift 2 ;;
    --predictions-dir)
      PREDICTIONS_DIR=$2; shift 2 ;;
    --predict-mode)
      PREDICT_MODE=$2; shift 2 ;;
    --skip-preprocessing)
      SKIP_PREPROCESSING=1; shift ;;
    --skip-postprocessing)
      SKIP_POSTPROCESSING=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      usage >&2
      echo "Unknown argument: $1" >&2
      exit 1 ;;
  esac
done

[[ -n "$SUBJECT_ID" ]] || { usage >&2; echo "Error: --subject is required." >&2; exit 1; }

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

run_cmd() {
  log "Executing: $*"
  "$@"
}

if [[ $SKIP_PREPROCESSING -eq 0 ]]; then
  run_cmd bash "$PROJECT_ROOT/Data-Preprocessing.sh" \
    --subject "$SUBJECT_ID" \
    --raw-input-root "$RAW_INPUT_ROOT" \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT"
else
  log "Skipping preprocessing as requested."
fi

mkdir -p "$PREDICTIONS_DIR"

run_cmd python3 "$PROJECT_ROOT/ddsurfer_predict_lh_dualstream.py" \
  --data_type "$DATA_TYPE" \
  --surf_hemi left \
  --device "$DEVICE" \
  --input_root "$OUTPUT_ROOT" \
  --output_dir "$PREDICTIONS_DIR" \
  --predict_mode "$PREDICT_MODE" \
  --subjects "$SUBJECT_ID"

run_cmd python3 "$PROJECT_ROOT/ddsurfer_predict_rh_dualstream.py" \
  --data_type "$DATA_TYPE" \
  --surf_hemi right \
  --device "$DEVICE" \
  --input_root "$OUTPUT_ROOT" \
  --output_dir "$PREDICTIONS_DIR" \
  --predict_mode "$PREDICT_MODE" \
  --subjects "$SUBJECT_ID"

if [[ $SKIP_POSTPROCESSING -eq 0 ]]; then
  run_cmd bash "$PROJECT_ROOT/utils/space_MNI2orig.sh" \
    --subject "$SUBJECT_ID" \
    --mode whole \
    --data-root "$OUTPUT_ROOT" \
    --pred-root "$PREDICTIONS_DIR"
else
  log "Skipping post-processing as requested."
fi

log "DDSurfer pipeline finished for subject ${SUBJECT_ID}."
