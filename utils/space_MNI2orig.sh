#!/usr/bin/env bash

###############################################################################
# DDSurfer Surface Post-processing (MNI → native space)
#
# This script converts predicted cortical surfaces (OBJ) to STL, applies the
# inverse deformation field stored in an ITK .tfm file, and writes the meshes
# back in subject-native space. The sequence mirrors the original workflow but
# exposes a clean CLI suitable for automated pipelines.
###############################################################################

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-python3}
OBJ2STL_PY=${OBJ2STL_PY:-"$PROJECT_ROOT/utils/obj2stl.py"}
TRANSFORM_PY=${TRANSFORM_PY:-"$PROJECT_ROOT/utils/space_MNI2orig.py"}

DEFAULT_SUBJECT="917255-retest-ANTs"
DEFAULT_MODE="whole" # Options: left, right, whole
DEFAULT_DATA_ROOT=${DEFAULT_DATA_ROOT:-"$PROJECT_ROOT/data_Reg/hcp/test"}
DEFAULT_PRED_ROOT=${DEFAULT_PRED_ROOT:-"$PROJECT_ROOT/pred_results_DDSurfer"}

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

usage() {
  cat <<'USAGE'
Usage: utils/space_MNI2orig.sh [options]

Options:
  -s, --subject <ID>        Subject identifier. Defaults to 917255-retest-ANTs.
  -m, --mode <MODE>         Hemisphere to process: left, right, or whole (default).
      --data-root <path>    Directory with preprocessed data (contains <ID>/ID-b0ToAtlasT2.tfm).
      --pred-root <path>    Directory containing predicted OBJ files.
      --python-bin <bin>    Python interpreter to invoke (default: python3).
  -h, --help                Show this message and exit.

Example:
  utils/space_MNI2orig.sh --subject 100307 --mode whole
USAGE
}

subject="$DEFAULT_SUBJECT"
mode="$DEFAULT_MODE"
data_root="$DEFAULT_DATA_ROOT"
pred_root="$DEFAULT_PRED_ROOT"

parse_args() {
  while (($#)); do
    case "$1" in
      -s|--subject)
        subject=$2; shift 2 ;;
      -m|--mode)
        mode=$2; shift 2 ;;
      --data-root)
        data_root=$2; shift 2 ;;
      --pred-root)
        pred_root=$2; shift 2 ;;
      --python-bin)
        PYTHON_BIN=$2; shift 2 ;;
      -h|--help)
        usage; exit 0 ;;
      *)
        usage >&2
        log "ERROR: unrecognised argument: $1"
        exit 1 ;;
    esac
  done
}

parse_args "$@"

case "$mode" in
  left|right|whole) ;;
  *)
    log "ERROR: mode must be left, right, or whole (received: $mode)"
    exit 1 ;;
esac

tfm_file="$data_root/$subject/${subject}-b0ToAtlasT2.tfm"
[[ -f "$tfm_file" ]] || { log "ERROR: transform file not found: $tfm_file"; exit 1; }

[[ -d "$pred_root" ]] || { log "ERROR: prediction directory not found: $pred_root"; exit 1; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { log "ERROR: python interpreter not found: $PYTHON_BIN"; exit 1; }

mni_dir="$pred_root/mni/$subject"
native_dir="$pred_root/native/$subject"

[[ -d "$mni_dir" ]] || { log "ERROR: MNI-space meshes not found: $mni_dir"; exit 1; }
mkdir -p "$native_dir"

convert_surface() {
  local hemi_long=$1   # left/right
  local hemi_short=$2  # lh/rh
  local surface=$3     # wm/pial

  local obj_in="$mni_dir/${subject}_predicted_${surface}_surface_${hemi_long}.obj"
  local stl_tmp="$native_dir/${subject}_predicted_${surface}_surface_${hemi_long}_mni.stl"
  local stl_out="$native_dir/${subject}_predicted_${surface}_${hemi_short}.stl"

  if [[ ! -f "$obj_in" ]]; then
    log "WARNING: OBJ not found for ${surface} (${hemi_long}): $obj_in"
    return
  fi

  log "Converting ${obj_in##*/} -> ${stl_tmp##*/}"
  "$PYTHON_BIN" "$OBJ2STL_PY" --input_obj "$obj_in" --output_stl "$stl_tmp"

  log "Applying inverse transform -> ${stl_out##*/}"
  "$PYTHON_BIN" "$TRANSFORM_PY" \
    --tfm_inverse_transform "$tfm_file" \
    --input_stl "$stl_tmp" \
    --output_stl "$stl_out"

  rm -f "$stl_tmp"
}

run_for_hemisphere() {
  local hemi=$1
  local hemi_short
  case "$hemi" in
    left) hemi_short="lh" ;;
    right) hemi_short="rh" ;;
    *) log "ERROR: invalid hemisphere '$hemi'"; exit 1 ;;
  esac

  log "Processing ${hemi} hemisphere"
  convert_surface "$hemi" "$hemi_short" "wm"
  convert_surface "$hemi" "$hemi_short" "pial"
}

log "Subject: $subject"
log "Mode: $mode"
log "Predictions (MNI): $mni_dir"
log "Predictions (native): $native_dir"
log "Transforms: $tfm_file"

case "$mode" in
  whole)
    run_for_hemisphere left
    run_for_hemisphere right
    ;;
  left|right)
    run_for_hemisphere "$mode"
    ;;
esac

log "Surface post-processing finished."
