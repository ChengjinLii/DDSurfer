#!/usr/bin/env bash

###############################################################################
# DDSurfer Data Preprocessing Pipeline
#
# The preprocessing workflow has four stages:
#   1. DTI estimation and atlas registration from raw diffusion inputs
#   2. Skull stripping of registered scalar maps
#   3. Resampling into the fixed DDSurfer template geometry
#   4. Per-volume z-score intensity normalisation
###############################################################################

set -euo pipefail
IFS=$'\n\t'

###############################################################################
# Configuration defaults
###############################################################################

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-python3}

DTI_PROCESSING_SCRIPT=${DTI_PROCESSING_SCRIPT:-"$SCRIPT_DIR/dti.sh"}
PYTHON_SKULL_STRIPPING_SCRIPT=${PYTHON_SKULL_STRIPPING_SCRIPT:-"$SCRIPT_DIR/mask.py"}
PYTHON_RESAMPLE_SCRIPT=${PYTHON_RESAMPLE_SCRIPT:-"$SCRIPT_DIR/resample.py"}
PYTHON_ZSCORE_SCRIPT=${PYTHON_ZSCORE_SCRIPT:-"$SCRIPT_DIR/normalize.py"}

DWI_RAW_INPUT_ROOT=${DWI_RAW_INPUT_ROOT:-"$PROJECT_ROOT/inputs"}
DDSURFER_TESTDATA_DIR=${DDSURFER_TESTDATA_DIR:-"$PROJECT_ROOT/outputs/.cache/dti"}

NEW_OUTPUT_BASE_DIR=${NEW_OUTPUT_BASE_DIR:-"$PROJECT_ROOT/outputs/.cache/volumes"}
LOG_DIR_BASE=${LOG_DIR_BASE:-"$PROJECT_ROOT/logs/preprocessing"}
SKIP_DTI_PROCESSING=${SKIP_DTI_PROCESSING:-0}
PREPROCESS_JOBS=${PREPROCESS_JOBS:-1}
MINIMAL=0

DTI_SLICER_PATH=${DTI_SLICER_PATH:-}
DTI_REFERENCE_IMAGE=${DTI_REFERENCE_IMAGE:-}
DTI_MASK_FLIP=${DTI_MASK_FLIP:-}
DWI_INPUT=""
BVAL_INPUT=""
BVEC_INPUT=""
MASK_INPUT=""

# DDSurfer template geometry hard-coded from the fixed preprocessing target
# space used by the original release.
RESAMPLE_TARGET_SIZE=(176 224 176)
RESAMPLE_TARGET_SPACING=(1.0 1.0 1.0)
RESAMPLE_TARGET_ORIGIN=(88.29999542236328 129.0 -69.0)
RESAMPLE_TARGET_DIRECTION=(-1.0 0.0 0.0 0.0 -1.0 0.0 0.0 0.0 1.0)

# Comma separated list of subject identifiers used when no CLI override is
# provided. Keeping the previous sample subject ensures backward compatibility.
DEFAULT_SUBJECTS=${DEFAULT_SUBJECTS:-"100307"}

###############################################################################
# Helper utilities
###############################################################################

usage() {
  cat <<'USAGE'
Usage: preprocessing/run.sh [options]

Subject selection:
  -s, --subject <ID>         Process a single subject (can be repeated)
      --subjects <ID,...>    Comma-separated list of subject identifiers
      --subjects-file <path> File containing one subject identifier per line

Raw inputs:
      --dwi <path>          Native 4D diffusion NIfTI
      --bval <path>         Diffusion b-values
      --bvec <path>         Diffusion directions (3xN or Nx3)
      --mask <path>         Brain mask on the native DWI grid
      --raw-input-root <path> DDParcel file layout or HCP subject tree

Directory overrides:
      --dti-output-root <path> Directory for automatically generated DTI maps
      --output-root <path>   Directory where resampled outputs are written
      --log-dir <path>       Directory for preprocessing logs
      --skip-dti-processing  Skip DTI estimation and require existing inputs
      --minimal              Only prepare the five inference scalar channels
      --jobs <N>             Concurrent independent DTI stages (default: 1)

Misc:
  -h, --help                 Show this message and exit

The atlas-template geometry is built into the script; no separate reference FA
template file is required at runtime.
Supply all four raw input files together for a single subject, or use directory
discovery. DTI maps are intermediate results, not required source inputs.
USAGE
}

declare -a SUBJECTS=()

log_file=""

log() {
  local message=$1
  if [[ -n "$log_file" ]]; then
    printf '%s\n' "$message" | tee -a "$log_file"
  else
    printf '%s\n' "$message"
  fi
}

die() {
  local message=$1
  if [[ -n "$log_file" ]]; then
    printf 'ERROR: %s\n' "$message" | tee -a "$log_file" >&2
  else
    printf 'ERROR: %s\n' "$message" >&2
  fi
  exit 1
}

require_file() {
  local path=$1
  local description=${2:-"Required file"}
  [[ -f "$path" ]] || die "$description not found at $path"
}

parse_args() {
  while (($#)); do
    case "$1" in
      -s|--subject)
        [[ $# -ge 2 ]] || die "Option $1 requires an argument"
        SUBJECTS+=("$2")
        shift 2
        ;;
      --subjects)
        [[ $# -ge 2 ]] || die "Option $1 requires an argument"
        IFS=',' read -r -a _subjects_from_cli <<<"$2"
        SUBJECTS+=("${_subjects_from_cli[@]}")
        shift 2
        ;;
      --subjects-file)
        [[ $# -ge 2 ]] || die "Option $1 requires an argument"
        [[ -f "$2" ]] || die "Subjects file not found: $2"
        while IFS= read -r line || [[ -n "$line" ]]; do
          [[ -n "${line// }" ]] && SUBJECTS+=("$line")
        done <"$2"
        shift 2
        ;;
      --raw-input-root)
        [[ $# -ge 2 ]] || die "Option $1 requires an argument"
        DWI_RAW_INPUT_ROOT="$2"
        shift 2
        ;;
      --dti-output-root|--input-root)
        [[ $# -ge 2 ]] || die "Option $1 requires an argument"
        DDSURFER_TESTDATA_DIR="$2"
        shift 2
        ;;
      --dwi|--bval|--bvec|--mask)
        [[ $# -ge 2 ]] || die "Option $1 requires an argument"
        case "$1" in
          --dwi) DWI_INPUT=$2 ;;
          --bval) BVAL_INPUT=$2 ;;
          --bvec) BVEC_INPUT=$2 ;;
          --mask) MASK_INPUT=$2 ;;
        esac
        shift 2
        ;;
      --output-root)
        [[ $# -ge 2 ]] || die "Option $1 requires an argument"
        NEW_OUTPUT_BASE_DIR="$2"
        shift 2
        ;;
      --log-dir)
        [[ $# -ge 2 ]] || die "Option $1 requires an argument"
        LOG_DIR_BASE="$2"
        shift 2
        ;;
      --minimal)
        MINIMAL=1
        shift
        ;;
      --jobs)
        [[ $# -ge 2 ]] || die "Option $1 requires an argument"
        PREPROCESS_JOBS=$2
        shift 2
        ;;
      --skip-dti-processing)
        SKIP_DTI_PROCESSING=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        usage >&2
        die "Unrecognised argument: $1"
        ;;
    esac
  done
}

ensure_subject_list() {
  if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
    IFS=',' read -r -a _default_subjects <<<"$DEFAULT_SUBJECTS"
    SUBJECTS=("${_default_subjects[@]}")
    log "INFO: No subjects provided via CLI. Falling back to DEFAULT_SUBJECTS=${DEFAULT_SUBJECTS}."
  fi
}

initialise_logging() {
  mkdir -p "$LOG_DIR_BASE"
  log_file="${LOG_DIR_BASE}/preprocessing_$(date +%Y%m%d-%H%M%S).log"
  : >"$log_file"
  log "==================================================================="
  log "DDSurfer preprocessing run started: $(date)"
  log "Raw DWI root:  $DWI_RAW_INPUT_ROOT"
  log "Input root:    $DDSURFER_TESTDATA_DIR"
  log "Output root:   $NEW_OUTPUT_BASE_DIR"
  log "Template size: ${RESAMPLE_TARGET_SIZE[*]}"
  log "Template spacing: ${RESAMPLE_TARGET_SPACING[*]}"
  log "Template origin: ${RESAMPLE_TARGET_ORIGIN[*]}"
  log "Log file:      $log_file"
  log "==================================================================="
}

check_tooling() {
  require_file "$DTI_PROCESSING_SCRIPT" "DTI processing script"
  require_file "$PYTHON_SKULL_STRIPPING_SCRIPT" "Skull stripping utility"
  require_file "$PYTHON_RESAMPLE_SCRIPT" "Resampling utility"
  require_file "$PYTHON_ZSCORE_SCRIPT" "Z-score utility"

  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python interpreter not found: $PYTHON_BIN"
}

subject_has_registered_dti_inputs() {
  local subject_id=$1
  local subject_input_dir=$2
  local -a expected_inputs=(
    "${subject_input_dir}/${subject_id}-dti-FractionalAnisotropy-Reg.nii.gz"
    "${subject_input_dir}/${subject_id}-dti-MinEigenvalue-Reg.nii.gz"
    "${subject_input_dir}/${subject_id}-dti-MidEigenvalue-Reg.nii.gz"
    "${subject_input_dir}/${subject_id}-dti-MaxEigenvalue-Reg.nii.gz"
    "${subject_input_dir}/${subject_id}-dti-MeanDiffusivity-Reg.nii.gz"
    "${subject_input_dir}/${subject_id}-mask-Reg.nii.gz"
    "${subject_input_dir}/${subject_id}-b0ToAtlasT2.tfm"
  )

  [[ "$MINIMAL" -eq 1 ]] || expected_inputs+=("${subject_input_dir}/${subject_id}-dti-Trace-Reg.nii.gz")
  local path
  for path in "${expected_inputs[@]}"; do
    file_is_usable "$path" || return 1
  done
  return 0
}

file_is_usable() {
  local path=$1
  [[ -s "$path" ]] || return 1
  if [[ "$path" == *.nii.gz ]]; then
    gzip -t "$path" >/dev/null 2>&1 || return 1
  fi
  return 0
}

run_dti_processing() {
  local subject_id=$1
  local subject_input_dir="${DDSURFER_TESTDATA_DIR}/${subject_id}"
  local -a command=(
    bash
    "$DTI_PROCESSING_SCRIPT"
    --subject "$subject_id"
    --input-root "$DWI_RAW_INPUT_ROOT"
    --output-root "$DDSURFER_TESTDATA_DIR"
    --python-bin "$PYTHON_BIN"
    --jobs "$PREPROCESS_JOBS"
  )
  [[ "$MINIMAL" -eq 0 ]] || command+=(--minimal)

  if [[ -n "$DWI_INPUT$BVAL_INPUT$BVEC_INPUT$MASK_INPUT" ]]; then
    command+=(--dwi "$DWI_INPUT" --bval "$BVAL_INPUT" --bvec "$BVEC_INPUT" --mask "$MASK_INPUT")
  fi

  if [[ -n "$DTI_SLICER_PATH" ]]; then
    command+=(--slicer-path "$DTI_SLICER_PATH")
  fi
  if [[ -n "$DTI_REFERENCE_IMAGE" ]]; then
    command+=(--reference-image "$DTI_REFERENCE_IMAGE")
  fi
  if [[ -n "$DTI_MASK_FLIP" ]]; then
    command+=(--mask-flip "$DTI_MASK_FLIP")
  fi

  log "Step 0 | DTI estimation and atlas registration"
  if [[ "$SKIP_DTI_PROCESSING" -eq 1 ]]; then
    if subject_has_registered_dti_inputs "$subject_id" "$subject_input_dir"; then
      log "  [skip] DTI processing explicitly disabled."
      return 0
    fi
    log "  [warn] Registered DTI maps are missing and DTI processing was disabled."
    return 1
  fi

  log "  [run] Computing DTI scalar maps from raw diffusion inputs"
  "${command[@]}" >>"$log_file" 2>&1 || return 1

  if ! subject_has_registered_dti_inputs "$subject_id" "$subject_input_dir"; then
    log "  [warn] DTI processing finished but required registered inputs are still missing."
    return 1
  fi

  "$PYTHON_BIN" "$PROJECT_ROOT/preprocessing/inputs.py" \
    --reuse-record "$subject_input_dir/raw_inputs.json" \
    --record "$NEW_OUTPUT_BASE_DIR/$subject_id/raw_inputs.json" >>"$log_file" 2>&1 || return 1

  return 0
}

process_subject() {
  local subject_id=$1
  log ""
  log "-------------------------------------------------------------------"
  log "Subject: $subject_id"
  log "-------------------------------------------------------------------"

  local subject_input_dir="${DDSURFER_TESTDATA_DIR}/${subject_id}"
  local subject_output_dir="${NEW_OUTPUT_BASE_DIR}/${subject_id}"
  local subject_mask="${subject_input_dir}/${subject_id}-mask-Reg.nii.gz"

  mkdir -p "$subject_input_dir"
  if ! run_dti_processing "$subject_id"; then
    log "  Skipping subject ${subject_id} because DTI inputs are unavailable."
    return 1
  fi

  log "Step 1 | Masking, fixed-grid resampling and z-score normalisation"
  local -a volume_command=("$PYTHON_BIN" "$SCRIPT_DIR/volumes.py" --subject "$subject_id"
    --input-dir "$subject_input_dir" --output-dir "$subject_output_dir"
    --mask-script "$PYTHON_SKULL_STRIPPING_SCRIPT" --resample-script "$PYTHON_RESAMPLE_SCRIPT"
    --normalize-script "$PYTHON_ZSCORE_SCRIPT")
  [[ "$MINIMAL" -eq 0 ]] || volume_command+=(--minimal)
  "${volume_command[@]}" >>"$log_file" 2>&1

  log "Completed subject ${subject_id}"
}

###############################################################################
# Script entry point
###############################################################################

parse_args "$@"
[[ "$PREPROCESS_JOBS" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
initialise_logging
check_tooling
ensure_subject_list
if [[ -n "$DWI_INPUT$BVAL_INPUT$BVEC_INPUT$MASK_INPUT" ]]; then
  [[ -n "$DWI_INPUT" && -n "$BVAL_INPUT" && -n "$BVEC_INPUT" && -n "$MASK_INPUT" ]] \
    || die "Provide --dwi, --bval, --bvec and --mask together"
  [[ ${#SUBJECTS[@]} -eq 1 ]] || die "Explicit input files can only be used for one subject"
fi

log "Subjects to process: ${SUBJECTS[*]}"
for subject_id in "${SUBJECTS[@]}"; do
  process_subject "$subject_id"
done

log ""
log "==================================================================="
log "All preprocessing tasks completed: $(date)"
log "==================================================================="
