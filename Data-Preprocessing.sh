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
PROJECT_ROOT="$SCRIPT_DIR"

PYTHON_BIN=${PYTHON_BIN:-python3}

DTI_PROCESSING_SCRIPT=${DTI_PROCESSING_SCRIPT:-"$PROJECT_ROOT/dti_processing/run_dti_processing.sh"}
PYTHON_SKULL_STRIPPING_SCRIPT=${PYTHON_SKULL_STRIPPING_SCRIPT:-"$PROJECT_ROOT/utils/skull_stripping.py"}
PYTHON_RESAMPLE_SCRIPT=${PYTHON_RESAMPLE_SCRIPT:-"$PROJECT_ROOT/utils/nifti_resample.py"}
PYTHON_ZSCORE_SCRIPT=${PYTHON_ZSCORE_SCRIPT:-"$PROJECT_ROOT/utils/nifti_zscore.py"}

DWI_RAW_INPUT_ROOT=${DWI_RAW_INPUT_ROOT:-"$PROJECT_ROOT/raw-dwi-inputs"}
DDSURFER_TESTDATA_DIR=${DDSURFER_TESTDATA_DIR:-"$PROJECT_ROOT/DTI-inputs"}

NEW_OUTPUT_BASE_DIR=${NEW_OUTPUT_BASE_DIR:-"$PROJECT_ROOT/data_Reg/test"}
LOG_DIR_BASE=${LOG_DIR_BASE:-"$PROJECT_ROOT/logs/preprocessing"}
SKIP_DTI_PROCESSING=${SKIP_DTI_PROCESSING:-0}

DTI_SLICER_PATH=${DTI_SLICER_PATH:-}
DTI_REFERENCE_IMAGE=${DTI_REFERENCE_IMAGE:-}
DTI_MASK_FLIP=${DTI_MASK_FLIP:-}

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
Usage: Data-Preprocessing.sh [options]

Subject selection:
  -s, --subject <ID>         Process a single subject (can be repeated)
      --subjects <ID,...>    Comma-separated list of subject identifiers
      --subjects-file <path> File containing one subject identifier per line

Directory overrides:
      --raw-input-root <path> Raw diffusion root with <ID>/T1w/Diffusion
      --input-root <path>    Directory containing subject-specific registered DTI inputs
      --output-root <path>   Directory where resampled outputs are written
      --log-dir <path>       Directory for preprocessing logs
      --skip-dti-processing  Skip DTI estimation and require existing inputs

Misc:
  -h, --help                 Show this message and exit

The atlas-template geometry is built into the script; no separate reference FA
template file is required at runtime.
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
      --input-root)
        [[ $# -ge 2 ]] || die "Option $1 requires an argument"
        DDSURFER_TESTDATA_DIR="$2"
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

run_resample() {
  local source_path=$1
  local output_path=$2
  "$PYTHON_BIN" "$PYTHON_RESAMPLE_SCRIPT" \
    --source_image_path "$source_path" \
    --target_size "${RESAMPLE_TARGET_SIZE[@]}" \
    --target_spacing "${RESAMPLE_TARGET_SPACING[@]}" \
    --target_origin "${RESAMPLE_TARGET_ORIGIN[@]}" \
    --target_direction "${RESAMPLE_TARGET_DIRECTION[@]}" \
    --output_file_path "$output_path" \
    >>"$log_file" 2>&1
}

###############################################################################
# Core processing routines
###############################################################################

declare -a _DTI_PARAMS_TO_MASK=(
  "FractionalAnisotropy"
  "MinEigenvalue"
  "MidEigenvalue"
  "MaxEigenvalue"
  "Trace"
  "MeanDiffusivity"
)

declare -a _FILES_TO_RESAMPLE=(
  "testdata:dti-MinEigenvalue-Reg-masked:MinEigenvalue"
  "testdata:dti-MidEigenvalue-Reg-masked:MidEigenvalue"
  "testdata:dti-Trace-Reg-masked:Trace"
  "testdata:dti-FractionalAnisotropy-Reg-masked:FA"
  "testdata:dti-MaxEigenvalue-Reg-masked:MaxEigenvalue"
  "testdata:dti-MeanDiffusivity-Reg-masked:MD"
)

subject_has_registered_dti_inputs() {
  local subject_id=$1
  local subject_input_dir=$2
  local -a expected_inputs=(
    "${subject_input_dir}/${subject_id}-dti-FractionalAnisotropy-Reg.nii.gz"
    "${subject_input_dir}/${subject_id}-dti-MinEigenvalue-Reg.nii.gz"
    "${subject_input_dir}/${subject_id}-dti-MidEigenvalue-Reg.nii.gz"
    "${subject_input_dir}/${subject_id}-dti-MaxEigenvalue-Reg.nii.gz"
    "${subject_input_dir}/${subject_id}-dti-Trace-Reg.nii.gz"
    "${subject_input_dir}/${subject_id}-dti-MeanDiffusivity-Reg.nii.gz"
    "${subject_input_dir}/${subject_id}-mask-Reg.nii.gz"
    "${subject_input_dir}/${subject_id}-b0ToAtlasT2.tfm"
  )

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
  local raw_subject_dir="${DWI_RAW_INPUT_ROOT}/${subject_id}/T1w/Diffusion"
  local -a command=(
    bash
    "$DTI_PROCESSING_SCRIPT"
    --subject "$subject_id"
    --input-root "$DWI_RAW_INPUT_ROOT"
    --output-root "$DDSURFER_TESTDATA_DIR"
    --python-bin "$PYTHON_BIN"
  )

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
  if subject_has_registered_dti_inputs "$subject_id" "$subject_input_dir"; then
    log "  [skip] Registered DTI inputs already exist."
    return 0
  fi

  if [[ "$SKIP_DTI_PROCESSING" -eq 1 ]]; then
    log "  [warn] Registered DTI inputs are missing and DTI processing was disabled."
    return 1
  fi

  if [[ ! -d "$raw_subject_dir" ]]; then
    log "  [warn] Raw diffusion directory not found: $raw_subject_dir"
    return 1
  fi

  log "  [run] Computing DTI scalar maps from raw diffusion inputs"
  "${command[@]}" >>"$log_file" 2>&1 || return 1

  if ! subject_has_registered_dti_inputs "$subject_id" "$subject_input_dir"; then
    log "  [warn] DTI processing finished but required registered inputs are still missing."
    return 1
  fi

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

  log "Step 1 | Input validation"
  local -a required_inputs=(
    "${subject_input_dir}/${subject_id}-dti-FractionalAnisotropy-Reg.nii.gz"
    "${subject_input_dir}/${subject_id}-dti-MinEigenvalue-Reg.nii.gz"
    "${subject_input_dir}/${subject_id}-dti-MidEigenvalue-Reg.nii.gz"
    "${subject_input_dir}/${subject_id}-dti-MaxEigenvalue-Reg.nii.gz"
    "${subject_input_dir}/${subject_id}-dti-Trace-Reg.nii.gz"
    "${subject_input_dir}/${subject_id}-dti-MeanDiffusivity-Reg.nii.gz"
    "$subject_mask"
    "${subject_input_dir}/${subject_id}-b0ToAtlasT2.tfm"
  )

  local missing_required=0
  local path
  for path in "${required_inputs[@]}"; do
    if ! file_is_usable "$path"; then
      log "  Missing required input: $path"
      missing_required=1
    fi
  done
  if [[ $missing_required -ne 0 ]]; then
    log "  Skipping subject ${subject_id} due to missing inputs."
    return 1
  fi

  mkdir -p "$subject_output_dir"
  cp -f "${subject_input_dir}/${subject_id}-b0ToAtlasT2.tfm" "${subject_output_dir}/${subject_id}-b0ToAtlasT2.tfm"

  log "Step 2 | Skull stripping"
  local param
  for param in "${_DTI_PARAMS_TO_MASK[@]}"; do
    local input_volume="${subject_input_dir}/${subject_id}-dti-${param}-Reg.nii.gz"
    local masked_volume="${subject_input_dir}/${subject_id}-dti-${param}-Reg-masked.nii.gz"
    if file_is_usable "$masked_volume"; then
      log "  [skip] ${param} already skull stripped."
      continue
    fi
    log "  [run] Skull stripping ${param}"
    "$PYTHON_BIN" "$PYTHON_SKULL_STRIPPING_SCRIPT" \
      --input_path "$input_volume" \
      --mask_path "$subject_mask" \
      --output_path "$masked_volume" \
      >>"$log_file" 2>&1
  done

  log "Step 3 | Resampling to template space"
  local resampled_mask="${subject_output_dir}/${subject_id}-brainmask_resampled.nii.gz"
  if file_is_usable "$resampled_mask"; then
    log "  [skip] Resampled brain mask already available."
  else
    log "  [run] Resampling brain mask"
    run_resample "$subject_mask" "$resampled_mask"
  fi

  local descriptor
  for descriptor in "${_FILES_TO_RESAMPLE[@]}"; do
    IFS=':' read -r source_scope source_stem target_suffix <<<"$descriptor"
    local source_path=""
    case "$source_scope" in
      testdata) source_path="${subject_input_dir}/${subject_id}-${source_stem}.nii.gz" ;;
      *) log "  [warn] Unknown source scope '${source_scope}' for descriptor '${descriptor}'"; continue ;;
    esac

    if [[ ! -f "$source_path" ]]; then
      log "  [warn] Source missing, skipping resample: $(basename "$source_path")"
      continue
    fi

    local target_path="${subject_output_dir}/${subject_id}-${target_suffix}.nii.gz"
    if file_is_usable "$target_path"; then
      log "  [skip] Resampled volume already exists: $(basename "$target_path")"
      continue
    fi

    log "  [run] Resampling $(basename "$source_path") -> $(basename "$target_path")"
    run_resample "$source_path" "$target_path"
  done

  log "Step 4 | Z-score normalisation"
  local zscore_args=()
  if [[ -f "$resampled_mask" ]]; then
    zscore_args=(--mask_file "$resampled_mask")
  else
    log "  [warn] Resampled mask not found. Z-scoring will proceed without an explicit mask."
  fi

  for descriptor in "${_FILES_TO_RESAMPLE[@]}"; do
    IFS=':' read -r _ source_stem target_suffix <<<"$descriptor"
    local volume="${subject_output_dir}/${subject_id}-${target_suffix}.nii.gz"
    if [[ ! -f "$volume" ]]; then
      log "  [warn] Skipping Z-score, volume not found: $(basename "$volume")"
      continue
    fi

    log "  [run] Z-score normalisation (in-place): $(basename "$volume")"
    "$PYTHON_BIN" "$PYTHON_ZSCORE_SCRIPT" \
      --input_file "$volume" \
      --output_file "$volume" \
      "${zscore_args[@]}" \
      >>"$log_file" 2>&1
  done

  log "Completed subject ${subject_id}"
}

###############################################################################
# Script entry point
###############################################################################

parse_args "$@"
initialise_logging
check_tooling
ensure_subject_list

log "Subjects to process: ${SUBJECTS[*]}"
for subject_id in "${SUBJECTS[@]}"; do
  process_subject "$subject_id"
done

log ""
log "==================================================================="
log "All preprocessing tasks completed: $(date)"
log "==================================================================="
