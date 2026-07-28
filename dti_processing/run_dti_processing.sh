#!/usr/bin/env bash

###############################################################################
# DDSurfer DTI Processing
#
# This script estimates the diffusion tensor, derives scalar maps, registers
# them to the atlas reference image, and writes the registered outputs into the
# DDSurfer `DTI-inputs/<subject>` layout consumed by the main preprocessing
# pipeline.
###############################################################################

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

SLICER_PATH=${SLICER_PATH:-/data01/software/slicer/Slicer-5.2.2-linux-amd64}
REFERENCE_IMAGE=${REFERENCE_IMAGE:-/data/chengjin/DDSurfer/template/100HCP-population-mean-T2-1mm.nii.gz}
PYTHON_BIN=${PYTHON_BIN:-python3}
INPUT_ROOT=${INPUT_ROOT:-"$PROJECT_ROOT/raw-dwi-inputs"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$PROJECT_ROOT/DTI-inputs"}
MASK_FLIP=${MASK_FLIP:-1}

SUBJECT_ID=""

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

run_python_helper() {
  OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} \
  OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} \
  MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} \
  NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1} \
  ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS:-1} \
  KMP_AFFINITY=${KMP_AFFINITY:-disabled} \
  KMP_BLOCKTIME=${KMP_BLOCKTIME:-0} \
  OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE} \
  "$PYTHON_BIN" "$@"
}

usage() {
  cat <<'USAGE'
Usage: dti_processing/run_dti_processing.sh --subject <ID> [options]

Required:
  -s, --subject <ID>          Subject identifier.

Options:
      --input-root <path>     Root containing <ID>/T1w/Diffusion raw inputs.
      --output-root <path>    Root where DDSurfer DTI inputs are written.
      --slicer-path <path>    Slicer installation used for DMRI CLI modules.
      --reference-image <path> Reference atlas/T2 image for registration.
      --python-bin <bin>      Python interpreter for helper conversion scripts.
      --mask-flip <mode>      Mask flip mode for normalization (default: 1).
  -h, --help                  Show this message and exit.
USAGE
}

while (($#)); do
  case "$1" in
    -s|--subject)
      [[ $# -ge 2 ]] || die "Option $1 requires an argument"
      SUBJECT_ID=$2
      shift 2
      ;;
    --input-root)
      [[ $# -ge 2 ]] || die "Option $1 requires an argument"
      INPUT_ROOT=$2
      shift 2
      ;;
    --output-root)
      [[ $# -ge 2 ]] || die "Option $1 requires an argument"
      OUTPUT_ROOT=$2
      shift 2
      ;;
    --slicer-path)
      [[ $# -ge 2 ]] || die "Option $1 requires an argument"
      SLICER_PATH=$2
      shift 2
      ;;
    --reference-image)
      [[ $# -ge 2 ]] || die "Option $1 requires an argument"
      REFERENCE_IMAGE=$2
      shift 2
      ;;
    --python-bin)
      [[ $# -ge 2 ]] || die "Option $1 requires an argument"
      PYTHON_BIN=$2
      shift 2
      ;;
    --mask-flip)
      [[ $# -ge 2 ]] || die "Option $1 requires an argument"
      MASK_FLIP=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -n "$SUBJECT_ID" ]] || die "--subject is required"
[[ -x "$SLICER_PATH/Slicer" ]] || die "Slicer executable not found at $SLICER_PATH/Slicer"
[[ -f "$REFERENCE_IMAGE" ]] || die "Reference image not found at $REFERENCE_IMAGE"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python interpreter not found: $PYTHON_BIN"

if [[ -n "${CONDA_DEFAULT_ENV:-}" ]] \
  && [[ "${CONDA_DEFAULT_ENV}" != "DDParcel" ]] \
  && [[ "${CONDA_DEFAULT_ENV}" != "dti-processing" ]]; then
  log "WARNING: expected conda env 'dti-processing' or 'DDParcel', current env is '${CONDA_DEFAULT_ENV}'."
fi

PYTHON_SCRIPTS_PATH="$SCRIPT_DIR/conversion"
INPUT_DIR="$INPUT_ROOT/$SUBJECT_ID/T1w/Diffusion"
OUTPUT_DIR="$OUTPUT_ROOT/$SUBJECT_ID"

[[ -d "$INPUT_DIR" ]] || die "Input diffusion directory not found: $INPUT_DIR"
mkdir -p "$OUTPUT_DIR"

if [[ -f "$INPUT_DIR/data.nii.gz" ]]; then
  DWI="$INPUT_DIR/data.nii.gz"
elif [[ -f "$INPUT_DIR/data.nii" ]]; then
  DWI="$INPUT_DIR/data.nii"
else
  die "Could not find diffusion volume at $INPUT_DIR/data.nii.gz or $INPUT_DIR/data.nii"
fi

BVAL="$INPUT_DIR/bvals"
BVEC="$INPUT_DIR/bvecs"
MASK="$INPUT_DIR/nodif_brain_mask.nii.gz"

for required_file in "$BVAL" "$BVEC" "$MASK"; do
  [[ -f "$required_file" ]] || die "Required file not found: $required_file"
done

link_input_into_output_dir() {
  local source_path=$1
  local target_path="$OUTPUT_DIR/$(basename "$source_path")"
  ln -sfn "$source_path" "$target_path"
  printf '%s\n' "$target_path"
}

DWI=$(link_input_into_output_dir "$DWI")
MASK=$(link_input_into_output_dir "$MASK")

prepend_ld_library_path() {
  local path_to_add=$1
  if [[ -d "$path_to_add" ]]; then
    export LD_LIBRARY_PATH="$path_to_add:${LD_LIBRARY_PATH:-}"
  fi
}

resolve_slicer_lib_dir() {
  find "$SLICER_PATH/lib" -maxdepth 1 -type d -name 'Slicer-*' | sort | tail -n 1
}

resolve_dmri_cli_dir() {
  local slicer_version_dir=$1
  find "$SLICER_PATH" -type d -path "*/SlicerDMRI/lib/${slicer_version_dir}/cli-modules" | sort | tail -n 1
}

resolve_dmri_qt_dir() {
  local slicer_version_dir=$1
  find "$SLICER_PATH" -type d -path "*/SlicerDMRI/lib/${slicer_version_dir}/qt-loadable-modules" | sort | tail -n 1
}

SLICER_LIB_DIR=$(resolve_slicer_lib_dir)
[[ -n "$SLICER_LIB_DIR" ]] || die "Unable to locate Slicer versioned lib directory under $SLICER_PATH/lib"
SLICER_VERSION_DIR=$(basename "$SLICER_LIB_DIR")

CLI_MODULES_PATH=$(resolve_dmri_cli_dir "$SLICER_VERSION_DIR")
CLI_MODULES_PATH2="$SLICER_LIB_DIR/cli-modules"
DMRI_QT_MODULES_DIR=$(resolve_dmri_qt_dir "$SLICER_VERSION_DIR")
TEEM_LIB_DIR=$(find "$SLICER_PATH/lib" -maxdepth 1 -type d -name 'Teem-*' | sort | tail -n 1)

[[ -n "$CLI_MODULES_PATH" ]] || die "Unable to locate SlicerDMRI CLI modules under $SLICER_PATH"
[[ -d "$CLI_MODULES_PATH2" ]] || die "Unable to locate core Slicer CLI modules under $SLICER_LIB_DIR"

[[ -x "$CLI_MODULES_PATH/DWIToDTIEstimation" ]] || die "DWIToDTIEstimation not found under $CLI_MODULES_PATH"
[[ -x "$CLI_MODULES_PATH/DiffusionTensorScalarMeasurements" ]] || die "DiffusionTensorScalarMeasurements not found under $CLI_MODULES_PATH"
[[ -x "$CLI_MODULES_PATH2/BRAINSFit" ]] || die "BRAINSFit not found under $CLI_MODULES_PATH2"
[[ -x "$CLI_MODULES_PATH2/ResampleScalarVectorDWIVolume" ]] || die "ResampleScalarVectorDWIVolume not found under $CLI_MODULES_PATH2"

prepend_ld_library_path "$SLICER_PATH/lib"
prepend_ld_library_path "$SLICER_LIB_DIR"
prepend_ld_library_path "$DMRI_QT_MODULES_DIR"
prepend_ld_library_path "$TEEM_LIB_DIR"
prepend_ld_library_path "$SLICER_PATH/lib/Python/lib"
prepend_ld_library_path "$CLI_MODULES_PATH2"
prepend_ld_library_path "/home/chengjin/miniconda3/lib"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  prepend_ld_library_path "$CONDA_PREFIX/lib"
fi

DWI_TO_DTI_ESTIMATION=("$SLICER_PATH/Slicer" --launch "$CLI_MODULES_PATH/DWIToDTIEstimation")
DTI_SCALARS=("$SLICER_PATH/Slicer" --launch "$CLI_MODULES_PATH/DiffusionTensorScalarMeasurements")
BRAINS_FIT=("$SLICER_PATH/Slicer" --launch "$CLI_MODULES_PATH2/BRAINSFit")
RESAMPLE_VOLUME=("$SLICER_PATH/Slicer" --launch "$CLI_MODULES_PATH2/ResampleScalarVectorDWIVolume")

NRRD_DWI="$OUTPUT_DIR/$SUBJECT_ID.nhdr"
NRRD_MASK="$OUTPUT_DIR/$SUBJECT_ID-mask.nhdr"
NRRD_DTI="$OUTPUT_DIR/$SUBJECT_ID-dti.nhdr"
NRRD_B0="$OUTPUT_DIR/$SUBJECT_ID-b0.nhdr"

NRRD_FA="$OUTPUT_DIR/$SUBJECT_ID-dti-FractionalAnisotropy.nhdr"
NRRD_TRACE="$OUTPUT_DIR/$SUBJECT_ID-dti-Trace.nhdr"
NRRD_MINEIG="$OUTPUT_DIR/$SUBJECT_ID-dti-MinEigenvalue.nhdr"
NRRD_MIDEIG="$OUTPUT_DIR/$SUBJECT_ID-dti-MidEigenvalue.nhdr"
NRRD_MAXEIG="$OUTPUT_DIR/$SUBJECT_ID-dti-MaxEigenvalue.nhdr"
NRRD_MD="$OUTPUT_DIR/$SUBJECT_ID-dti-MeanDiffusivity.nhdr"

TFM="$OUTPUT_DIR/$SUBJECT_ID-b0ToAtlasT2.tfm"

NII_FA_REG="$OUTPUT_DIR/$SUBJECT_ID-dti-FractionalAnisotropy-Reg.nii.gz"
NII_TRACE_REG="$OUTPUT_DIR/$SUBJECT_ID-dti-Trace-Reg.nii.gz"
NII_MINEIG_REG="$OUTPUT_DIR/$SUBJECT_ID-dti-MinEigenvalue-Reg.nii.gz"
NII_MIDEIG_REG="$OUTPUT_DIR/$SUBJECT_ID-dti-MidEigenvalue-Reg.nii.gz"
NII_MD_REG="$OUTPUT_DIR/$SUBJECT_ID-dti-MeanDiffusivity-Reg.nii.gz"
NII_MAXEIG_REG="$OUTPUT_DIR/$SUBJECT_ID-dti-MaxEigenvalue-Reg.nii.gz"
NII_MASK_REG="$OUTPUT_DIR/$SUBJECT_ID-mask-Reg.nii.gz"

NII_FA_REG_NORM="$OUTPUT_DIR/$SUBJECT_ID-dti-FractionalAnisotropy-Reg-NormMasked.nii.gz"
NII_TRACE_REG_NORM="$OUTPUT_DIR/$SUBJECT_ID-dti-Trace-Reg-NormMasked.nii.gz"
NII_MINEIG_REG_NORM="$OUTPUT_DIR/$SUBJECT_ID-dti-MinEigenvalue-Reg-NormMasked.nii.gz"
NII_MIDEIG_REG_NORM="$OUTPUT_DIR/$SUBJECT_ID-dti-MidEigenvalue-Reg-NormMasked.nii.gz"

log "Starting DTI processing for $SUBJECT_ID"
log "Raw diffusion input: $INPUT_DIR"
log "DDSurfer DTI output: $OUTPUT_DIR"
log "Reference image: $REFERENCE_IMAGE"

if [[ ! -f "$NRRD_DWI" || ! -f "$NRRD_MASK" ]]; then
  log "Converting diffusion volume and mask to NHDR"
  run_python_helper "$PYTHON_SCRIPTS_PATH/nhdr_write.py" --nifti "$DWI" --bval "$BVAL" --bvec "$BVEC" --nhdr "$NRRD_DWI"
  run_python_helper "$PYTHON_SCRIPTS_PATH/nhdr_write.py" --nifti "$MASK" --nhdr "$NRRD_MASK"
fi

if [[ ! -f "$NRRD_DTI" || ! -f "$NRRD_B0" ]]; then
  log "Estimating diffusion tensor and b0 image"
  "${DWI_TO_DTI_ESTIMATION[@]}" --enumeration WLS "$NRRD_DWI" "$NRRD_DTI" "$NRRD_B0"
fi

if [[ ! -f "$NRRD_FA" || ! -f "$NRRD_TRACE" || ! -f "$NRRD_MINEIG" || ! -f "$NRRD_MIDEIG" || ! -f "$NRRD_MAXEIG" || ! -f "$NRRD_MD" ]]; then
  log "Computing tensor-derived scalar maps"
  "${DTI_SCALARS[@]}" --enumeration FractionalAnisotropy "$NRRD_DTI" "$NRRD_FA"
  "${DTI_SCALARS[@]}" --enumeration Trace "$NRRD_DTI" "$NRRD_TRACE"
  "${DTI_SCALARS[@]}" --enumeration MinEigenvalue "$NRRD_DTI" "$NRRD_MINEIG"
  "${DTI_SCALARS[@]}" --enumeration MidEigenvalue "$NRRD_DTI" "$NRRD_MIDEIG"
  "${DTI_SCALARS[@]}" --enumeration MaxEigenvalue "$NRRD_DTI" "$NRRD_MAXEIG"
  "${DTI_SCALARS[@]}" --enumeration MeanDiffusivity "$NRRD_DTI" "$NRRD_MD"
fi

if [[ ! -f "$TFM" ]]; then
  log "Registering b0 to atlas reference"
  "${BRAINS_FIT[@]}" \
    --fixedVolume "$REFERENCE_IMAGE" \
    --movingVolume "$NRRD_B0" \
    --linearTransform "$TFM" \
    --useRigid \
    --useAffine
fi

if [[ ! -f "$NII_FA_REG" || ! -f "$NII_TRACE_REG" || ! -f "$NII_MINEIG_REG" || ! -f "$NII_MIDEIG_REG" || ! -f "$NII_MAXEIG_REG" || ! -f "$NII_MD_REG" || ! -f "$NII_MASK_REG" ]]; then
  log "Resampling scalar maps into atlas reference space"
  "${RESAMPLE_VOLUME[@]}" -i linear "$NRRD_FA" --Reference "$REFERENCE_IMAGE" --transformationFile "$TFM" "$NII_FA_REG"
  "${RESAMPLE_VOLUME[@]}" -i linear "$NRRD_TRACE" --Reference "$REFERENCE_IMAGE" --transformationFile "$TFM" "$NII_TRACE_REG"
  "${RESAMPLE_VOLUME[@]}" -i linear "$NRRD_MINEIG" --Reference "$REFERENCE_IMAGE" --transformationFile "$TFM" "$NII_MINEIG_REG"
  "${RESAMPLE_VOLUME[@]}" -i linear "$NRRD_MIDEIG" --Reference "$REFERENCE_IMAGE" --transformationFile "$TFM" "$NII_MIDEIG_REG"
  "${RESAMPLE_VOLUME[@]}" -i linear "$NRRD_MD" --Reference "$REFERENCE_IMAGE" --transformationFile "$TFM" "$NII_MD_REG"
  "${RESAMPLE_VOLUME[@]}" -i linear "$NRRD_MAXEIG" --Reference "$REFERENCE_IMAGE" --transformationFile "$TFM" "$NII_MAXEIG_REG"
  "${RESAMPLE_VOLUME[@]}" -i nn "$MASK" --Reference "$REFERENCE_IMAGE" --transformationFile "$TFM" "$NII_MASK_REG"
fi

if [[ ! -f "$NII_FA_REG_NORM" || ! -f "$NII_TRACE_REG_NORM" || ! -f "$NII_MINEIG_REG_NORM" || ! -f "$NII_MIDEIG_REG_NORM" ]]; then
  log "Normalizing selected registered scalar maps inside the mask"
  run_python_helper "$SCRIPT_DIR/normalize.py" --input "$NII_FA_REG" --mask "$NII_MASK_REG" --output "$NII_FA_REG_NORM" --flip "$MASK_FLIP"
  run_python_helper "$SCRIPT_DIR/normalize.py" --input "$NII_TRACE_REG" --mask "$NII_MASK_REG" --output "$NII_TRACE_REG_NORM" --flip "$MASK_FLIP"
  run_python_helper "$SCRIPT_DIR/normalize.py" --input "$NII_MINEIG_REG" --mask "$NII_MASK_REG" --output "$NII_MINEIG_REG_NORM" --flip "$MASK_FLIP"
  run_python_helper "$SCRIPT_DIR/normalize.py" --input "$NII_MIDEIG_REG" --mask "$NII_MASK_REG" --output "$NII_MIDEIG_REG_NORM" --flip "$MASK_FLIP"
fi

for expected_output in \
  "$TFM" \
  "$NII_MASK_REG" \
  "$NII_FA_REG" \
  "$NII_TRACE_REG" \
  "$NII_MINEIG_REG" \
  "$NII_MIDEIG_REG" \
  "$NII_MAXEIG_REG" \
  "$NII_MD_REG"; do
  [[ -f "$expected_output" ]] || die "Expected output missing: $expected_output"
done

log "DTI processing completed successfully for $SUBJECT_ID"
