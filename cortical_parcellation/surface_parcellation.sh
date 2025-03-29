#!/bin/bash
# set -euo pipefail

# ===================== Environment Setup =====================
source /data06/software/bashrc
export FREESURFER_HOME=/data06/software/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=/data06/chengjin/dwi2surfer/FastSurfer-dev/aaa_test

# ===================== Input Parameters =====================
INPUT_SURFACE="${SUBJECTS_DIR}/surf/100307_predicted_wm_surface_left.stl"
INPUT_VOLUME="/data04/chengjin/DDsurfer/DDSurfer-pre-release/testdata/100307/100307-DDSurfer-wmparc-Reg.mgz"

mris_convert "${SUBJECTS_DIR}/surf/100307_predicted_wm_surface_left.stl" "${SUBJECTS_DIR}/surf/lh.smoothwm"
mris_convert "${SUBJECTS_DIR}/surf/100307_predicted_wm_surface_left.stl" "${SUBJECTS_DIR}/surf/lh.white"

# ===================== Initialization Checks =====================
echo "----------------[ Initialization Checks ]----------------"
[ -f "${INPUT_SURFACE}" ] || { echo "ERROR: Input surface file missing"; exit 1; }
[ -f "${INPUT_VOLUME}" ] || { echo "ERROR: Input volume file missing"; exit 1; }
mkdir -p "${SUBJECTS_DIR}/label"

# ===================== Surface Processing Pipeline =====================
echo -e "\n----------------[ Step 1/7: Surface Inflation ]----------------"
# mris_inflate "${INPUT_SURFACE}" "${SUBJECTS_DIR}/surf/lh.inflated.stl"
mv "${SUBJECTS_DIR}/surf/rh.sulc" "${SUBJECTS_DIR}/surf/lh.sulc"
echo "Surface inflation completed → lh.inflated.stl"

echo -e "\n----------------[ Step 2/7: Generate Spherical Surface ]----------------"
mris_sphere "${SUBJECTS_DIR}/surf/lh.inflated.stl" "${SUBJECTS_DIR}/surf/lh.sphere.stl"
echo "Spherical surface generated → lh.sphere.stl"

echo -e "\n----------------[ Step 3/7: Format Conversion ]----------------"
mris_convert "${SUBJECTS_DIR}/surf/lh.sphere.stl" "${SUBJECTS_DIR}/surf/lh.sphere"
echo "Format conversion completed → lh.sphere"

# ===================== Registration Workflow =====================
echo -e "\n----------------[ Step 4/7: Surface Registration ]----------------"
mris_register "${SUBJECTS_DIR}/surf/lh.sphere.stl" \
    "${FREESURFER_HOME}/average/lh.average.curvature.filled.buckner40.tif" \
    "${SUBJECTS_DIR}/surf/lh.sphere.reg.stl"
echo "Surface registration completed → lh.sphere.reg.stl"

echo -e "\n----------------[ Step 5/7: Convert Registration Result ]----------------"
mris_convert "${SUBJECTS_DIR}/surf/lh.sphere.reg.stl" "${SUBJECTS_DIR}/surf/lh.sphere.reg"
echo "Registration result converted → lh.sphere.reg"

# ===================== Automated Labeling =====================
echo -e "\n----------------[ Step 6/7: Automated Parcellation ]----------------"
mris_ca_label -nbrs 3 \
    -l "${SUBJECTS_DIR}/label/lh.cortex.label" \
    -aseg "${INPUT_VOLUME}" \
    "${SUBJECT_ID}" lh \
    "${SUBJECTS_DIR}/surf/lh.sphere.reg" \
    "${FREESURFER_HOME}/average/lh.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs" \
    "${SUBJECTS_DIR}/label/lh.aparc.annot"
echo "Automated labeling completed → label/lh.aparc.annot"

# ===================== Completion Notice =====================
echo -e "\n\033[32m✔ All processing steps completed successfully!\033[0m"
echo "Output directory: ${SUBJECTS_DIR}"
