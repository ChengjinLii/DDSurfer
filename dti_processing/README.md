# dti_processing

This directory contains the DTI-estimation stage used by
`Data-Preprocessing.sh`.

It covers:

- DWI NIfTI to NHDR conversion
- tensor estimation and b0 extraction
- scalar-map generation
- rigid/affine registration to the atlas reference image
- scalar-map resampling into atlas space
- optional scalar normalization inside the registered mask

Main entry point:

- `run_dti_processing.sh`

Expected raw-input layout:

- `<input_root>/<subject_id>/T1w/Diffusion/data.nii.gz`
- `<input_root>/<subject_id>/T1w/Diffusion/bvals`
- `<input_root>/<subject_id>/T1w/Diffusion/bvecs`
- `<input_root>/<subject_id>/T1w/Diffusion/nodif_brain_mask.nii.gz`

Default output layout:

- `DTI-inputs/<subject_id>/<subject_id>-dti-*-Reg.nii.gz`
- `DTI-inputs/<subject_id>/<subject_id>-mask-Reg.nii.gz`
- `DTI-inputs/<subject_id>/<subject_id>-b0ToAtlasT2.tfm`

Example:

```bash
bash dti_processing/run_dti_processing.sh \
  --subject 100307 \
  --input-root ./raw-dwi-inputs \
  --output-root ./DTI-inputs
```
