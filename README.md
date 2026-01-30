## DDSurfer Release

DDSurfer reconstructs white and pial cortical surfaces from diffusion MRI inputs.
This release bundles preprocessing utilities, dual-stream TANet inference, and
post-processing tools in a single repository.

### Typical Workflow

1. **Prepare data**  
   Place each subject under `DTI-inputs/<subID>` with the diffusion-derived
   volumes required by `Data-Preprocessing.sh` (FA, eigenvalues, trace, MD,
   brain mask, transform).

2. **Run preprocessing**  
   ```bash
   bash Data-Preprocessing.sh --subject <subID>
   ```
   Use `--input-root` / `--output-root` if your directories differ from the
   defaults.

3. **Predict cortical surfaces (MNI space outputs)**  
   ```bash
   python3 ddsurfer_predict_lh_dualstream.py --subjects <subID>
   python3 ddsurfer_predict_rh_dualstream.py --subjects <subID>
   ```
   Adjust `--device`, `--data_type`, and `--input_root` / `--output_dir` as
   needed. The scripts require the outputs from preprocessing and write meshes
   to `pred_results_DDSurfer/mni/<subID>/`.

4. **Transform meshes back to native space**  
   ```bash
   bash utils/space_MNI2orig.sh --subject <subID> --mode whole
   ```
   The resulting surfaces are written to `pred_results_DDSurfer/native/<subID>/`.

5. **One-command pipeline (Python)**  
   ```bash
   python3 run_ddsurfer_pipeline.py --subject <subID>
   ```

6. **One-command pipeline (Shell)**  
   ```bash
   bash run_ddsurfer_pipeline.sh --subject <subID>
   ```

Both wrappers accept flags to skip preprocessing or post-processing and expose
the same configuration knobs (input/output roots, device, prediction mode).

### Key Dependencies

- Python 3.8+
- PyTorch with CUDA support (optional for GPU acceleration)
- SimpleITK, nibabel, trimesh, tqdm, pytorch3d (for loss utilities)

Refer to project-specific requirements for exact versions used during training.

### Support

Open issues or questions can be directed through the repository’s issue tracker.

## Acknowledgments

This work is in part supported by the National Key R&D Program of China (No. 2023YFE0118600), the National Natural Science Foundation of China (No. 62371107).
