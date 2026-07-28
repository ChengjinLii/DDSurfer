## DDSurfer Release

DDSurfer reconstructs white and pial cortical surfaces from diffusion MRI inputs.
This release bundles preprocessing utilities, dual-stream TANet inference, and
post-processing tools in a single repository.

## Publication

DDSurfer has been accepted and published online in **Advanced Science**:

> Chengjin Li, Wei Zhang, Xi Zhu, Yuqian Chen, Nir A. Sochen, Jarrett Rushmore,
> Carl-Fredrik Westin, Yogesh Rathi, Lauren J. O'Donnell, Ofer Pasternak, and
> Fan Zhang. "DDSurfer: A Weakly-Supervised Dual-Stream Deep Learning Framework
> for Cortical Surface Reconstruction From Diffusion MRI." *Advanced Science*
> (2026): e76596. https://doi.org/10.1002/advs.76596

If you use DDSurfer in your research, please cite:

```bibtex
@article{Li2026DDSurfer,
  author  = {Li, Chengjin and Zhang, Wei and Zhu, Xi and Chen, Yuqian and
             Sochen, Nir A. and Rushmore, Jarrett and Westin, Carl-Fredrik and
             Rathi, Yogesh and O'Donnell, Lauren J. and Pasternak, Ofer and
             Zhang, Fan},
  title   = {DDSurfer: A Weakly-Supervised Dual-Stream Deep Learning Framework
             for Cortical Surface Reconstruction From Diffusion MRI},
  journal = {Advanced Science},
  year    = {2026},
  pages   = {e76596},
  doi     = {10.1002/advs.76596},
  url     = {https://doi.org/10.1002/advs.76596}
}
```

### Typical Workflow

1. **Prepare data**  
   Use one of these input layouts:
   - place raw diffusion inputs under `raw-dwi-inputs/<subID>/T1w/Diffusion/`
   - or place precomputed registered DTI inputs under `DTI-inputs/<subID>/`

2. **Run preprocessing**  
   ```bash
   bash Data-Preprocessing.sh --subject <subID>
   ```
   The preprocessing stages are DTI estimation, atlas registration, skull
   stripping, template-space resampling, and z-score normalisation. Use
   `--raw-input-root`, `--input-root`, and `--output-root` if your directories
   differ from the defaults.

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
