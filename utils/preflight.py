"""Validate a pipeline run before starting expensive image processing."""

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil

from preprocessing.inputs import fingerprint, validate_diffusion_inputs, validate_mask
from utils.files import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_digest(path):
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(f'Missing or empty file: {path}')
    return sha256_file(path)


def asset(folder, name):
    if Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('Model asset names must be plain filenames')
    return Path(folder) / name


def slicer_files(required=True):
    selected = os.environ.get('DTI_SLICER_PATH') or os.environ.get('SLICER_PATH')
    found = shutil.which('Slicer') if not selected else None
    home = Path(selected) if selected else (Path(found).parent if found else None)
    if home is None:
        if required:
            raise FileNotFoundError('Set SLICER_PATH to a Slicer installation with SlicerDMRI')
        return []
    versions = sorted(path for path in (home / 'lib').glob('Slicer-*') if path.is_dir())
    extensions = sorted(home.glob(f'**/SlicerDMRI/lib/{versions[-1].name}/cli-modules')) if versions else []
    files = [home / 'Slicer']
    if versions and extensions:
        files += [extensions[-1] / name for name in ('DWIToDTIEstimation', 'DiffusionTensorScalarMeasurements')]
        files += [versions[-1] / 'cli-modules' / name for name in ('BRAINSFit', 'ResampleScalarVectorDWIVolume')]
    elif required:
        raise FileNotFoundError('Slicer core and SlicerDMRI CLI modules are required')
    if required:
        for path in files:
            if not path.is_file() or not os.access(path, os.X_OK):
                raise FileNotFoundError(f'Slicer module missing or not executable: {path}')
    return files


def reference_image():
    return Path(os.environ.get('DTI_REFERENCE_IMAGE') or os.environ.get('REFERENCE_IMAGE')
                or ROOT / 'template/100HCP-population-mean-T2-1mm.nii.gz')


def mask_is_ready(args, files):
    root = args.output_root.resolve() / args.subject
    output = root / 'dti' / f'{args.subject}-brainmask.nii.gz'
    record = root / 'logs/mask.json'
    if not output.exists():
        return False
    expected = dict(method='freesurfer_synthstrip', b0_threshold=50.,
                    inputs={name: fingerprint(files[name]) for name in ('dwi', 'bval', 'bvec')})
    if not record.is_file():
        raise ValueError('Existing generated mask has no record; use a new output directory')
    saved = json.loads(record.read_text())
    if saved.get('signature') != expected or saved.get('output') != fingerprint(output):
        raise ValueError('Generated mask or its inputs changed; use a new output directory')
    validate_mask(files['dwi'], output)
    return True


def inspect_run(args, files):
    """Hash the inputs and scientific settings; do not allocate GPU memory."""
    import nibabel as nib
    import numpy as np
    from inference.coordinates import Geometry, MODALITIES
    from inference.predict import load_checksums
    from utils.surface_io import load_mesh

    if args.data_type != 'hcp':
        raise ValueError('These checkpoints require --data-type hcp')
    config_path = args.checkpoint_root / 'manifest.json'
    config = json.loads(config_path.read_text())
    checksums = load_checksums(args.checkpoint_root)
    assets = {'manifest': file_digest(config_path)}
    models = []
    for hemi in ('left', 'right'):
        geometry = Geometry(config['geometry']['affine'], config['geometry']['full_shape'], hemi)
        template = asset(args.template_dir, config['templates'][hemi]['file'])
        template_digest = file_digest(template)
        if template_digest != config['templates'][hemi]['sha256']:
            raise ValueError(f'Template checksum mismatch: {template}')
        mesh = load_mesh(template)
        vertices = np.asarray(mesh.vertices) - geometry.crop_origin
        if (not len(vertices) or not len(mesh.faces) or not np.isfinite(vertices).all()
                or ((vertices < 0) | (vertices > np.array(geometry.crop_shape) - 1)).any()):
            raise ValueError(f'Invalid template or template outside the trained crop: {template}')
        assets[f'{hemi}_template'] = template_digest
        for kind in ('white', 'pial') if args.predict_mode == 'all' else ('white',):
            name = config['checkpoint_files'][hemi][kind]
            path = asset(args.checkpoint_root, name)
            actual = file_digest(path)
            if actual != checksums.get(name):
                raise ValueError(f'Checkpoint checksum mismatch: {path}')
            assets[name] = actual
            models.append((path, actual))

    if files is None:
        # Retained preprocessing is a separate, cache-based mode, not a final-result receipt.
        folder = args.output_root.resolve() / args.subject / '.cache/volumes' / args.subject
        for modality in MODALITIES:
            path = folder / f'{args.subject}-{modality}.nii.gz'
            Geometry(config['geometry']['affine'], config['geometry']['full_shape'], 'left').check_image(nib.load(path), path)
        file_digest(folder / f'{args.subject}-b0ToAtlasT2.tfm')
        return dict(signature=None, config=config, models=models)

    validate_diffusion_inputs(files)
    if 'mask' in files:
        validate_mask(files['dwi'], files['mask'])
    else:
        mask_is_ready(args, files)
    inputs = {name: dict(path=str(path), sha256=file_digest(path)) for name, path in files.items()}
    reference = reference_image()
    image = nib.load(str(reference))
    if len(image.shape) != 3 or not np.isfinite(image.affine).all():
        raise ValueError('The registration reference must be a valid 3D image')
    assets['reference'] = file_digest(reference)
    settings = {name: getattr(args, name) for name in (
        'subject', 'data_type', 'device', 'precision', 'predict_mode', 'cpu_threads', 'max_gpu_gib',
        'preprocess_jobs', 'auto_mask', 'mask_threads', 'freesurfer')}
    tools = slicer_files(required=False)
    if args.freesurfer:
        from postprocessing.environment import check_environment
        environment = check_environment(args.freesurfer_home, args.postprocess_hemi, args.postprocess_atlases)
        tools += environment['files']
        settings.update({name: getattr(args, name) for name in (
            'postprocess_hemi', 'postprocess_atlases', 'postprocess_threads', 'postprocess_serial')})
        if args.brain_source is not None:
            from postprocessing.geometry import volume_files
            inputs['brain_source'] = [dict(path=str(path), sha256=file_digest(path))
                                      for path in volume_files(args.brain_source)]
    if 'mask' not in files:
        if args.freesurfer_home is not None:
            home = Path(args.freesurfer_home)
            tools += [home / 'bin/mri_synthstrip', home / 'models/synthstrip.1.pt']
        elif shutil.which('mri_synthstrip'):
            tools.append(Path(shutil.which('mri_synthstrip')))
    tool_digests = {str(path.resolve()): file_digest(path) for path in tools if path.is_file()}
    environment = {name: value for name, value in os.environ.items()
                   if name.startswith(('DTI_', 'PYTHON_', 'ITK_', 'OMP_', 'MKL_', 'OPENBLAS_', 'KMP_', 'NUMEXPR_'))
                   or name in ('REFERENCE_IMAGE', 'MASK_FLIP', 'SKIP_DTI_PROCESSING', 'SLICER_PATH',
                               'FREESURFER_HOME', 'FS_LICENSE', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH')}
    for name in ('DTI_PROCESSING_SCRIPT', 'PYTHON_SKULL_STRIPPING_SCRIPT', 'PYTHON_RESAMPLE_SCRIPT', 'PYTHON_ZSCORE_SCRIPT'):
        if os.environ.get(name):
            assets[name] = file_digest(os.environ[name])
    code = {str(path.relative_to(ROOT)): file_digest(path)
            for folder in ('inference', 'model', 'preprocessing', 'postprocessing', 'utils')
            for path in (ROOT / folder).rglob('*') if path.suffix in ('.py', '.sh')}
    for name in ('run_ddsurfer_pipeline.py', 'DDSurfer_predict.py'):
        code[name] = file_digest(ROOT / name)
    packages = {name: metadata.version(name) for name in ('torch', 'numpy', 'nibabel', 'scipy', 'SimpleITK', 'trimesh', 'pynrrd')}
    signature = digest(dict(inputs=inputs, assets=assets, settings=settings, tools=tool_digests,
                            environment=environment, code=code, packages=packages))
    return dict(signature=signature, config=config, models=models)


def check_device(args, config):
    import torch
    from inference.predict import resolve_precision

    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cpu' and device.index is not None):
        raise ValueError('Use --device cpu or cuda:<index>')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; select --device cpu or install a compatible CUDA runtime')
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device index out of range: {index}')
        with torch.cuda.device(index):
            precision = resolve_precision(args.precision, device, config['precision'])
            free, _ = torch.cuda.mem_get_info(index)
            if min(args.max_gpu_gib, free / 1024**3 - 4) < 8:
                raise RuntimeError('Inference requires an 8-GiB GPU budget plus 4 GiB of free-memory margin')
            torch.empty(1, device=device)
    else:
        precision = resolve_precision(args.precision, device, config['precision'])
    return precision


def check_runtime(args, files, inspection):
    import SimpleITK
    import torch
    from inference.predict import load_model

    check_device(args, inspection['config'])
    # Strict CPU loading catches incompatible weights before any DTI fitting.
    torch.set_num_threads(args.cpu_threads)
    for path, checksum in inspection['models']:
        model = load_model(path, checksum, inspection['config']['architecture'], torch.device('cpu'))
        del model
    if files is not None:
        slicer_files()
        if 'mask' not in files and not mask_is_ready(args, files):
            from preprocessing.brain_mask import synthstrip_environment
            executable, env = synthstrip_environment(args.freesurfer_home, args.mask_threads)
            home = env.get('FREESURFER_HOME')
            if home:
                file_digest(Path(home) / 'models/synthstrip.1.pt')
    if args.freesurfer:
        from postprocessing.environment import check_environment
        check_environment(args.freesurfer_home, args.postprocess_hemi, args.postprocess_atlases)
