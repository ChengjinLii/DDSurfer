"""Validate the FreeSurfer tools and atlas files used by surface processing."""
import os
from pathlib import Path


def configured_home(args):
    backend = getattr(args, 'postprocess_backend', getattr(args, 'backend', 'auto'))
    runtime = getattr(args, 'postprocess_runtime', getattr(args, 'core_runtime', None))
    if runtime is not None:
        if backend == 'freesurfer':
            raise ValueError('Choose a local runtime or --postprocess-backend freesurfer, not both')
        return runtime
    return None if backend == 'native' else args.freesurfer_home


def check_environment(home=None, hemisphere='both', atlas_names='aparc,aparc.a2009s', license_file=None, backend='auto'):
    if backend not in ('auto', 'native', 'freesurfer'):
        raise ValueError('Choose postprocessing backend auto, native or freesurfer')
    if backend == 'freesurfer' and home is None:
        raise ValueError('Set FREESURFER_HOME or pass --freesurfer-home for the freesurfer backend')
    if home is None:
        from postprocessing.core.config import default_prefix
        home = default_prefix()
    home = Path(home).expanduser().resolve()
    runtime_manifest = None
    if (home / 'runtime.json').is_file() or backend == 'native':
        if backend == 'freesurfer':
            raise ValueError('This is a local build; select backend native or auto instead')
        from postprocessing.core.runtime import validate_runtime
        runtime_manifest = validate_runtime(home, hemisphere, atlas_names)
    elif not home.is_dir():
        raise FileNotFoundError('Local surface components have not been built. Run '
                                'python -m postprocessing.core.build --jobs 4, or configure FREESURFER_HOME')
    hemis = ('lh', 'rh') if hemisphere == 'both' else (('lh',) if hemisphere == 'left' else ('rh',))
    atlases = tuple(dict.fromkeys(name.strip() for name in atlas_names.split(',') if name.strip()))
    if not atlases or any(Path(a).name != a or a in ('.', '..') for a in atlases):
        raise ValueError('Specify valid cortical atlas names')
    modern = os.access(str(home / 'bin/mris_place_surface'), os.X_OK)
    metric = 'mris_place_surface' if modern else 'mris_thickness'
    names = ('mris_curvature', 'mris_inflate', 'mris_sphere', 'mris_register', metric,
             'mri_label2label', 'mri_surf2surf')
    binaries = [home / 'bin' / name for name in names]
    for binary in binaries:
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise FileNotFoundError(f'FreeSurfer executable not available: {binary.name}')
    license_file = Path(license_file if license_file is not None else
                        os.environ.get('FS_LICENSE', str(home / 'license.txt'))).expanduser().resolve()
    if not license_file.is_file() or not license_file.stat().st_size:
        raise FileNotFoundError('Set FS_LICENSE to a valid FreeSurfer license file')
    fsaverage = home / 'subjects/fsaverage'
    references = []
    for hemi in hemis:
        references += [fsaverage / 'label' / f'{hemi}.cortex.label',
                       fsaverage / 'surf' / f'{hemi}.sphere.reg',
                       home / 'average' / f'{hemi}.average.curvature.filled.buckner40.tif']
        references += [fsaverage / 'label' / f'{hemi}.{atlas}.annot' for atlas in atlases]
    for path in references:
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f'FreeSurfer atlas file missing or empty: {path}')
    files = binaries + references
    if runtime_manifest is not None:
        files += [home / 'runtime.json'] + [home / name for name in runtime_manifest['files']]
    return dict(home=home, hemis=hemis, atlases=atlases, modern_metrics=modern,
                metric_binary=metric, license_file=license_file, fsaverage=fsaverage,
                minimal_runtime=runtime_manifest is not None, files=list(dict.fromkeys(files)))
