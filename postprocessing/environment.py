"""Validate the FreeSurfer tools and atlas files used by surface processing."""
import os
from pathlib import Path


def check_environment(home, hemisphere='both', atlas_names='aparc,aparc.a2009s'):
    if home is None:
        raise ValueError('Set FREESURFER_HOME or pass --freesurfer-home')
    home = Path(home).expanduser().resolve()
    hemis = ('lh', 'rh') if hemisphere == 'both' else (('lh',) if hemisphere == 'left' else ('rh',))
    atlases = tuple(dict.fromkeys(name.strip() for name in atlas_names.split(',') if name.strip()))
    if not atlases or any(Path(a).name != a or a in ('.', '..') for a in atlases):
        raise ValueError('Specify valid cortical atlas names')
    modern = os.access(str(home / 'bin/mris_place_surface'), os.X_OK)
    metric = 'mris_place_surface' if modern else 'mris_thickness'
    names = ('mris_curvature', 'mris_inflate', 'mris_sphere', 'mris_register', metric,
             'mri_label2label', 'mri_surf2surf', 'mri_annotation2label')
    binaries = [home / 'bin' / name for name in names]
    for binary in binaries:
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise FileNotFoundError(f'FreeSurfer executable not available: {binary.name}')
    license_file = Path(os.environ.get('FS_LICENSE', str(home / 'license.txt'))).expanduser().resolve()
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
    return dict(home=home, hemis=hemis, atlases=atlases, modern_metrics=modern,
                metric_binary=metric, license_file=license_file, fsaverage=fsaverage,
                files=binaries + references)
