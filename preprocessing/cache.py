"""Validation of scalar images, transforms, and detached NRRD payloads."""
from pathlib import Path

import nibabel as nib
import numpy as np
import nrrd
import SimpleITK as sitk


def related_files(path):
    path = Path(path)
    files = [path]
    if path.suffix.lower() == '.nhdr':
        try:
            header = nrrd.read_header(str(path))
        except (nrrd.NRRDError, UnicodeError) as error:
            raise ValueError(f'Invalid NRRD header: {path}') from error
        name = header.get('data file', header.get('datafile'))
        if not isinstance(name, str) or name.startswith('LIST'):
            raise ValueError(f'Expected one detached NRRD data file: {path}')
        files.append(path.parent / name)
    elif path.suffix.lower() == '.tfm':
        inverse = path.with_name(path.stem + '_Inverse.h5')
        if inverse.is_file():
            files.append(inverse)
    return files


def validate_output(path):
    path = Path(path)
    for file in related_files(path):
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f'Missing or empty image data: {file}')
    if str(path).endswith(('.nii', '.nii.gz', '.mgz', '.mgh')):
        image = nib.load(str(path))
        if not np.isfinite(image.affine).all() or not np.isfinite(np.asarray(image.dataobj)).all():
            raise ValueError(f'Nonfinite stage output: {path}')
    elif path.suffix.lower() == '.tfm':
        sitk.ReadTransform(str(path))
    elif path.suffix.lower() in ('.nhdr', '.nrrd'):
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        if path.suffix.lower() == '.nhdr' and not related_files(path)[1].is_symlink():
            # Validate owned payloads too; an intact header cannot prove a write completed.
            try:
                nrrd.read_data(nrrd.read_header(str(path)), filename=str(path))
            except (nrrd.NRRDError, EOFError, OSError) as error:
                raise ValueError(f'Invalid detached NRRD data: {path}') from error
