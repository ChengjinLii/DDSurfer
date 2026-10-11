"""Pack surface overlays for one FreeSurfer mapping and preserve its scalar outputs."""
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np

FIELDS = ('thickness', 'curv', 'sulc', 'area.pial')


def write_stack(surf, hemi, destination, vertex_count, fields=None):
    values = []
    for field in FIELDS:
        data = (fields[field] if fields is not None and field in fields
                else nib.freesurfer.read_morph_data(str(Path(surf) / f'{hemi}.{field}')))
        if data.shape != (vertex_count,) or not np.isfinite(data).all():
            raise ValueError(f'Invalid {hemi}.{field} overlay')
        values.append(data)
    data = np.stack(values, axis=-1)[:, None, None, :].astype(np.float32)
    header = nib.freesurfer.mghformat.MGHHeader()
    header.set_data_shape(data.shape)
    header.set_data_dtype(np.float32)
    # Match the synthetic MRI header used by FreeSurfer's curvature-file reader.
    header['dof'] = 1
    header['Mdc'] = [[-1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]
    nib.save(nib.MGHImage(data, header.get_affine(), header), str(destination))


def copy_frames(mapped, outputs):
    mapped = Path(mapped)
    if len(outputs) != len(FIELDS):
        raise ValueError('Each mapped surface overlay needs one output')
    sources = [mapped.with_name(f'{mapped.stem}{frame:04d}{mapped.suffix}') for frame in range(len(FIELDS))]
    reference = None
    for source in sources:
        image = nib.MGHImage.from_bytes(source.read_bytes())
        data = np.asarray(image.dataobj)
        if (data.ndim != 3 or data.shape[1:] != (1, 1) or not data.shape[0] or
                not np.isfinite(data).all() or not np.isfinite(image.affine).all()):
            raise ValueError('Invalid mapped surface overlay frame')
        if reference is None:
            reference = image
        elif image.shape != reference.shape or not np.array_equal(image.affine, reference.affine):
            raise ValueError('Mapped surface overlay frames disagree')
    # FreeSurfer --split writes complete MGH metadata; do not reserialize with nibabel.
    for source, output in zip(sources, outputs):
        shutil.copyfile(source, output)
