"""Regional surface statistics without requiring a tissue segmentation volume."""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def write_stats(subject_dir, hemi, atlas):
    subject_dir = Path(subject_dir)
    surf, label = subject_dir / 'surf', subject_dir / 'label'
    indices, _, names = nib.freesurfer.read_annot(str(label / f'{hemi}.{atlas}.annot'))
    count = len(indices)
    cortex = np.zeros(count, dtype=bool)
    cortex_indices = nib.freesurfer.read_label(str(label / f'{hemi}.cortex.label'))
    if np.any(cortex_indices < 0) or np.any(cortex_indices >= count):
        raise ValueError('Cortex label contains invalid vertex indices')
    cortex[cortex_indices] = True
    fields = {}
    for field in ('area', 'area.pial', 'thickness', 'curv', 'white.K'):
        data = nib.freesurfer.read_morph_data(str(surf / f'{hemi}.{field}'))
        if len(data) != count or not np.isfinite(data).all():
            raise ValueError(f'Invalid {hemi}.{field} overlay')
        fields[field] = data
    rows = []
    for index, name in enumerate(names):
        name = name.decode()
        if name.lower() in ('unknown', '???', 'medial_wall', 'corpuscallosum'):
            continue
        mask = cortex & (indices == index)
        if not mask.any():
            continue
        thickness = fields['thickness'][mask]
        rows.append((name, int(mask.sum()), float(fields['area'][mask].sum()),
                     float(fields['area.pial'][mask].sum()), float(thickness.mean()), float(thickness.std()),
                     float(np.abs(fields['curv'][mask]).mean()), float(np.abs(fields['white.K'][mask]).mean())))
    if not rows:
        raise ValueError('No annotated cortical vertices found')
    output = subject_dir / 'stats' / f'{hemi}.{atlas}.stats'
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = (('StructName', 'NA'), ('NumVert', 'unitless'), ('SurfArea', 'mm^2'), ('PialArea', 'mm^2'),
               ('ThickAvg', 'mm'), ('ThickStd', 'mm'), ('MeanAbsCurv', 'mm^-1'), ('MeanAbsGaussCurv', 'mm^-2'))
    lines = ['# DDSurfer regional surface statistics', '# Cortex label excludes non-cortical vertices.',
             '# Area is allocated to incident vertices; thickness mean/std use unweighted vertices (ddof=0).',
             '# No voxel-wise segmentation volumes or intracranial volume estimates are included.',
             f'# subjectname {subject_dir.name}', f'# hemi {hemi}', f'# annotation {atlas}',
             f'# NRows {len(rows)}', f'# NTableCols {len(columns)}']
    for index, (column, unit) in enumerate(columns, 1):
        lines += [f'# TableCol {index} ColHeader {column}', f'# TableCol {index} Units {unit}']
    lines += ['# ColHeaders ' + ' '.join(column for column, _ in columns)]
    for row in rows:
        lines.append(f'{row[0]} {row[1]} ' + ' '.join(f'{value:.6f}' for value in row[2:]))
    temp = output.with_suffix('.tmp')
    temp.write_text('\n'.join(lines) + '\n')
    temp.replace(output)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject-dir', type=Path, required=True)
    parser.add_argument('--hemi', choices=('lh', 'rh'), required=True)
    parser.add_argument('--atlas', required=True)
    args = parser.parse_args()
    write_stats(args.subject_dir, args.hemi, args.atlas)
