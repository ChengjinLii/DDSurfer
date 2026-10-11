"""Extract regional labels without changing annotations or surface coordinates."""
from pathlib import Path

import nibabel as nib
import numpy as np


def read_regions(annotation):
    codes, table, names = nib.freesurfer.read_annot(str(annotation), orig_ids=True)
    if table.shape != (len(names), 5):
        raise ValueError('Annotation needs a complete color table')
    known = set(int(code) for code in table[:, -1])
    if any(int(code) not in known for code in np.unique(codes) if code > 0):
        raise ValueError('Annotation contains colors absent from its color table')
    regions = []
    used_codes, used_names = set(), set()
    for code, name in zip(table[:, -1], names):
        indices = np.flatnonzero(codes == code)
        if not len(indices):
            continue
        name = name.decode('utf-8')
        if (not name or Path(name).name != name or name in ('.', '..') or
                any(character in name for character in ('\x00', '\n', '\r', '/', '\\'))):
            raise ValueError('Invalid annotation region name')
        if int(code) in used_codes or name in used_names:
            raise ValueError('Annotation region names and colors must be unique')
        used_codes.add(int(code))
        used_names.add(name)
        regions.append((name, indices))
    return len(codes), regions


def write_labels(surface, subject, vertex_count, regions, outputs, vertices=None):
    if vertices is None:
        vertices, _ = nib.freesurfer.read_geometry(str(surface))
    if vertices.shape != (vertex_count, 3) or not np.isfinite(vertices).all():
        raise ValueError('Annotation and surface vertices disagree')
    if len(outputs) != len(regions) or not regions:
        raise ValueError('Each annotation region needs one output label')
    if any(character in subject for character in ('\x00', '\n', '\r')):
        raise ValueError('Invalid label subject name')
    for _, indices in regions:
        if np.any(indices < 0) or np.any(indices >= vertex_count):
            raise ValueError('Label contains invalid vertex indices')
    for (_, indices), output in zip(regions, outputs):
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', encoding='utf-8', newline='\n') as stream:
            stream.write(f'#!ascii label  , from subject {subject} vox2ras=TkReg\n{len(indices)}\n')
            # Preserve FreeSurfer's vertex order, surface-RAS coordinates and text precision.
            for start in range(0, len(indices), 4096):
                chosen = indices[start:start + 4096]
                stream.write(''.join('%d  %2.3f  %2.3f  %2.3f 0.0000000000\n' % (index, *point)
                                     for index, point in zip(chosen, vertices[chosen])))
