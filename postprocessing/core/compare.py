"""Compare decoded postprocessing payloads, ignoring path/time text in headers."""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from utils.files import atomic_json, sha256_file


SURFACES = ('white', 'pial', 'orig', 'smoothwm', 'inflated', 'sphere', 'sphere.reg')
MORPH = ('area', 'area.pial', 'white.H', 'white.K', 'pial.H', 'pial.K',
         'curv', 'curv.pial', 'sulc', 'inflated.H', 'inflated.K', 'thickness')


def _array(a, b):
    a, b = np.asarray(a), np.asarray(b)
    equal = a.shape == b.shape and np.array_equal(a, b)
    result = dict(equal=bool(equal), reference_shape=list(a.shape), candidate_shape=list(b.shape))
    if a.shape == b.shape and a.size and a.dtype.kind in 'fiu' and b.dtype.kind in 'fiu':
        result['max_abs_difference'] = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    return result


def compare(reference, candidate, hemis=('lh',), atlases=('aparc', 'aparc.a2009s')):
    reference, candidate = Path(reference), Path(candidate)
    records = {}

    def record(name, action):
        a, b = reference / name, candidate / name
        if not a.is_file() or not b.is_file():
            records[name] = dict(equal=False, missing_reference=not a.is_file(), missing_candidate=not b.is_file())
            return
        result = action(a, b)
        result['bytes_equal'] = sha256_file(a) == sha256_file(b)
        records[name] = result

    def mesh(a, b):
        av, af, ai = nib.freesurfer.read_geometry(str(a), read_metadata=True)
        bv, bf, bi = nib.freesurfer.read_geometry(str(b), read_metadata=True)
        vertex, faces = _array(av, bv), _array(af, bf)
        geometry_keys = ('valid', 'volume', 'voxelsize', 'xras', 'yras', 'zras', 'cras')
        geometry = all(_array(ai[key], bi[key])['equal'] for key in geometry_keys if key in ai and key in bi)
        geometry = geometry and all((key in ai) == (key in bi) for key in geometry_keys)
        return dict(equal=vertex['equal'] and faces['equal'] and geometry,
                    vertices=vertex, faces=faces, volume_geometry_equal=geometry)

    def annotation(a, b):
        aa, ba = nib.freesurfer.read_annot(str(a)), nib.freesurfer.read_annot(str(b))
        labels, colors = _array(aa[0], ba[0]), _array(aa[1], ba[1])
        names = aa[2] == ba[2]
        return dict(equal=labels['equal'] and colors['equal'] and names,
                    labels=labels, colors=colors, names_equal=names,
                    agreement=float(np.mean(aa[0] == ba[0])) if aa[0].shape == ba[0].shape else None)

    def image(a, b):
        def read(path):
            with nib.openers.ImageOpener(str(path)) as stream:
                img = nib.MGHImage.from_file_map({'image': nib.FileHolder(fileobj=stream)}, mmap=False)
                return np.array(img.dataobj), img.affine.copy(), img.header.binaryblock
        ai, bi = read(a), read(b)
        values, affine = _array(ai[0], bi[0]), _array(ai[1], bi[1])
        header_equal = ai[2] == bi[2]
        return dict(equal=values['equal'] and affine['equal'] and header_equal,
                    values=values, affine=affine, header_equal=header_equal)

    def label(a, b):
        # Compare coordinates and label scalars as well as the membership/order.
        aa, ba = [np.loadtxt(str(path), skiprows=2, ndmin=2) for path in (a, b)]
        return _array(aa, ba)

    def stats(a, b):
        rows = lambda path: [line.strip().split() for line in path.read_text().splitlines()
                             if line.strip() and not line.startswith('#')]
        aa, ba = rows(a), rows(b)
        return dict(equal=aa == ba, reference_rows=len(aa), candidate_rows=len(ba))

    for name in ('brain.mgz', 'orig.mgz'):
        record('mri/' + name, image)
    for hemi in hemis:
        for name in SURFACES:
            record(f'surf/{hemi}.{name}', mesh)
        for name in MORPH:
            record(f'surf/{hemi}.{name}', lambda a, b: _array(
                nib.freesurfer.read_morph_data(str(a)), nib.freesurfer.read_morph_data(str(b))))
        record(f'label/{hemi}.cortex.label', label)
        for atlas in atlases:
            record(f'label/{hemi}.{atlas}.annot', annotation)
            record(f'stats/{hemi}.{atlas}.stats', stats)
            folder = Path('label') / f'{hemi}.{atlas}'
            names = {path.name for root in (reference, candidate) for path in (root / folder).glob('*.label')}
            if not names:
                records[str(folder)] = dict(equal=False, error='No regional label files')
            for name in sorted(names):
                record(str(folder / name), label)
        for field in ('thickness', 'curv', 'sulc', 'area.pial'):
            record(f'fsaverage/{hemi}.{field}.mgh', image)
    return dict(reference=str(reference), candidate=str(candidate),
                comparison='exact decoded payloads; path/time header text is not compared',
                all_payloads_equal=all(item['equal'] for item in records.values()),
                files=len(records), different=[name for name, result in records.items() if not result['equal']],
                records=records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hemi', choices=('left', 'right', 'both'), default='left')
    parser.add_argument('--atlases', default='aparc,aparc.a2009s')
    args = parser.parse_args(argv)
    from postprocessing.core.runtime import selection
    hemis, atlases = selection(args.hemi, args.atlases)
    result = compare(args.reference, args.candidate, hemis, atlases)
    atomic_json(args.output, result)
    print(f"Payloads equal: {result['all_payloads_equal']}; checked: {result['files']}; "
          f"different: {len(result['different'])}; details: {args.output}")
    return 0 if result['all_payloads_equal'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
