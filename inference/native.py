"""Transform scanner-RAS atlas meshes to native RAS using the registration pull map."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import SimpleITK as sitk

from utils.files import atomic_json, sha256_file
from utils.surface_io import load_mesh, write_obj


def atlas_ras_to_native_ras(vertices, transform):
    # BRAINSFit: fixed=atlas, moving=b0. The saved ITK map is atlas LPS -> native LPS.
    # Both mesh interfaces use RAS, so flip x/y before AND after TransformPoint.
    flip = np.array([-1., -1., 1.])
    native_lps = np.asarray([transform.TransformPoint(tuple(v)) for v in np.asarray(vertices)*flip])
    return native_lps*flip


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', required=True)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--pred-root', required=True, type=Path)
    parser.add_argument('--predict-mode', choices=('wm','all'), default='all')
    parser.add_argument('--output-dir', type=Path, help='Final native surface directory.')
    args = parser.parse_args(argv)
    transform_path = args.data_root/args.subject/f'{args.subject}-b0ToAtlasT2.tfm'
    transform = sitk.ReadTransform(str(transform_path))
    directory = args.output_dir if args.output_dir is not None else args.pred_root/'native'/args.subject
    directory.mkdir(parents=True,exist_ok=True)
    if args.output_dir is not None:
        logs = directory.parent / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        saved_transform = logs / 'b0ToAtlasT2.tfm'
        shutil.copyfile(transform_path, saved_transform)
    else:
        saved_transform = transform_path
    for hemi,short in [('left','lh'),('right','rh')]:
        for surface in (['wm','pial'] if args.predict_mode=='all' else ['wm']):
            source=args.pred_root/'mni'/args.subject/f'{args.subject}_predicted_{surface}_surface_{hemi}.obj'
            mesh=load_mesh(source)
            vertices=atlas_ras_to_native_ras(mesh.vertices,transform)
            if not np.isfinite(vertices).all():raise ValueError('Nonfinite native vertices')
            stem = f'{short}.{"white" if surface == "wm" else "pial"}'
            output=directory/(f'{stem}.obj' if args.output_dir is not None else f'{args.subject}_predicted_{surface}_{short}.obj')
            error = write_obj(output, vertices, mesh.faces)
            metadata_path = logs / f'{stem}.json' if args.output_dir is not None else output.with_suffix('.json')
            atomic_json(metadata_path,dict(coordinate_space='native_scanner_RAS_mm',
                        transform=str(saved_transform),transform_sha256=sha256_file(saved_transform),
                        transform_direction='atlas_LPS_to_native_LPS_pull_map',source_sha256=sha256_file(source),
                        output_sha256=sha256_file(output), max_export_error_mm=error))
            print(f'Saved {output} (native RAS mm)',flush=True)


if __name__=='__main__':
    main()
