"""Compact, content-checked receipts for completed native-space results."""

import json
from pathlib import Path

from utils.files import sha256_file
from utils.preflight import digest


def result_receipt(args):
    root = args.output_root.resolve() / args.subject
    required = [root / 'logs' / name for name in ('raw_inputs.json', 'dti.json', 'b0ToAtlasT2.tfm')]
    for modality in ('FA', 'MD', 'MinEigenvalue', 'MidEigenvalue', 'MaxEigenvalue', 'Trace'):
        path = root / 'dti' / f'{args.subject}-{modality}.nii.gz'
        if modality != 'Trace' or not args.skip_preprocessing:
            required.append(path)
    for hemi in ('lh', 'rh'):
        for kind in ('white', 'pial') if args.predict_mode == 'all' else ('white',):
            required.append(root / 'ddsurfer' / f'{hemi}.{kind}.obj')
    folders = ['dti', 'ddsurfer']
    if args.freesurfer:
        folders += ['surf', 'label', 'stats', 'fsaverage', 'mri']
        hemis = ('lh', 'rh') if args.postprocess_hemi == 'both' else (('lh',) if args.postprocess_hemi == 'left' else ('rh',))
        atlases = set(a.strip() for a in args.postprocess_atlases.split(',') if a.strip())
        required += [root / 'mri' / name for name in ('brain.mgz', 'orig.mgz')]
        for hemi in hemis:
            required += [root / 'surf' / f'{hemi}.{name}' for name in (
                'white', 'pial', 'curv', 'sulc', 'thickness', 'area', 'inflated', 'sphere', 'sphere.reg')]
            required.append(root / 'label' / f'{hemi}.cortex.label')
            for atlas in atlases:
                required += [root / 'label' / f'{hemi}.{atlas}.annot', root / 'stats' / f'{hemi}.{atlas}.stats']
            required += [root / 'fsaverage' / f'{hemi}.{name}.mgh' for name in ('thickness', 'curv', 'sulc', 'area.pial')]
    paths = set(required)
    for name in folders:
        directory = root / name
        if directory.is_symlink():
            raise ValueError(f'Result directories must not be symlinks: {directory}')
        for path in directory.rglob('*'):
            if path.is_symlink():
                raise ValueError(f'Result files must not be symlinks: {path}')
            if path.is_file():
                paths.add(path)
    for name in ('mask.json', 'qc.json'):
        path = root / 'logs' / name
        if path.exists():
            paths.add(path)
    inventory = {}
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file() or not path.stat().st_size:
            raise ValueError(f'Result missing, empty or symlinked: {path}')
        inventory[str(path.relative_to(root))] = sha256_file(path)
    maps = json.loads((root / 'logs/dti.json').read_text()).get('scalar_maps', {})
    for record in maps.values():
        if inventory.get(record['file']) != record['sha256']:
            raise ValueError('Native DTI map does not match its export record')
    return dict(output_count=len(inventory), output_digest=digest(inventory))


def reusable_result(args, signature):
    if signature is None:
        return False
    path = args.output_root.resolve() / args.subject / 'logs/pipeline.json'
    try:
        saved = json.loads(path.read_text())
        return (saved['subject'] == args.subject and saved['completed'] is True
                and saved['result']['signature'] == signature
                and saved['result'] == dict(signature=signature, **result_receipt(args)))
    except (OSError, ValueError, KeyError, TypeError):
        return False
