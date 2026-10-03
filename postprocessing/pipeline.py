"""Compute native-space curvature, thickness, and cortical parcellations."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from threading import Lock
from time import perf_counter

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np

from inference.coordinates import atomic_json, sha256_file
from postprocessing.geometry import load_pair, prepare_brain, prepare_surface_reference, vertex_area, volume_files, write_surface

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', required=True)
    parser.add_argument('--native-root', type=Path,
                        help='Root containing <subject>/ddsurfer/ with native white/pial files.')
    for hemi in ('lh', 'rh'):
        for kind in ('white', 'pial'):
            parser.add_argument(f'--{hemi}-{kind}', type=Path, help=f'{hemi} {kind} surface in native scanner RAS mm.')
    parser.add_argument('--brain-source', type=Path, help='Optional native scalar MRI; not needed for surface-only processing.')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs', help='FreeSurfer SUBJECTS_DIR.')
    parser.add_argument('--hemi', choices=('left', 'right', 'both'), default='both')
    parser.add_argument('--atlases', default='aparc,aparc.a2009s')
    parser.add_argument('--freesurfer-home', type=Path, default=os.environ.get('FREESURFER_HOME'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--serial', action='store_true', help='Process hemispheres sequentially.')
    return parser.parse_args(argv)


def run(args):
    if Path(args.subject).name != args.subject or args.subject in ('', '.', '..', 'fsaverage'):
        raise ValueError('Invalid subject identifier')
    if args.threads < 1:
        raise ValueError('--threads must be positive')
    if args.freesurfer_home is None:
        raise ValueError('Set FREESURFER_HOME or pass --freesurfer-home')
    home = args.freesurfer_home.resolve()
    root = args.output_root.resolve()
    hemis = ('lh', 'rh') if args.hemi == 'both' else (('lh',) if args.hemi == 'left' else ('rh',))
    surfaces = {}
    for hemi in hemis:
        surfaces[hemi] = {}
        for kind in ('white', 'pial'):
            path = getattr(args, f'{hemi}_{kind}')
            if path is not None and args.native_root is not None:
                raise ValueError('Choose explicit surface files or --native-root, not both')
            if path is None and args.native_root is not None:
                name = 'wm' if kind == 'white' else kind
                directory = args.native_root / args.subject
                stems = [directory / 'ddsurfer' / f'{hemi}.{kind}', directory / f'{hemi}.{kind}',
                         directory / f'{args.subject}_predicted_{name}_{hemi}']
                candidates = [Path(f'{stem}{suffix}') for stem in stems for suffix in ('.obj', '.stl', '.ply', '.off', '')]
                path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if path is None:
                raise ValueError(f'Provide --{hemi}-{kind}')
            surfaces[hemi][kind] = path.resolve()
    atlases = tuple(dict.fromkeys(name.strip() for name in args.atlases.split(',') if name.strip()))
    if not atlases or any(Path(a).name != a or a in ('.', '..') for a in atlases):
        raise ValueError('Specify valid cortical atlas names')
    modern_metrics = os.access(str(home / 'bin/mris_place_surface'), os.X_OK)
    metric_binary = 'mris_place_surface' if modern_metrics else 'mris_thickness'
    binaries = ('mris_curvature', 'mris_inflate', 'mris_sphere', 'mris_register', metric_binary,
                'mri_label2label', 'mri_surf2surf', 'mri_annotation2label')
    for name in binaries:
        if not os.access(str(home / 'bin' / name), os.X_OK):
            raise FileNotFoundError(f'FreeSurfer executable not available: {name}')
    license_file = Path(os.environ.get('FS_LICENSE', str(home / 'license.txt')))
    if not license_file.is_file():
        raise FileNotFoundError('Set FS_LICENSE to a valid FreeSurfer license file')
    fsaverage = home / 'subjects/fsaverage'
    for hemi in hemis:
        required = [fsaverage / 'label' / f'{hemi}.cortex.label',
                    home / 'average' / f'{hemi}.average.curvature.filled.buckner40.tif']
        required += [fsaverage / 'label' / f'{hemi}.{atlas}.annot' for atlas in atlases]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)

    files = volume_files(args.brain_source) if args.brain_source is not None else []
    for hemi in hemis:
        files += list(surfaces[hemi].values())
    signature = dict(inputs={str(path): sha256_file(path) for path in files},
                     hemis=list(hemis), atlases=list(atlases), freesurfer_home=str(home),
                     metrics=metric_binary, threads=args.threads, serial=args.serial,
                     reference_orientation='LIA',
                     reference_content='native_MRI' if args.brain_source is not None else 'geometry_only')
    subject_dir = root / args.subject
    signature_path = subject_dir / 'logs/inputs.json'
    if any((subject_dir / name).exists() for name in ('mri', 'surf', 'label', 'stats', 'fsaverage')) and not signature_path.exists():
        raise ValueError('Existing subject output has no input record; use a new output directory')
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError('Existing FreeSurfer outputs have different inputs/settings; use a new output directory')
    for name in ('mri', 'surf', 'label', 'stats', 'logs', 'fsaverage'):
        (subject_dir / name).mkdir(parents=True, exist_ok=True)
    atomic_json(signature_path, signature)
    average_link = root / 'fsaverage'
    if average_link.exists() or average_link.is_symlink():
        if average_link.resolve() != fsaverage.resolve():
            raise ValueError('SUBJECTS_DIR/fsaverage points to an unexpected template')
    else:
        average_link.symlink_to(fsaverage, target_is_directory=True)
    workers = min(len(hemis), args.threads, 1 if args.serial else 2)
    per_hemi_threads = max(1, args.threads // workers)
    env = dict(os.environ, FREESURFER_HOME=str(home), SUBJECTS_DIR=str(root), FS_LICENSE=str(license_file.resolve()),
               PATH=str(home / 'bin') + os.pathsep + os.environ.get('PATH', ''),
               OMP_NUM_THREADS=str(per_hemi_threads), ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(per_hemi_threads),
               OPENBLAS_NUM_THREADS='1')
    log_path = subject_dir / 'logs/postprocess.log'
    completed_path = subject_dir / 'logs/completed.json'
    done = json.loads(completed_path.read_text()) if completed_path.exists() else {}
    state_lock = Lock()
    timings_path = subject_dir / 'logs/timings.json'
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    qc = dict(coordinate_space='native_scanner_RAS_mm', surface_storage='FreeSurfer_surface_RAS_mm',
              reference_content=signature['reference_content'], hemispheres={})

    def execute(key, command, outputs):
        outputs = [Path(path) for path in outputs]
        if key in done and all(path.is_file() and sha256_file(path) == done[key].get(str(path)) for path in outputs):
            print(f'[skip] {key}', flush=True)
            return
        stage_log = subject_dir / 'logs' / f'{key.split(".")[0]}.log'
        with stage_log.open('a') as log:
            message = f'[run] {key}: {shlex.join([str(v) for v in command])}'
            print(message, flush=True)
            log.write(message + '\n'); log.flush()
            with state_lock, log_path.open('a') as summary:
                summary.write(message + '\n')
            started = perf_counter()
            try:
                subprocess.run([str(v) for v in command], check=True, env=env, cwd=subject_dir, stdout=log, stderr=log)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f'{key} failed (exit {error.returncode}); see {stage_log}') from error
        if not all(path.is_file() and path.stat().st_size > 0 for path in outputs):
            raise RuntimeError(f'{key} returned without the expected outputs; see {stage_log}')
        with state_lock:
            done[key] = {str(path): sha256_file(path) for path in outputs}
            atomic_json(completed_path, done)
            timings[key] = dict(seconds=perf_counter() - started)
            atomic_json(timings_path, timings)

    brain_path = subject_dir / 'mri/brain.mgz'
    pairs = {hemi: load_pair(surfaces[hemi]['white'], surfaces[hemi]['pial']) for hemi in hemis}
    image = (prepare_brain(args.brain_source, brain_path) if args.brain_source is not None
             else prepare_surface_reference([mesh for pair in pairs.values() for mesh in pair], brain_path))
    shutil.copyfile(brain_path, subject_dir / 'mri/orig.mgz')
    surf = subject_dir / 'surf'
    label = subject_dir / 'label'
    def process_hemi(hemi):
        white, pial = pairs[hemi]
        hemi_qc = {}
        for name, mesh in (('white', white), ('pial', pial)):
            path = surf / f'{hemi}.{name}'
            hemi_qc[name] = write_surface(mesh, image, path)
            area_path = surf / (f'{hemi}.area' + ('.pial' if name == 'pial' else ''))
            if modern_metrics:
                execute(f'{hemi}.{name}.area', ['mris_place_surface', '--area-map', path, area_path], [area_path])
            else:
                nib.freesurfer.write_morph_data(str(area_path), vertex_area(mesh.vertices, mesh.faces), fnum=len(mesh.faces))
            execute(f'{hemi}.{name}.curvature', ['mris_curvature', '-w', path],
                    [surf / f'{hemi}.{name}.H', surf / f'{hemi}.{name}.K'])
        shutil.copyfile(surf / f'{hemi}.white', surf / f'{hemi}.orig')
        shutil.copyfile(surf / f'{hemi}.white', surf / f'{hemi}.smoothwm')
        if modern_metrics:
            for name in ('white', 'pial'):
                output = surf / (f'{hemi}.curv' + ('.pial' if name == 'pial' else ''))
                execute(f'{hemi}.{name}.curv', ['mris_place_surface', '--curv-map', surf / f'{hemi}.{name}',
                        '2', '10', output], [output])
        else:
            shutil.copyfile(surf / f'{hemi}.white.H', surf / f'{hemi}.curv')
            shutil.copyfile(surf / f'{hemi}.pial.H', surf / f'{hemi}.curv.pial')
        execute(f'{hemi}.inflate', ['mris_inflate', surf / f'{hemi}.white', surf / f'{hemi}.inflated'],
                [surf / f'{hemi}.inflated', surf / f'{hemi}.sulc'])
        execute(f'{hemi}.inflated.curvature', ['mris_curvature', '-w', surf / f'{hemi}.inflated'],
                [surf / f'{hemi}.inflated.H', surf / f'{hemi}.inflated.K'])
        execute(f'{hemi}.sphere', ['mris_sphere', surf / f'{hemi}.inflated', surf / f'{hemi}.sphere'],
                [surf / f'{hemi}.sphere'])
        execute(f'{hemi}.register', ['mris_register', '-curv', surf / f'{hemi}.sphere',
                home / 'average' / f'{hemi}.average.curvature.filled.buckner40.tif', surf / f'{hemi}.sphere.reg'],
                [surf / f'{hemi}.sphere.reg'])
        if modern_metrics:
            thickness_command = ['mris_place_surface', '--thickness', surf / f'{hemi}.white',
                                 surf / f'{hemi}.pial', '20', '5', surf / f'{hemi}.thickness']
        else:
            thickness_command = ['mris_thickness', args.subject, hemi, 'thickness']
        execute(f'{hemi}.thickness', thickness_command, [surf / f'{hemi}.thickness'])
        execute(f'{hemi}.cortex', ['mri_label2label', '--srclabel', fsaverage / 'label' / f'{hemi}.cortex.label',
                '--srcsubject', 'fsaverage', '--trgsubject', args.subject, '--trglabel', label / f'{hemi}.cortex.label',
                '--regmethod', 'surface', '--hemi', hemi], [label / f'{hemi}.cortex.label'])
        for atlas in atlases:
            annot = label / f'{hemi}.{atlas}.annot'
            execute(f'{hemi}.{atlas}.annot', ['mri_surf2surf', '--srcsubject', 'fsaverage', '--trgsubject', args.subject,
                    '--hemi', hemi, '--sval-annot', fsaverage / 'label' / f'{hemi}.{atlas}.annot', '--tval', annot], [annot])
            regions = label / f'{hemi}.{atlas}'
            regions.mkdir(exist_ok=True)
            indices, _, names = nib.freesurfer.read_annot(str(annot))
            present = set(np.unique(indices))
            expected = [regions / f'{hemi}.{name.decode()}.label' for index, name in enumerate(names)
                        if index in present and name.decode().lower() not in ('unknown', '???', 'medial_wall')]
            if not expected:
                raise ValueError(f'{hemi}.{atlas} annotation contains no cortical regions')
            execute(f'{hemi}.{atlas}.labels', ['mri_annotation2label', '--subject', args.subject, '--hemi', hemi,
                    '--annotation', atlas, '--outdir', regions], expected)
            stats = subject_dir / 'stats' / f'{hemi}.{atlas}.stats'
            execute(f'{hemi}.{atlas}.stats', [sys.executable, Path(__file__).resolve().with_name('stats.py'),
                    '--subject-dir', subject_dir, '--hemi', hemi, '--atlas', atlas], [stats])
        for metric in ('thickness', 'curv', 'sulc', 'area.pial'):
            output = subject_dir / 'fsaverage' / f'{hemi}.{metric}.mgh'
            execute(f'{hemi}.{metric}.fsaverage', ['mri_surf2surf', '--srcsubject', args.subject, '--trgsubject',
                    'fsaverage', '--hemi', hemi, '--sval', surf / f'{hemi}.{metric}', '--tval', output], [output])
        thickness = nib.freesurfer.read_morph_data(str(surf / f'{hemi}.thickness'))
        if len(thickness) != len(white.vertices) or not np.isfinite(thickness).all():
            raise ValueError('Invalid thickness output')
        qc['hemispheres'][hemi] = hemi_qc
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(process_hemi, hemis))
    atomic_json(subject_dir / 'logs/qc.json', qc)
    print(f'FreeSurfer postprocessing completed: {subject_dir}', flush=True)


if __name__ == '__main__':
    run(parse_args())
