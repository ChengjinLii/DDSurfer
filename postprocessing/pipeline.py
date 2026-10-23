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
import tempfile
from threading import Lock
from time import perf_counter

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np

from utils.files import atomic_json, sha256_file
from utils.stages import StageCache, subject_lock
from postprocessing.geometry import load_pair, prepare_brain, prepare_surface_reference, vertex_area, volume_files, write_surface
from postprocessing.environment import check_environment, configured_home
from postprocessing.labels import read_regions, write_labels
from postprocessing.overlays import FIELDS, copy_frames, write_stack
from postprocessing.stats import read_measurements, write_stats

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
    parser.add_argument('--backend', choices=('auto', 'native', 'freesurfer'), default='auto',
                        help='Use a local source build or an existing FreeSurfer installation.')
    parser.add_argument('--runtime', '--core-runtime', dest='core_runtime', type=Path,
                        help='Use a locally compiled surface runtime instead of a full installation.')
    parser.add_argument('--license-file', type=Path, help='Private activation file; otherwise use FS_LICENSE.')
    parser.add_argument('--sphere-seed', type=int, help='Optional positive seed for reproducible sphere comparisons.')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--serial', action='store_true', help='Process hemispheres sequentially.')
    return parser.parse_args(argv)


def run(args):
    if Path(args.subject).name != args.subject or args.subject in ('', '.', '..', 'fsaverage'):
        raise ValueError('Invalid subject identifier')
    with subject_lock(args.output_root / args.subject / 'logs/postprocess.lock'):
        return run_locked(args)


def run_locked(args):
    if Path(args.subject).name != args.subject or args.subject in ('', '.', '..', 'fsaverage'):
        raise ValueError('Invalid subject identifier')
    if args.threads < 1:
        raise ValueError('--threads must be positive')
    sphere_seed = getattr(args, 'sphere_seed', None)
    if sphere_seed is not None and not 0 < sphere_seed < 2 ** 31:
        raise ValueError('--sphere-seed must be a positive 32-bit integer')
    runtime = getattr(args, 'core_runtime', None)
    if runtime is not None and not (runtime / 'runtime.json').is_file():
        raise ValueError('--runtime requires a local build with a runtime.json manifest')
    environment = check_environment(configured_home(args), args.hemi, args.atlases,
                                    license_file=getattr(args, 'license_file', None), backend=args.backend)
    home = environment['home']
    root = args.output_root.resolve()
    hemis = environment['hemis']
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
    atlases = environment['atlases']
    modern_metrics = environment['modern_metrics']
    metric_binary = environment['metric_binary']
    license_file = environment['license_file']
    fsaverage = environment['fsaverage']

    files = volume_files(args.brain_source) if args.brain_source is not None else []
    for hemi in hemis:
        files += list(surfaces[hemi].values())
    signature = dict(inputs={str(path): sha256_file(path) for path in files},
                     hemis=list(hemis), atlases=list(atlases), freesurfer_home=str(home),
                     metrics=metric_binary, threads=args.threads, serial=args.serial,
                     reference_orientation='LIA',
                     reference_content='native_MRI' if args.brain_source is not None else 'geometry_only')
    if sphere_seed is not None:
        signature['sphere_seed'] = sphere_seed
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
        try:
            average_link.symlink_to(fsaverage, target_is_directory=True)
        except FileExistsError:
            if average_link.resolve() != fsaverage.resolve():
                raise ValueError('SUBJECTS_DIR/fsaverage points to an unexpected template')
    workers = min(len(hemis), args.threads, 1 if args.serial else 2)
    per_hemi_threads = max(1, args.threads // workers)
    env = dict(os.environ, FREESURFER_HOME=str(home), SUBJECTS_DIR=str(root), FS_LICENSE=str(license_file.resolve()),
               PATH=str(home / 'bin') + os.pathsep + os.environ.get('PATH', ''),
               OMP_NUM_THREADS=str(per_hemi_threads), ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(per_hemi_threads),
               OPENBLAS_NUM_THREADS='1')
    if environment['minimal_runtime']:
        # Avoid private libraries and commands from an unrelated full installation.
        for name in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT'):
            env.pop(name, None)
        env['PATH'] = str(home / 'bin') + os.pathsep + os.defpath
    log_path = subject_dir / 'logs/postprocess.log'
    completed_path = subject_dir / 'logs/completed.json'
    done = json.loads(completed_path.read_text()) if completed_path.exists() else {}
    dependencies_path = subject_dir / 'logs/stage_inputs.json'
    dependencies = json.loads(dependencies_path.read_text()) if dependencies_path.exists() else {}
    preparation = StageCache(subject_dir / 'logs/preparation.json')
    state_lock = Lock()
    timings_path = subject_dir / 'logs/timings.json'
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    qc = dict(coordinate_space='native_scanner_RAS_mm', surface_storage='FreeSurfer_surface_RAS_mm',
              reference_content=signature['reference_content'], hemispheres={})

    def run_command(key, command):
        command = list(command)
        if environment['minimal_runtime'] and command[0] != sys.executable:
            command[0] = home / 'bin' / command[0]
        stage_log = subject_dir / 'logs' / f'{key.split(".")[0]}.log'
        with stage_log.open('a') as log:
            message = f'[run] {key}: {shlex.join([str(v) for v in command])}'
            print(message, flush=True)
            log.write(message + '\n'); log.flush()
            with state_lock, log_path.open('a') as summary:
                summary.write(message + '\n')
            try:
                subprocess.run([str(v) for v in command], check=True, env=env, cwd=subject_dir, stdout=log, stderr=log)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f'{key} failed (exit {error.returncode}); see {stage_log}') from error

    def execute(key, command, outputs, inputs):
        outputs = [Path(path) for path in outputs]
        executable = Path(sys.executable) if command[0] == sys.executable else home / 'bin' / command[0]
        implementation = [executable, Path(__file__)]
        if command[0] == sys.executable:
            implementation.append(Path(command[1]))
        signature = dict(command=[str(v) for v in command],
                         inputs=preparation.fingerprints(list(inputs) + implementation))
        if (dependencies.get(key) == signature and key in done and
                all(path.is_file() and preparation.digest(path) == done[key].get(str(path)) for path in outputs)):
            print(f'[skip] {key}', flush=True)
            return
        with state_lock:
            done.pop(key, None)
            dependencies.pop(key, None)
            atomic_json(completed_path, done)
            atomic_json(dependencies_path, dependencies)
        before = {path: (path.stat().st_ino, path.stat().st_mtime_ns, path.stat().st_ctime_ns)
                  for path in outputs if path.is_file()}
        started = perf_counter()
        run_command(key, command)
        stage_log = subject_dir / 'logs' / f'{key.split(".")[0]}.log'
        if not all(path.is_file() and path.stat().st_size > 0 for path in outputs):
            raise RuntimeError(f'{key} returned without the expected outputs; see {stage_log}')
        if any((path.stat().st_ino, path.stat().st_mtime_ns, path.stat().st_ctime_ns) == before.get(path)
               for path in outputs):
            raise RuntimeError(f'{key} did not refresh the expected outputs; see {stage_log}')
        with state_lock:
            done[key] = {str(path): sha256_file(path) for path in outputs}
            dependencies[key] = signature
            atomic_json(completed_path, done)
            atomic_json(dependencies_path, dependencies)
            timings[key] = dict(seconds=perf_counter() - started)
            atomic_json(timings_path, timings)

    def python_stage(key, inputs, outputs, action):
        changed = preparation.run(key, list(inputs) + [Path(__file__)], outputs, action)
        if changed:
            with state_lock:
                done.pop(key, None)
                dependencies.pop(key, None)
                timings[key] = dict(seconds=preparation.records[key]['seconds'])
                atomic_json(completed_path, done)
                atomic_json(dependencies_path, dependencies)
                atomic_json(timings_path, timings)

    brain_path = subject_dir / 'mri/brain.mgz'
    surf = subject_dir / 'surf'
    label = subject_dir / 'label'
    prepared_qc = subject_dir / 'logs/prepared_qc.json'
    prepared_outputs = [brain_path, subject_dir / 'mri/orig.mgz', prepared_qc]
    prepared_outputs += [surf / f'{hemi}.{kind}' for hemi in hemis for kind in ('white', 'pial', 'orig', 'smoothwm')]

    def prepare(staged):
        targets = dict(zip(prepared_outputs, staged))
        pairs = {hemi: load_pair(surfaces[hemi]['white'], surfaces[hemi]['pial']) for hemi in hemis}
        image = (prepare_brain(args.brain_source, targets[brain_path]) if args.brain_source is not None
                 else prepare_surface_reference([mesh for pair in pairs.values() for mesh in pair], targets[brain_path]))
        shutil.copyfile(targets[brain_path], targets[subject_dir / 'mri/orig.mgz'])
        checks = {}
        for hemi, pair in pairs.items():
            checks[hemi] = {kind: write_surface(mesh, image, targets[surf / f'{hemi}.{kind}'])
                            for kind, mesh in zip(('white', 'pial'), pair)}
            for kind in ('orig', 'smoothwm'):
                shutil.copyfile(targets[surf / f'{hemi}.white'], targets[surf / f'{hemi}.{kind}'])
        atomic_json(targets[prepared_qc], checks)

    preparation.run('native_surfaces', files + [Path(__file__), ROOT / 'postprocessing/geometry.py',
                    ROOT / 'utils/surface_io.py', ROOT / 'utils/stages.py'], prepared_outputs, prepare, settings=signature)
    prepared = json.loads(prepared_qc.read_text())

    def process_hemi(hemi):
        hemi_qc = prepared[hemi]
        fallback_pair = []
        white_vertices = None
        measurements = None

        def write_fallback_area(staged, name):
            if not fallback_pair:
                fallback_pair.extend(load_pair(surfaces[hemi]['white'], surfaces[hemi]['pial']))
            mesh = fallback_pair[0 if name == 'white' else 1]
            nib.freesurfer.write_morph_data(str(staged[0]), vertex_area(mesh.vertices, mesh.faces), fnum=len(mesh.faces))

        for name in ('white', 'pial'):
            path = surf / f'{hemi}.{name}'
            area_path = surf / (f'{hemi}.area' + ('.pial' if name == 'pial' else ''))
            if modern_metrics:
                execute(f'{hemi}.{name}.area', ['mris_place_surface', '--area-map', path, area_path], [area_path], [path])
            else:
                preparation.run(f'{hemi}.{name}.area', [surfaces[hemi][name], ROOT / 'postprocessing/geometry.py'],
                                [area_path], lambda staged: write_fallback_area(staged, name))
            execute(f'{hemi}.{name}.curvature', ['mris_curvature', '-w', path],
                    [surf / f'{hemi}.{name}.H', surf / f'{hemi}.{name}.K'], [path])
        fallback_pair.clear()
        if modern_metrics:
            for name in ('white', 'pial'):
                output = surf / (f'{hemi}.curv' + ('.pial' if name == 'pial' else ''))
                execute(f'{hemi}.{name}.curv', ['mris_place_surface', '--curv-map', surf / f'{hemi}.{name}',
                        '2', '10', output], [output], [surf / f'{hemi}.{name}'])
        else:
            for name, suffix in (('white', ''), ('pial', '.pial')):
                source = surf / f'{hemi}.{name}.H'
                python_stage(f'{hemi}.{name}.curv', [source], [surf / f'{hemi}.curv{suffix}'],
                             lambda staged: shutil.copyfile(source, staged[0]))
        execute(f'{hemi}.inflate', ['mris_inflate', surf / f'{hemi}.white', surf / f'{hemi}.inflated'],
                [surf / f'{hemi}.inflated', surf / f'{hemi}.sulc'], [surf / f'{hemi}.white'])
        execute(f'{hemi}.inflated.curvature', ['mris_curvature', '-w', surf / f'{hemi}.inflated'],
                [surf / f'{hemi}.inflated.H', surf / f'{hemi}.inflated.K'], [surf / f'{hemi}.inflated'])
        sphere_command = ['mris_sphere']
        if sphere_seed is not None:
            sphere_command += ['-seed', str(sphere_seed)]
        execute(f'{hemi}.sphere', sphere_command + [surf / f'{hemi}.inflated', surf / f'{hemi}.sphere'],
                [surf / f'{hemi}.sphere'], [surf / f'{hemi}.inflated'])
        execute(f'{hemi}.register', ['mris_register', '-curv', surf / f'{hemi}.sphere',
                home / 'average' / f'{hemi}.average.curvature.filled.buckner40.tif', surf / f'{hemi}.sphere.reg'],
                [surf / f'{hemi}.sphere.reg'], [surf / f'{hemi}.sphere', surf / f'{hemi}.sulc',
                surf / f'{hemi}.smoothwm', surf / f'{hemi}.inflated.H', surf / f'{hemi}.curv',
                home / 'average' / f'{hemi}.average.curvature.filled.buckner40.tif'])
        if modern_metrics:
            thickness_command = ['mris_place_surface', '--thickness', surf / f'{hemi}.white',
                                 surf / f'{hemi}.pial', '20', '5', surf / f'{hemi}.thickness']
        else:
            thickness_command = ['mris_thickness', args.subject, hemi, 'thickness']
        execute(f'{hemi}.thickness', thickness_command, [surf / f'{hemi}.thickness'],
                [surf / f'{hemi}.white', surf / f'{hemi}.pial'])
        execute(f'{hemi}.cortex', ['mri_label2label', '--srclabel', fsaverage / 'label' / f'{hemi}.cortex.label',
                '--srcsubject', 'fsaverage', '--trgsubject', args.subject, '--trglabel', label / f'{hemi}.cortex.label',
                '--regmethod', 'surface', '--hemi', hemi], [label / f'{hemi}.cortex.label'],
                [fsaverage / 'label' / f'{hemi}.cortex.label', fsaverage / 'surf' / f'{hemi}.sphere.reg',
                 surf / f'{hemi}.sphere.reg', surf / f'{hemi}.white'])

        # Share immutable inputs across atlases only when their stages need rebuilding.
        def export_labels(staged, vertex_count, regions):
            nonlocal white_vertices
            if white_vertices is None:
                white_vertices, _ = nib.freesurfer.read_geometry(str(surf / f'{hemi}.white'))
            write_labels(surf / f'{hemi}.white', args.subject, vertex_count, regions, staged, vertices=white_vertices)

        def export_stats(staged, atlas):
            nonlocal measurements
            if measurements is None:
                measurements = read_measurements(subject_dir, hemi)
            write_stats(subject_dir, hemi, atlas, output=staged[0], measurements=measurements)

        for atlas in atlases:
            annot = label / f'{hemi}.{atlas}.annot'
            execute(f'{hemi}.{atlas}.annot', ['mri_surf2surf', '--srcsubject', 'fsaverage', '--trgsubject', args.subject,
                    '--hemi', hemi, '--sval-annot', fsaverage / 'label' / f'{hemi}.{atlas}.annot', '--tval', annot], [annot],
                    [fsaverage / 'label' / f'{hemi}.{atlas}.annot', fsaverage / 'surf' / f'{hemi}.sphere.reg',
                     surf / f'{hemi}.sphere.reg'])
            regions_dir = label / f'{hemi}.{atlas}'
            vertex_count, regions = read_regions(annot)
            if not any(name.lower() not in ('unknown', '???', 'medial_wall') for name, _ in regions):
                raise ValueError(f'{hemi}.{atlas} annotation contains no cortical regions')
            expected = [regions_dir / f'{hemi}.{name}.label' for name, _ in regions]
            python_stage(f'{hemi}.{atlas}.labels', [annot, surf / f'{hemi}.white',
                         ROOT / 'postprocessing/labels.py'], expected,
                         lambda staged: export_labels(staged, vertex_count, regions))
            stats = subject_dir / 'stats' / f'{hemi}.{atlas}.stats'
            python_stage(f'{hemi}.{atlas}.stats',
                         [annot, label / f'{hemi}.cortex.label', surf / f'{hemi}.area', surf / f'{hemi}.area.pial',
                          surf / f'{hemi}.thickness', surf / f'{hemi}.curv', surf / f'{hemi}.white.K',
                          ROOT / 'postprocessing/stats.py'], [stats],
                         lambda staged: export_stats(staged, atlas))

        def transfer_overlays(staged):
            with tempfile.TemporaryDirectory(prefix=f'.{hemi}-overlays-', dir=subject_dir / 'logs') as temporary:
                stack, mapped = Path(temporary) / 'source.mgh', Path(temporary) / 'mapped.mgh'
                write_stack(surf, hemi, stack, hemi_qc['white']['vertices'],
                            fields=measurements[1] if measurements is not None else None)
                run_command(f'{hemi}.fsaverage', ['mri_surf2surf', '--srcsubject', args.subject, '--trgsubject',
                            'fsaverage', '--hemi', hemi, '--sval', stack, '--tval', mapped, '--split'])
                copy_frames(mapped, staged)

        python_stage(f'{hemi}.fsaverage',
                     [surf / f'{hemi}.{field}' for field in FIELDS] +
                     [surf / f'{hemi}.sphere.reg', fsaverage / 'surf' / f'{hemi}.sphere.reg',
                      home / 'bin/mri_surf2surf', ROOT / 'postprocessing/overlays.py'],
                     [subject_dir / 'fsaverage' / f'{hemi}.{field}.mgh' for field in FIELDS], transfer_overlays)
        thickness = (measurements[1]['thickness'] if measurements is not None
                     else nib.freesurfer.read_morph_data(str(surf / f'{hemi}.thickness')))
        if len(thickness) != hemi_qc['white']['vertices'] or not np.isfinite(thickness).all():
            raise ValueError('Invalid thickness output')
        qc['hemispheres'][hemi] = hemi_qc
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(process_hemi, hemis))
    atomic_json(subject_dir / 'logs/qc.json', qc)
    print(f'FreeSurfer postprocessing completed: {subject_dir}', flush=True)


if __name__ == '__main__':
    run(parse_args())
