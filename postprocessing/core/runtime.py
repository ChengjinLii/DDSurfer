"""Verify a locally compiled runtime without requiring a full FreeSurfer install."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess

from postprocessing.core.config import COMMANDS, FS_REVISION, default_prefix, host
from utils.files import sha256_file

SCHEMA = 'ddsurfer-local-surface-runtime-1'


def selection(hemisphere, atlas_names):
    if hemisphere not in ('left', 'right', 'both'):
        raise ValueError('Hemisphere must be left, right or both')
    hemis = ('lh', 'rh') if hemisphere == 'both' else (('lh',) if hemisphere == 'left' else ('rh',))
    atlases = tuple(dict.fromkeys(name.strip() for name in atlas_names.split(',') if name.strip()))
    if not atlases or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', a) or a in ('.', '..') for a in atlases):
        raise ValueError('Specify valid cortical atlas names')
    return hemis, atlases


def required_files(hemis, atlases):
    files = ['bin/' + name for name in COMMANDS]
    files += ['build-stamp.txt', 'FreeSurferColorLUT.txt', 'LICENSE.txt']
    for hemi in hemis:
        files += [f'average/{hemi}.average.curvature.filled.buckner40.tif',
                  f'subjects/fsaverage/surf/{hemi}.sphere.reg',
                  f'subjects/fsaverage/surf/{hemi}.white',
                  f'subjects/fsaverage/surf/{hemi}.orig',
                  f'subjects/fsaverage/label/{hemi}.cortex.label']
        files += [f'subjects/fsaverage/label/{hemi}.{atlas}.annot' for atlas in atlases]
    return files


def shared_libraries(binary):
    system, _ = host()
    name = 'ldd' if system == 'Linux' else 'otool'
    tool = shutil.which(name)
    if tool is None:
        raise FileNotFoundError(f'Install {name} to check native library dependencies')
    command = [tool, str(binary)] if system == 'Linux' else [tool, '-L', str(binary)]
    env = dict(os.environ)
    for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT', 'DYLD_LIBRARY_PATH', 'DYLD_INSERT_LIBRARIES'):
        env.pop(key, None)
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    if result.returncode or 'not found' in result.stdout or 'not found' in result.stderr:
        raise RuntimeError(f'Unresolved shared libraries for {Path(binary).name}:\n{result.stdout}{result.stderr}')
    return result.stdout.strip()


def validate_runtime(home, hemisphere='both', atlas_names='aparc,aparc.a2009s', check_libraries=True):
    system, machine = host()
    if home is None:
        home = default_prefix()
    root = Path(home).expanduser().resolve()
    record = root / 'runtime.json'
    if not record.is_file():
        raise FileNotFoundError('Local surface components have not been built. Run '
                                'python -m postprocessing.core.build --jobs 4, or select an existing FreeSurfer installation')
    manifest = json.loads(record.read_text())
    if (manifest.get('schema') != SCHEMA or manifest.get('source_revision') != FS_REVISION or
            manifest.get('activation_key_included') is not False):
        raise ValueError('Unsupported surface runtime; rebuild locally from the pinned source')
    if (manifest.get('system'), manifest.get('machine')) != (system, machine):
        raise ValueError('Surface runtime was built for a different platform; rebuild it on this machine')
    hemis, atlases = selection(hemisphere, atlas_names)
    records = manifest.get('files', {})
    if not set(required_files(hemis, atlases)).issubset(records):
        raise ValueError('Local runtime omits required commands or atlas files')
    if (root / 'license.txt').exists() or (root / '.license').exists():
        raise ValueError('Keep the private activation key outside the runtime; set FS_LICENSE instead')
    for name, info in records.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or not relative.parts:
            raise ValueError('Invalid path in runtime manifest')
        path = root.joinpath(*relative.parts)
        parents = [path] + list(path.parents)[:len(relative.parts) - 1]
        if root not in path.resolve().parents or any(parent.is_symlink() for parent in parents):
            raise ValueError('Local runtime must not link to an external installation')
        if not path.is_file() or path.stat().st_size != info['bytes'] or sha256_file(path) != info['sha256']:
            raise ValueError(f'Local runtime checksum mismatch: {name}')
    actual = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()}
    if actual != set(records) | {'runtime.json'}:
        raise ValueError('Local runtime contains unrecorded files')
    for name in COMMANDS:
        path = root / 'bin' / name
        if not os.access(path, os.X_OK):
            raise ValueError(f'Local runtime command is not executable: {name}')
        with path.open('rb') as stream:
            magic = stream.read(4)
        if system == 'Linux' and magic != b'\x7fELF':
            raise ValueError(f'Expected a local ELF executable: {name}')
        if system == 'Darwin' and magic not in (b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf', b'\xca\xfe\xba\xbe'):
            raise ValueError(f'Expected a local Mach-O executable: {name}')
        if check_libraries:
            shared_libraries(path)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--hemi', choices=('left', 'right', 'both'), default='both')
    parser.add_argument('--atlases', default='aparc,aparc.a2009s')
    args = parser.parse_args(argv)
    home = args.runtime if args.runtime is not None else default_prefix()
    manifest = validate_runtime(home, args.hemi, args.atlases)
    print(json.dumps(dict(runtime=str(home), source_revision=manifest['source_revision'],
                          files=len(manifest['files']), system=manifest['system'], machine=manifest['machine']), indent=2))


if __name__ == '__main__':
    main()
