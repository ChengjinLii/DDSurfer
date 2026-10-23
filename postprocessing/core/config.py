"""Pinned source revisions and locations for locally compiled surface tools."""
import hashlib
import json
import os
from pathlib import Path
import platform


ROOT = Path(__file__).resolve().parent
FS_REVISION = '7eb846079b0dc0c92e8313205a3d2387b5c7a354'
FS_URL = 'https://github.com/freesurfer/freesurfer.git'
COMMANDS = ('mris_curvature', 'mris_inflate', 'mris_sphere', 'mris_register',
            'mris_place_surface', 'mri_label2label', 'mri_surf2surf')
SOURCE_DIRECTORIES = ('include', 'utils', 'packages', 'cmake', 'mris_curvature', 'mris_inflate',
                      'mris_sphere', 'mris_register', 'mris_make_surfaces', 'mri_label2label', 'mri_surf2surf', 'resurf/Code')
DEPENDENCIES = {
    'itk': dict(filename='itk-4.13.2.tar.gz', directory='ITK-4.13.2',
                url='https://codeload.github.com/InsightSoftwareConsortium/ITK/tar.gz/refs/tags/v4.13.2',
                sha256='98c2fd826e1987d797521d83031fcaa328135daf6524f7823363d66ab288c545'),
    'zlib': dict(filename='zlib-1.3.1.tar.gz', directory='zlib-1.3.1',
                 url='https://zlib.net/fossils/zlib-1.3.1.tar.gz',
                 sha256='9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23'),
}


def host():
    system = platform.system()
    if system not in ('Linux', 'Darwin'):
        raise ValueError('Native Windows builds are not supported; use WSL2 or a supported FreeSurfer installation')
    machine = platform.machine().lower()
    machine = {'amd64': 'x86_64', 'aarch64': 'arm64'}.get(machine, machine)
    return system, machine


def cache_root():
    return Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache').expanduser() / 'ddsurfer/postprocessing'


def recipe_digest():
    digest = hashlib.sha256(json.dumps(dict(source=FS_REVISION, dependencies=DEPENDENCIES), sort_keys=True).encode())
    files = [ROOT / name for name in ('build.py', 'runtime.py', 'config.py', 'archive.py')]
    files += sorted(path for path in (ROOT / 'cmake').glob('*') if path.is_file())
    files += [ROOT.parent / 'assets/manifest.json']
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def default_prefix():
    system, machine = host()
    return cache_root() / 'native' / f'{system.lower()}-{machine}-{recipe_digest()[:16]}'
