"""Checked extraction shared by atlas data and source-dependency preparation."""
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

from utils.files import sha256_file

ASSETS = Path(__file__).resolve().parents[1] / 'assets'


def extract(archive, destination, root_name, max_bytes=1024 ** 3):
    destination = Path(destination)
    seen, total = set(), 0
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or '\\' in member.name or
                    not path.parts or path.parts[0] != root_name or
                    not (member.isfile() or member.isdir())):
                raise ValueError('Unsafe path or file type in source/data archive')
            if len(path.parts) == 1:
                if not member.isdir():
                    raise ValueError('Invalid archive root')
                continue
            relative = PurePosixPath(*path.parts[1:])
            if relative in seen:
                raise ValueError('Duplicate archive member')
            seen.add(relative)
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            total += member.size
            if total > max_bytes:
                raise ValueError('Source/data archive exceeds the extraction limit')
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)
            target.chmod(0o644)


def assets(package=None):
    root = Path(package) if package is not None else ASSETS
    record = root / 'manifest.json'
    manifest = json.loads(record.read_text())
    name = manifest.get('archive', '')
    if (manifest.get('schema') != 'ddsurfer-atlas-1' or Path(name).name != name or
            not name.endswith('.tar.gz') or not re.fullmatch(r'[a-f0-9]{64}', manifest.get('sha256', ''))):
        raise ValueError('Invalid atlas-data manifest')
    archive = root / name
    if archive.stat().st_size != manifest.get('bytes') or sha256_file(archive) != manifest['sha256']:
        raise ValueError('Atlas archive checksum mismatch')
    if any(name.startswith('bin/') or name in ('license.txt', '.license', 'runtime.json') for name in manifest['files']):
        raise ValueError('Atlas package must contain data/notices only, not executables or activation keys')
    return archive, manifest


def install_assets(destination, package=None):
    destination = Path(destination)
    archive, manifest = assets(package)
    extract(archive, destination, 'atlas', max_bytes=256 * 1024 ** 2)
    actual = {str(path.relative_to(destination)) for path in destination.rglob('*') if path.is_file()}
    if actual != set(manifest['files']):
        raise ValueError('Atlas archive and file manifest disagree')
    for name, record in manifest['files'].items():
        path = destination / name
        if path.stat().st_size != record['bytes'] or sha256_file(path) != record['sha256']:
            raise ValueError(f'Atlas file checksum mismatch: {name}')
    return manifest
