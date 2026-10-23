"""Compile only DDSurfer's required surface commands from pinned upstream source."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

from postprocessing.core.archive import extract, install_assets
from postprocessing.core.config import (COMMANDS, DEPENDENCIES, FS_REVISION, FS_URL, ROOT,
                                       SOURCE_DIRECTORIES, cache_root, default_prefix, host, recipe_digest)
from postprocessing.core.runtime import SCHEMA, validate_runtime
from utils.files import atomic_json, sha256_file


def require_tool(value, name):
    candidate = shutil.which(os.path.expanduser(str(value)) if value else name)
    if candidate is None or not Path(candidate).is_file() or not os.access(candidate, os.X_OK):
        raise FileNotFoundError(f'Missing build tool: {name}. See postprocessing/core/README.md for installation commands')
    return str(Path(candidate).resolve())


def run(command, log, env):
    print('[build] ' + ' '.join(str(value) for value in command), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        subprocess.run([str(value) for value in command], check=True, env=env, stdout=stream, stderr=stream)


def download(record, folder, offline=False):
    path = folder / record['filename']
    if path.is_file():
        if sha256_file(path) != record['sha256']:
            raise ValueError(f'Source archive checksum mismatch: {path.name}; replace the damaged download')
        return path
    if offline:
        raise FileNotFoundError(f'Offline build needs the checked source archive: {path}')
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='.download-', dir=folder, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            print('[download] ' + record['url'], flush=True)
            with urllib.request.urlopen(record['url'], timeout=120) as source:
                shutil.copyfileobj(source, stream)
            stream.close()
            if sha256_file(temporary) != record['sha256']:
                raise ValueError('Downloaded source checksum mismatch: ' + path.name)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def prepare_dependency(name, work, offline=False):
    record = DEPENDENCIES[name]
    archive = download(record, work / 'downloads', offline)
    destination = work / 'deps' / record['directory']
    marker = destination / '.ddsurfer-source.json'
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.source-', dir=destination.parent) as temporary:
            staged = Path(temporary) / 'tree'
            staged.mkdir()
            extract(archive, staged, record['directory'])
            atomic_json(staged / '.ddsurfer-source.json', record)
            staged.rename(destination)
    elif not marker.is_file() or json.loads(marker.read_text()) != record:
        raise ValueError(f'Unrecognized dependency source directory: {destination}; choose a fresh --work-dir')
    if name == 'itk':
        compiler = destination / 'Modules/ThirdParty/VNL/src/vxl/vcl/vcl_compiler.h'
        old = '# elif (__GNUC__==8)'
        new = '# elif (__GNUC__>=8) // DDSurfer: accept newer GCC with the GCC 8 compatibility definitions.'
        text = compiler.read_text()
        if text.count(old) == 1:
            compiler.write_text(text.replace(old, new))
        elif new not in text:
            raise ValueError('Unrecognized ITK compiler header; refusing an unchecked compatibility patch')
    return destination


def prepare_source(work, selected, git, offline, env):
    source = selected.resolve() if selected is not None else work / 'source'
    log = work / 'source.log'
    if not (source / '.git').is_dir():
        if selected is not None or offline:
            raise FileNotFoundError('Provide a Git checkout of the pinned FreeSurfer source for this build')
        if source.exists() and any(source.iterdir()):
            raise ValueError('Source destination is not empty; choose another --work-dir')
        source.mkdir(parents=True, exist_ok=True)
        run([git, 'init', source], log, env)
        run([git, '-C', source, 'remote', 'add', 'origin', FS_URL], log, env)
        run([git, '-C', source, 'fetch', '--depth=1', '--filter=blob:none', 'origin', FS_REVISION], log, env)
        run([git, '-C', source, 'sparse-checkout', 'set', '--cone', *SOURCE_DIRECTORIES], log, env)
        run([git, '-C', source, 'checkout', '--detach', FS_REVISION], log, env)
    revision = subprocess.check_output([git, '-C', str(source), 'rev-parse', 'HEAD'], text=True, env=env).strip()
    if revision != FS_REVISION:
        raise ValueError('FreeSurfer source must match pinned revision ' + FS_REVISION)
    dirty = subprocess.run([git, '-C', str(source), 'diff', '--quiet', 'HEAD', '--', *SOURCE_DIRECTORIES], env=env)
    if dirty.returncode:
        raise ValueError('Selected upstream source contains local edits; refusing to call it an unchanged source build')
    untracked = subprocess.check_output([git, '-C', str(source), 'ls-files', '--others', '--exclude-standard',
                                         '--', *SOURCE_DIRECTORIES], text=True, env=env).strip()
    if untracked:
        raise ValueError('Selected upstream source contains untracked files; use a clean checkout')
    return source


def stage_source(source, work):
    stage = work / 'stage'
    marker = stage / '.ddsurfer/source-revision.txt'
    if not stage.exists():
        stage.mkdir()
        for name in SOURCE_DIRECTORIES:
            shutil.copytree(source / name, stage / name, symlinks=True)
        (stage / '.ddsurfer').mkdir()
        marker.write_text(FS_REVISION + '\n')
    elif not marker.is_file() or marker.read_text().strip() != FS_REVISION:
        raise ValueError('Unrecognized staged source; choose a fresh --work-dir')
    else:
        for name in SOURCE_DIRECTORIES:
            original, copied = source / name, stage / name
            expected = {path.relative_to(original) for path in original.rglob('*') if path.is_file() or path.is_symlink()}
            actual = {path.relative_to(copied) for path in copied.rglob('*') if path.is_file() or path.is_symlink()}
            if actual != expected:
                raise ValueError('Staged source differs from upstream; choose a fresh --work-dir')
            for relative in expected:
                a, b = original / relative, copied / relative
                same = (a.is_symlink() and b.is_symlink() and os.readlink(a) == os.readlink(b))
                if not same and (a.is_symlink() or b.is_symlink() or sha256_file(a) != sha256_file(b)):
                    raise ValueError(f'Staged source was edited: {name}/{relative}')
    shutil.copyfile(ROOT / 'cmake/CMakeLists.txt', stage / 'CMakeLists.txt')
    shutil.copyfile(ROOT / 'cmake/help_header.py', stage / '.ddsurfer/help_header.py')
    return stage


def build(args):
    system, machine = host()
    if args.jobs < 1:
        raise ValueError('--jobs must be positive')
    cmake = require_tool(args.cmake, 'cmake')
    git = require_tool(None, 'git')
    cc = require_tool(os.environ.get('CC'), 'cc')
    cxx = require_tool(os.environ.get('CXX'), 'c++')
    version = subprocess.check_output([cmake, '--version'], text=True).splitlines()[0]
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', version)
    if not match or not (3, 24) <= tuple(map(int, match.groups()[:2])) < (4, 0):
        raise ValueError('Use CMake >=3.24,<4 for this pinned upstream source')
    prefix = (args.prefix if args.prefix is not None else default_prefix()).expanduser().absolute()
    work = (args.work_dir if args.work_dir is not None else cache_root() / 'build' / recipe_digest()[:16]).expanduser().absolute()
    if prefix == work or work in prefix.parents or prefix in work.parents:
        # Keeping the installed runtime outside the build tree permits later build-cache cleanup.
        raise ValueError('Install prefix and build work directory must be separate')
    env = dict(os.environ, CC=cc, CXX=cxx, CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs),
               OMP_NUM_THREADS=str(args.jobs), OPENBLAS_NUM_THREADS='1')
    for name in ('FREESURFER_HOME', 'SUBJECTS_DIR', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT'):
        env.pop(name, None)
    work.mkdir(parents=True, exist_ok=True)
    from utils.stages import subject_lock
    with subject_lock(work / 'build.lock'):
        if prefix.exists():
            record = validate_runtime(prefix)
            if record.get('recipe_sha256') != recipe_digest():
                raise ValueError('Existing runtime uses a different build recipe; choose another --prefix')
            print('[ready] ' + str(prefix), flush=True)
            return prefix
        source = prepare_source(work, args.source_root, git, args.offline, env)
        itk = prepare_dependency('itk', work, args.offline)
        zlib = prepare_dependency('zlib', work, args.offline)
        itk_prefix, itk_build = work / 'itk-install', work / 'itk-build'
        run([cmake, '-S', itk, '-B', itk_build, '-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_CXX_STANDARD=11', '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
             '-DBUILD_SHARED_LIBS=OFF', '-DBUILD_TESTING=OFF', '-DITK_BUILD_DEFAULT_MODULES=OFF',
             '-DITKGroup_Core=OFF', '-DModule_ITKCommon=ON', '-DCMAKE_INSTALL_PREFIX=' + str(itk_prefix)],
            work / 'itk-configure.log', env)
        run([cmake, '--build', itk_build, '--parallel', args.jobs], work / 'itk-build.log', env)
        run([cmake, '--install', itk_build], work / 'itk-install.log', env)
        stage = stage_source(source, work)
        native = work / 'surface-build'
        run([cmake, '-S', stage, '-B', native, '-DCMAKE_BUILD_TYPE=Release',
             '-DITK_DIR=' + str(itk_prefix / 'lib/cmake/ITK-4.13'),
             '-DDDSURFER_ZLIB_SOURCE=' + str(zlib), '-DPython3_EXECUTABLE=' + sys.executable],
            work / 'surface-configure.log', env)
        run([cmake, '--build', native, '--parallel', args.jobs, '--target', *COMMANDS], work / 'surface-build.log', env)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.local-surface-', dir=prefix.parent) as temporary:
            runtime = Path(temporary) / 'runtime'
            runtime.mkdir()
            install_assets(runtime)
            (runtime / 'bin').mkdir()
            for name in COMMANDS:
                shutil.copy2(native / 'bin' / name, runtime / 'bin' / name)
                run([runtime / 'bin' / name, '--version'], work / 'smoke.log', env)
            (runtime / 'build-stamp.txt').write_text('ddsurfer-local-freesurfer-7eb8460\n')
            for name, origin in [('license.ITK-source.txt', itk / 'LICENSE'), ('notice.ITK-source.txt', itk / 'NOTICE'),
                                 ('license.zlib-source.txt', zlib / 'LICENSE')]:
                if origin.is_file():
                    shutil.copyfile(origin, runtime / 'docs' / name)
            files = {str(path.relative_to(runtime)): dict(bytes=path.stat().st_size, sha256=sha256_file(path))
                     for path in runtime.rglob('*') if path.is_file()}
            record = dict(schema=SCHEMA, source_revision=FS_REVISION, system=system, machine=machine,
                          recipe_sha256=recipe_digest(), activation_key_included=False,
                          build=dict(cmake=version, cxx=subprocess.check_output([cxx, '--version'], text=True).splitlines()[0],
                                     dependencies=DEPENDENCIES, itk_compiler_guard_patch=True,
                                     environment_flags={name: env.get(name, '') for name in ('CFLAGS', 'CXXFLAGS', 'LDFLAGS')}), files=files)
            atomic_json(runtime / 'runtime.json', record)
            validate_runtime(runtime)
            runtime.rename(prefix)
        print('[ready] ' + str(prefix), flush=True)
    return prefix


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--prefix', type=Path, help='Fresh installation directory; defaults to a recipe-addressed user cache.')
    parser.add_argument('--work-dir', type=Path, help='Build/download cache; keep it outside the installed runtime.')
    parser.add_argument('--source-root', type=Path, help='Optional clean Git checkout of the pinned source revision.')
    parser.add_argument('--cmake', type=Path, help='Override the CMake executable.')
    parser.add_argument('--offline', action='store_true', help='Require an existing source checkout and cached dependency archives.')
    args = parser.parse_args(argv)
    try:
        build(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Build failed: {error}\nSee the build logs and postprocessing/core/README.md.\n')


if __name__ == '__main__':
    main()
