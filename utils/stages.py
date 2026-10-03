"""Content-checked stage reuse and atomic publication of pipeline outputs."""
from contextlib import contextmanager, ExitStack
import fcntl
import json
import os
from pathlib import Path
import tempfile
from threading import Lock
from time import perf_counter

from utils.files import atomic_json, sha256_file


@contextmanager
def subject_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another process is using this subject cache: {path.parent}')
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class StageCache:
    def __init__(self, record, related_files=None, validate=None):
        self.path = Path(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.related_files = related_files or (lambda path: [Path(path)])
        self.validate = validate or self._validate_file
        self.lock = Lock()
        self.digests = {}

    @staticmethod
    def _validate_file(path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing or empty stage output: {path}')

    def digest(self, path):
        path = Path(path)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        with self.lock:
            if key not in self.digests:
                self.digests[key] = sha256_file(path)
            return self.digests[key]

    def fingerprints(self, paths):
        return {str(file.resolve()): self.digest(file)
                for path in paths for file in self.related_files(Path(path))}

    def run(self, key, inputs, outputs, action, settings=None):
        outputs = [Path(path) for path in outputs]
        signature = json.loads(json.dumps(dict(inputs=self.fingerprints(inputs), settings=settings)))
        previous = self.records.get(key)
        if previous is not None and previous['signature'] == signature:
            try:
                valid = self.fingerprints(outputs) == previous['outputs']
            except (OSError, ValueError):
                valid = False
            if valid:
                print(f'[skip] {key}', flush=True)
                return False
        # A receipt is valid only after every file (including detached data) is published.
        with self.lock:
            self.records.pop(key, None)
            atomic_json(self.path, self.records)
        started = perf_counter()
        print(f'[run] {key}', flush=True)
        with ExitStack() as stack:
            staged = []
            for output in outputs:
                output.parent.mkdir(parents=True, exist_ok=True)
                folder = Path(stack.enter_context(tempfile.TemporaryDirectory(
                    prefix='.stage-', dir=str(output.parent))))
                staged.append(folder / output.name)
            action(staged)
            for path in staged:
                self.validate(path)
            for source, target in zip(staged, outputs):
                for companion in self.related_files(source)[1:]:
                    if companion.is_symlink():
                        destination = target.parent / companion.name
                        if not destination.exists() and not destination.is_symlink():
                            destination.symlink_to(companion.resolve())
                        elif destination.resolve() != companion.resolve():
                            raise ValueError(f'Detached input filename conflict: {destination}')
                    else:
                        if companion.parent != source.parent:
                            raise ValueError('Detached output data must be beside its header')
                        os.replace(companion, target.parent / companion.name)
                os.replace(source, target)
        fingerprints = self.fingerprints(outputs)
        with self.lock:
            self.records[key] = dict(signature=signature, outputs=fingerprints,
                                     seconds=perf_counter() - started)
            atomic_json(self.path, self.records)
        return True
