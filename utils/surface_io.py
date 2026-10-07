"""Indexed triangular mesh I/O and compatible STL loading."""

from pathlib import Path
import os

import numpy as np
import trimesh


def load_mesh(path, *, indexed=True):
    mesh = trimesh.load(str(path), process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError('Expected a single triangular surface')
    if not np.isfinite(mesh.vertices).all():
        raise ValueError('Nonfinite surface vertices')
    if indexed and Path(path).suffix.lower() == '.stl':
        mesh, _, _ = index_triangles(mesh.vertices[mesh.faces])
    return mesh


def index_triangles(triangles):
    points = np.asarray(triangles, dtype=np.float64).reshape(-1, 3)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError('Expected finite triangle coordinates')
    # STL stores triangle corners, not vertex indices. Weld identical points
    # without rounding away nearby but distinct cortical vertices.
    _, first, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    remap = np.empty(len(first), dtype=np.int64)
    remap[order] = np.arange(len(first))
    first, inverse = first[order], remap[inverse]
    mesh = trimesh.Trimesh(vertices=points[first], faces=inverse.reshape(-1, 3), process=False)
    return mesh, first, inverse


def write_obj(path, vertices, faces):
    """Write scanner-RAS millimetre coordinates with an explicit Slicer header."""
    path = Path(path)
    if path.suffix.lower() != '.obj':
        raise ValueError('Surface outputs must use .obj')
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Expected finite Nx3 vertices')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Invalid triangle indices')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.obj')
    try:
        # Slicer assumes LPS for unmarked OBJ files; declare RAS without changing geometry.
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(
            temp, header='DDSurfer surface. SPACE=RAS')
        stored = load_mesh(temp)
        if stored.vertices.shape != vertices.shape or not np.array_equal(stored.faces, faces):
            raise ValueError('OBJ export changed vertex indices or connectivity')
        error = float(np.max(np.abs(stored.vertices - vertices)))
        if error > 1e-5:
            raise ValueError('OBJ export changed millimetre coordinates')
        os.replace(temp, path)
        return error
    finally:
        if temp.exists():
            temp.unlink()
