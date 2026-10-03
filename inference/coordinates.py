"""One explicit RAS <-> crop-voxel contract for inputs, labels and exports."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine

from utils.files import atomic_json, sha256_file
from utils.surface_io import write_obj


MODALITIES = ("FA", "MidEigenvalue", "MinEigenvalue", "MaxEigenvalue", "MD")


class Geometry:
    def __init__(self, affine, shape, hemisphere):
        self.affine = np.asarray(affine, dtype=np.float64)
        self.shape = tuple(int(x) for x in shape)
        self.hemisphere = hemisphere
        if hemisphere not in ("left", "right"):
            raise ValueError("Hemisphere must be left or right")
        if self.shape != (176, 224, 176):
            raise ValueError(f"Expected DDSurfer full shape (176,224,176), got {self.shape}")
        # Coordinates use a 1-mm RAS-aligned grid.
        if self.affine.shape != (4, 4) or not np.isfinite(self.affine).all():
            raise ValueError("Expected a finite 4x4 image affine")
        if not np.allclose(self.affine[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
            raise ValueError("Invalid homogeneous affine")
        if not np.allclose(self.affine[:3, :3], np.eye(3), atol=1e-5, rtol=0):
            raise ValueError("Expected the fixed 1-mm RAS-aligned DDSurfer grid")
        self.inverse = np.linalg.inv(self.affine)
        self.crop_origin = np.array([0 if hemisphere == "left" else 64, 0, 0])
        self.crop_shape = (112, self.shape[1], self.shape[2])
        self.x_slice = slice(0, 112) if hemisphere == "left" else slice(64, 176)

    @classmethod
    def from_image(cls, path, hemisphere):
        image = nib.load(str(path))
        geometry = cls(image.affine, image.shape, hemisphere)
        geometry.check_image(image, path)
        return geometry

    def check_image(self, image, path="image"):
        if tuple(image.shape) != self.shape or not np.allclose(image.affine, self.affine, atol=1e-5, rtol=0):
            raise ValueError(f"Image geometry mismatch: {path}")
        qform, qcode = image.get_qform(coded=True)
        sform, scode = image.get_sform(coded=True)
        if qcode and scode and not np.allclose(qform, sform, atol=1e-4, rtol=0):
            raise ValueError(f"Conflicting qform/sform: {path}")

    def ras_to_crop(self, vertices):
        return apply_affine(self.inverse, np.asarray(vertices)) - self.crop_origin

    def crop_to_ras(self, vertices):
        return apply_affine(self.affine, np.asarray(vertices) + self.crop_origin)

    def to_dict(self):
        return {"hemisphere": self.hemisphere,
                "affine": self.affine.tolist(), "full_shape": list(self.shape),
                "crop_origin": self.crop_origin.tolist(), "crop_shape": list(self.crop_shape)}


def export_crop_mesh(path, vertices, faces, geometry):
    """Export model crop indices as scanner RAS."""
    path = Path(path)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Expected finite Nx3 crop vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Invalid export triangle indices")
    ras = geometry.crop_to_ras(vertices)
    path.parent.mkdir(parents=True, exist_ok=True)
    error = write_obj(path, ras, faces)
    atomic_json(path.with_suffix(".json"), {"coordinate_space": "scanner_RAS_mm",
                "geometry": geometry.to_dict(), "vertices": len(vertices), "faces": len(faces),
                "max_export_error_mm": error, "sha256": sha256_file(path)})
    return error
