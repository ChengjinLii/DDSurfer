"""Apply an ITK .tfm transform to STL meshes (MNI → native space).

This utility converts predicted cortical surfaces stored as STL meshes in LPS
orientation to RAS, applies a SimpleITK transform, and writes both RAS- and
LPS-oriented outputs back to disk. The computational steps follow the original
DDSurfer release while providing a clearer interface and logging suitable for
open-source maintenance.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import SimpleITK as sitk
import trimesh

LOGGER = logging.getLogger(__name__)


def _flip_lps_ras(vertices: np.ndarray) -> np.ndarray:
    """Convert between LPS and RAS coordinate conventions."""
    return vertices * np.array([-1.0, -1.0, 1.0], dtype=np.float64)


def transform_mesh_with_tfm(
    input_stl: Path,
    output_stl: Path,
    tfm_inverse_transform: Path,
    debug_vertices: int = 0,
) -> None:
    """Apply the inverse TFM transform to an input STL mesh."""
    if not input_stl.is_file():
        raise FileNotFoundError(f"Input STL file not found: {input_stl}")
    if not tfm_inverse_transform.is_file():
        raise FileNotFoundError(f"Transform file not found: {tfm_inverse_transform}")

    LOGGER.debug("Loading ITK transform from %s", tfm_inverse_transform)
    transform = sitk.ReadTransform(str(tfm_inverse_transform))
    LOGGER.info("Loaded %s transform", transform.GetName())

    LOGGER.debug("Loading STL mesh from %s", input_stl)
    mesh_lps = trimesh.load(str(input_stl), process=False)
    LOGGER.info("Mesh loaded: %d vertices, %d faces", len(mesh_lps.vertices), len(mesh_lps.faces))

    vertices_lps = mesh_lps.vertices.astype(np.float64)
    if debug_vertices > 0:
        LOGGER.debug("First %d LPS vertices:\n%s", debug_vertices, vertices_lps[:debug_vertices])

    vertices_ras = _flip_lps_ras(vertices_lps)
    if debug_vertices > 0:
        LOGGER.debug("First %d RAS vertices:\n%s", debug_vertices, vertices_ras[:debug_vertices])

    LOGGER.info("Applying inverse transform to %d vertices", len(vertices_ras))
    start = perf_counter()
    transformed_vertices_ras = np.asarray(
        [transform.TransformPoint(tuple(vertex)) for vertex in vertices_ras],
        dtype=np.float64,
    )
    elapsed = perf_counter() - start
    LOGGER.info("Transform applied in %.2f seconds", elapsed)

    if debug_vertices > 0:
        LOGGER.debug(
            "First %d transformed RAS vertices:\n%s",
            debug_vertices,
            transformed_vertices_ras[:debug_vertices],
        )

    output_mesh_ras = trimesh.Trimesh(
        vertices=transformed_vertices_ras,
        faces=mesh_lps.faces,
        process=False,
    )
    output_stl.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing transformed RAS mesh to %s", output_stl)
    output_mesh_ras.export(str(output_stl))

    vertices_lps_out = _flip_lps_ras(transformed_vertices_ras)
    if debug_vertices > 0:
        LOGGER.debug(
            "First %d transformed LPS vertices:\n%s",
            debug_vertices,
            vertices_lps_out[:debug_vertices],
        )

    output_mesh_lps = trimesh.Trimesh(
        vertices=vertices_lps_out,
        faces=mesh_lps.faces,
        process=False,
    )
    lps_path = output_stl.with_name(f"{output_stl.stem}_orig{output_stl.suffix}")
    LOGGER.info("Writing transformed LPS mesh to %s", lps_path)
    output_mesh_lps.export(str(lps_path))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply an inverse ITK transform to an STL mesh (MNI→native)."
    )
    parser.add_argument(
        "--tfm_inverse_transform",
        required=True,
        type=Path,
        help="Path to the ITK .tfm transform file.",
    )
    parser.add_argument(
        "--input_stl",
        required=True,
        type=Path,
        help="Path to the input STL file (LPS orientation).",
    )
    parser.add_argument(
        "--output_stl",
        required=True,
        type=Path,
        help="Destination for the transformed STL (RAS orientation).",
    )
    parser.add_argument(
        "--debug-vertices",
        type=int,
        default=0,
        help="Print coordinates for the first N vertices at each stage.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    LOGGER.info("Input STL: %s", args.input_stl)
    LOGGER.info("Output STL: %s", args.output_stl)
    LOGGER.info("Inverse transform: %s", args.tfm_inverse_transform)

    try:
        transform_mesh_with_tfm(
            input_stl=args.input_stl,
            output_stl=args.output_stl,
            tfm_inverse_transform=args.tfm_inverse_transform,
            debug_vertices=args.debug_vertices,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.exception("Surface transformation failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
