"""Resample a NIfTI volume into a target image space."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence

import SimpleITK as sitk

LOGGER = logging.getLogger(__name__)


def _normalise_direction(direction: Sequence[float]) -> tuple[float, ...]:
    """Validate and normalise a 3D direction cosine matrix."""
    if len(direction) != 9:
        msg = "target direction must contain exactly 9 values for a 3D image."
        raise ValueError(msg)
    return tuple(float(value) for value in direction)


def _load_target_geometry_from_image(target_image_path: Path) -> tuple[tuple[float, ...], tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    """Load spacing/size/origin/direction from a reference image."""
    LOGGER.info("Loading target image: %s", target_image_path)
    target_image = sitk.ReadImage(str(target_image_path))
    return (
        tuple(float(value) for value in target_image.GetSpacing()),
        tuple(int(value) for value in target_image.GetSize()),
        tuple(float(value) for value in target_image.GetOrigin()),
        _normalise_direction(target_image.GetDirection()),
    )


def _resolve_target_geometry(args: argparse.Namespace) -> tuple[tuple[float, ...], tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    """Resolve target geometry from either a reference image or explicit values."""
    if args.target_image_path is not None:
        return _load_target_geometry_from_image(args.target_image_path)

    if None in (args.target_size, args.target_spacing, args.target_origin, args.target_direction):
        msg = (
            "Either --target_image_path must be provided, or all of "
            "--target_size/--target_spacing/--target_origin/--target_direction must be set."
        )
        raise ValueError(msg)

    return (
        tuple(float(value) for value in args.target_spacing),
        tuple(int(value) for value in args.target_size),
        tuple(float(value) for value in args.target_origin),
        _normalise_direction(args.target_direction),
    )


def resample_image_to_target_space(
    source_image_path: Path,
    output_file_path: Path,
    target_spacing: Sequence[float],
    target_size: Sequence[int],
    target_origin: Sequence[float],
    target_direction: Sequence[float],
) -> None:
    """Resample `source_image_path` into an explicitly specified voxel grid."""
    LOGGER.info("Loading source image: %s", source_image_path)
    source_image = sitk.ReadImage(str(source_image_path))

    LOGGER.debug(
        "Target geometry | spacing=%s size=%s origin=%s direction=%s",
        target_spacing,
        target_size,
        target_origin,
        target_direction,
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetInterpolator(sitk.sitkLinear)

    LOGGER.info("Resampling volume...")
    resampled_image = resampler.Execute(source_image)

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampled_image, str(output_file_path))
    LOGGER.info("Resampled image written to %s", output_file_path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample an image to a target reference space.")
    parser.add_argument("--source_image_path", required=True, type=Path, help="Path to the source NIfTI image.")
    parser.add_argument("--target_image_path", type=Path, help="Path to the target NIfTI image.")
    parser.add_argument("--target_size", nargs=3, type=int, metavar=("NX", "NY", "NZ"), help="Explicit output size.")
    parser.add_argument(
        "--target_spacing",
        nargs=3,
        type=float,
        metavar=("SX", "SY", "SZ"),
        help="Explicit output voxel spacing.",
    )
    parser.add_argument(
        "--target_origin",
        nargs=3,
        type=float,
        metavar=("OX", "OY", "OZ"),
        help="Explicit output image origin.",
    )
    parser.add_argument(
        "--target_direction",
        nargs=9,
        type=float,
        metavar=("D11", "D12", "D13", "D21", "D22", "D23", "D31", "D32", "D33"),
        help="Explicit 3x3 direction cosine matrix in row-major order.",
    )
    parser.add_argument("--output_file_path", required=True, type=Path, help="Destination for the resampled image.")
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
    target_spacing, target_size, target_origin, target_direction = _resolve_target_geometry(args)
    resample_image_to_target_space(
        args.source_image_path,
        args.output_file_path,
        target_spacing=target_spacing,
        target_size=target_size,
        target_origin=target_origin,
        target_direction=target_direction,
    )


if __name__ == "__main__":
    main()
