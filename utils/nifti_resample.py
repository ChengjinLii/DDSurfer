"""Resample a NIfTI volume into a target image space."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import SimpleITK as sitk

LOGGER = logging.getLogger(__name__)


def resample_image_to_target_space(
    source_image_path: Path,
    target_image_path: Path,
    output_file_path: Path,
) -> None:
    """Resample `source_image_path` into the voxel grid of `target_image_path`."""
    LOGGER.info("Loading source image: %s", source_image_path)
    source_image = sitk.ReadImage(str(source_image_path))

    LOGGER.info("Loading target image: %s", target_image_path)
    target_image = sitk.ReadImage(str(target_image_path))

    target_spacing = target_image.GetSpacing()
    target_size = target_image.GetSize()
    target_origin = target_image.GetOrigin()
    target_direction = target_image.GetDirection()

    LOGGER.debug(
        "Target image properties | spacing=%s size=%s origin=%s",
        target_spacing,
        target_size,
        target_origin,
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
    parser.add_argument("--target_image_path", required=True, type=Path, help="Path to the target NIfTI image.")
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
    resample_image_to_target_space(args.source_image_path, args.target_image_path, args.output_file_path)


if __name__ == "__main__":
    main()
