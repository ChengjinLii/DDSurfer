"""Mask-based skull stripping utility.

This script multiplies an input anatomical or diffusion-derived volume by a
binary brain mask to remove non-brain voxels. The behaviour matches the
original DDSurfer helper while providing a clearer interface and logging.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np

LOGGER = logging.getLogger(__name__)


def skull_strip(volume_path: Path, mask_path: Path, output_path: Path) -> None:
    """Apply a binary mask to the input volume."""
    LOGGER.debug("Loading volume from %s", volume_path)
    volume_img = nib.load(str(volume_path))
    volume_data = volume_img.get_fdata(dtype=np.float32)

    LOGGER.debug("Loading mask from %s", mask_path)
    mask_data = nib.load(str(mask_path)).get_fdata().astype(bool)

    if mask_data.shape != volume_data.shape:
        raise ValueError(
            f"Mask shape {mask_data.shape} does not match volume shape {volume_data.shape}"
        )

    stripped = np.where(mask_data, volume_data, 0.0)

    output_img = nib.Nifti1Image(stripped, volume_img.affine, volume_img.header)
    output_img.set_data_dtype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output_img, str(output_path))
    LOGGER.info("Skull-stripped volume saved to %s", output_path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a binary mask for skull stripping.")
    parser.add_argument("--input_path", required=True, type=Path, help="Path to the input NIfTI volume.")
    parser.add_argument("--mask_path", required=True, type=Path, help="Path to the brain mask NIfTI volume.")
    parser.add_argument("--output_path", required=True, type=Path, help="Destination for the stripped volume.")
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
    skull_strip(args.input_path, args.mask_path, args.output_path)


if __name__ == "__main__":
    main()
