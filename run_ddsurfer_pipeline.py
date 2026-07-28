"""End-to-end DDSurfer pipeline runner.

Given a subject identifier, this script orchestrates preprocessing, surface
prediction for both hemispheres, and post-processing back to native space.
Each stage delegates to existing project utilities to preserve the original
logic while providing a modern, automation-friendly interface.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent


def run_command(command: Sequence[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    """Execute `command`, logging the shell equivalent for reproducibility."""
    logging.info("Executing: %s", " ".join(shlex.quote(part) for part in command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DDSurfer pipeline for a single subject.")
    parser.add_argument("--subject", required=True, help="Subject identifier, matching directory names in the input tree.")
    parser.add_argument(
        "--data-type",
        default="hcp",
        help="Dataset type (passed through to the prediction scripts).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device to use for prediction (e.g. cuda:0 or cpu).",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "DTI-inputs",
        help="Directory containing subject-specific registered DTI inputs.",
    )
    parser.add_argument(
        "--raw-input-root",
        type=Path,
        default=PROJECT_ROOT / "raw-dwi-inputs",
        help="Directory containing raw diffusion inputs under <ID>/T1w/Diffusion.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data_Reg" / "test",
        help="Destination for resampled, preprocessed volumes.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=PROJECT_ROOT / "pred_results_DDSurfer",
        help="Directory where predicted meshes will be stored.",
    )
    parser.add_argument(
        "--predict-mode",
        choices=("wm", "all"),
        default="all",
        help="Prediction mode passed to the hemisphere scripts.",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Assume preprocessing outputs already exist and skip Data-Preprocessing.sh.",
    )
    parser.add_argument(
        "--skip-postprocessing",
        action="store_true",
        help="Skip utils/space_MNI2orig.sh (useful when only meshes are required).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def build_preprocessing_command(args: argparse.Namespace) -> List[str]:
    return [
        "bash",
        str(PROJECT_ROOT / "Data-Preprocessing.sh"),
        "--subject",
        args.subject,
        "--raw-input-root",
        str(args.raw_input_root),
        "--input-root",
        str(args.input_root),
        "--output-root",
        str(args.output_root),
    ]


def build_prediction_command(
    script_path: Path,
    args: argparse.Namespace,
    hemisphere: str,
) -> List[str]:
    return [
        "python3",
        str(script_path),
        "--data_type",
        args.data_type,
        "--surf_hemi",
        hemisphere,
        "--device",
        args.device,
        "--input_root",
        str(args.output_root),
        "--output_dir",
        str(args.predictions_dir),
        "--predict_mode",
        args.predict_mode,
        "--subjects",
        args.subject,
    ]


def build_postprocessing_command(args: argparse.Namespace) -> List[str]:
    return [
        "bash",
        str(PROJECT_ROOT / "utils" / "space_MNI2orig.sh"),
        "--subject",
        args.subject,
        "--mode",
        "whole",
        "--data-root",
        str(args.output_root),
        "--pred-root",
        str(args.predictions_dir),
    ]


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    if not args.skip_preprocessing:
        run_command(build_preprocessing_command(args))
    else:
        logging.info("Skipping preprocessing as requested.")

    left_script = PROJECT_ROOT / "ddsurfer_predict_lh_dualstream.py"
    right_script = PROJECT_ROOT / "ddsurfer_predict_rh_dualstream.py"

    run_command(build_prediction_command(left_script, args, "left"))
    run_command(build_prediction_command(right_script, args, "right"))

    if args.skip_postprocessing:
        logging.info("Skipping post-processing as requested.")
        return

    run_command(build_postprocessing_command(args))


if __name__ == "__main__":
    main()
