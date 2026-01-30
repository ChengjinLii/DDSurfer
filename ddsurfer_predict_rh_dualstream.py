"""DDSurfer surface prediction pipeline (right hemisphere defaults).

This script mirrors the historical DDSurfer inference pipeline for predicting
white-matter (WM) and pial cortical surfaces using the cascaded dual-stream
TANet model. The logic has been reorganised for clarity, configurability, and
logging while preserving the original computational steps.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import nibabel as nib
import numpy as np
import torch
import trimesh
from tqdm import tqdm

from net.ddsurfer_tanet_dualstream_cascaded import TANet
from utils.mesh import taubin_smooth

LOGGER = logging.getLogger(__name__)


VOLUME_MODALITIES: Sequence[str] = (
    "FA",
    "MidEigenvalue",
    "MinEigenvalue",
    "MaxEigenvalue",
    "MD",
)


@dataclass(frozen=True)
class PredictionConfig:
    """Encapsulates configuration required for surface prediction."""

    data_type: str
    hemisphere: str
    predict_mode: str
    device: torch.device
    step_size: float
    n_svf: int
    n_res: int
    template_dir: Path
    checkpoint_root: Path
    output_dir: Path
    input_root: Path
    subjects: Sequence[str] | None = None

    @property
    def template_mesh_path(self) -> Path:
        return self.template_dir / f"hcp_hemi-{self.hemisphere}_init_160k.obj"

    def checkpoint_path(self, surface: str) -> Path:
        suffix = "rh"
        return self.checkpoint_root / self.data_type / f"DDCSR_{surface}_{suffix}_Full_DualStream_no_b0_Cascaded.pt"

    @property
    def translation_vector(self) -> np.ndarray:
        if self.hemisphere == "right":
            return np.array([21.0, 132.0, 70.0], dtype=np.float32)
        return np.array([85.0, 132.0, 70.0], dtype=np.float32)

    @property
    def x_slice(self) -> slice:
        if self.hemisphere == "left":
            return slice(0, 112)
        if self.hemisphere == "right":
            return slice(64, None)
        msg = f"Unsupported hemisphere: {self.hemisphere}"
        raise ValueError(msg)


class SurfacePredictor:
    """Wraps TANet inference for predicting DDSurfer cortical surfaces."""

    def __init__(self, cfg: PredictionConfig) -> None:
        self.cfg = cfg
        self.device = cfg.device

        LOGGER.info("Loading template mesh from %s", cfg.template_mesh_path)
        mesh_init = trimesh.load(cfg.template_mesh_path, process=False)
        vertices = mesh_init.vertices
        if cfg.hemisphere == "right":
            vertices = vertices.copy()
            vertices[:, 0] -= 64.0

        self.vert_init = vertices
        self.face_init = mesh_init.faces

        self.wm_model = self._load_model("white")
        self.pial_model = self._load_model("pial") if cfg.predict_mode == "all" else None

    def _load_model(self, surface: str) -> TANet:
        LOGGER.info("Loading %s surface model for hemisphere '%s'", surface, self.cfg.hemisphere)
        if surface == "pial":
            hidden = [16, 32, 32, 32, 32]
        else:
            hidden = [16, 32, 64, 128, 256]

        model = TANet(
            C_in=5,
            C_hid=hidden,
            inshape=[112, 224, 176],
            step_size=self.cfg.step_size,
            M=self.cfg.n_svf,
            R=self.cfg.n_res,
            device=self.device,
        )

        ckpt_path = self.cfg.checkpoint_path(surface)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

        state_dict = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _load_subject_volumes(self, subject_dir: Path) -> torch.Tensor:
        volumes: List[np.ndarray] = []
        for modality in VOLUME_MODALITIES:
            volume_path = subject_dir / f"{subject_dir.name}-{modality}.nii.gz"
            if not volume_path.exists():
                raise FileNotFoundError(f"Missing volume for modality '{modality}': {volume_path}")
            data = nib.load(volume_path.as_posix()).get_fdata()
            data = data[self.cfg.x_slice, :, :]
            volumes.append(data)

        stacked = np.stack(volumes, axis=0).astype(np.float32)
        return torch.from_numpy(stacked).unsqueeze(0).to(self.device).float()

    def predict(self, subject_dir: Path, output_dir: Path) -> float:
        LOGGER.info(
            "Predicting %s surfaces for subject %s (%s hemisphere)",
            self.cfg.predict_mode.upper(),
            subject_dir.name,
            self.cfg.hemisphere,
        )
        start = time.perf_counter()

        volumes = self._load_subject_volumes(subject_dir)
        vertices_init = torch.tensor(self.vert_init[None], device=self.device, dtype=torch.float32)
        faces = torch.tensor(self.face_init[None], device=self.device, dtype=torch.long)

        with torch.no_grad():
            vertices_wm = self.wm_model(vertices_init, volumes)
            vertices_wm = taubin_smooth(vertices_wm, faces, n_iters=10)
            verts_wm_np = vertices_wm[0].cpu().numpy()

            verts_pial_np = None
            if self.cfg.predict_mode == "all":
                if self.pial_model is None:
                    raise RuntimeError("Pial model requested but not initialised.")
                wm_tensor = torch.tensor(verts_wm_np[None], device=self.device, dtype=torch.float32)
                vertices_pial = self.pial_model(wm_tensor, volumes)
                verts_pial_np = vertices_pial[0].cpu().numpy()

        translation = self.cfg.translation_vector

        mni_dir = output_dir / "mni" / subject_dir.name
        mni_dir.mkdir(parents=True, exist_ok=True)

        transformed_wm = verts_wm_np - translation
        mesh_wm = trimesh.Trimesh(vertices=transformed_wm, faces=self.face_init, process=False)
        mesh_wm.export(mni_dir / f"{subject_dir.name}_predicted_wm_surface_{self.cfg.hemisphere}.obj")

        if self.cfg.predict_mode == "all" and verts_pial_np is not None:
            transformed_pial = verts_pial_np - translation
            mesh_pial = trimesh.Trimesh(vertices=transformed_pial, faces=self.face_init, process=False)
            mesh_pial.export(mni_dir / f"{subject_dir.name}_predicted_pial_surface_{self.cfg.hemisphere}.obj")

        elapsed = time.perf_counter() - start
        LOGGER.info("Subject %s processed in %.2f seconds", subject_dir.name, elapsed)
        return elapsed


def discover_subjects(cfg: PredictionConfig) -> List[Path]:
    if cfg.subjects:
        subjects: List[Path] = []
        for subject in cfg.subjects:
            subject_dir = cfg.input_root / subject
            if not subject_dir.is_dir():
                LOGGER.warning("Subject directory not found: %s", subject_dir)
                continue
            subjects.append(subject_dir)
        return subjects
    return [path for path in sorted(cfg.input_root.iterdir()) if path.is_dir()]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict DDSurfer cortical surfaces (right hemisphere defaults).")
    parser.add_argument("--data_type", default="hcp", help="Dataset type: e.g. hcp or dhcp.")
    parser.add_argument(
        "--surf_hemi",
        default="right",
        choices=("left", "right"),
        help="Hemisphere to process.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device identifier.")
    parser.add_argument("--step_size", default=0.02, type=float, help="Integration step size for TANet.")
    parser.add_argument("--n_svf", default=2, type=int, help="Number of stationary velocity fields for TANet.")
    parser.add_argument("--n_res", default=3, type=int, help="Number of resolution levels for TANet.")
    parser.add_argument(
        "--predict_mode",
        default="all",
        choices=("wm", "all"),
        help="Whether to predict only the WM surface or both WM and pial surfaces.",
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        help="Root directory containing preprocessed volumetric inputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./pred_results_DDSurfer"),
        help="Directory where predicted surfaces will be written.",
    )
    parser.add_argument(
        "--template_dir",
        type=Path,
        default=Path("./template"),
        help="Directory containing the mesh template files.",
    )
    parser.add_argument(
        "--checkpoint_root",
        type=Path,
        default=Path("./ckpts"),
        help="Directory containing trained model checkpoints.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Optional list of subject identifiers. Defaults to all directories under input_root.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.input_root is None:
        args.input_root = Path(f"./data_Reg/{args.data_type}/test")
    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(message)s",
    )

    cfg = PredictionConfig(
        data_type=args.data_type,
        hemisphere=args.surf_hemi,
        predict_mode=args.predict_mode,
        device=torch.device(args.device),
        step_size=args.step_size,
        n_svf=args.n_svf,
        n_res=args.n_res,
        template_dir=args.template_dir,
        checkpoint_root=args.checkpoint_root,
        output_dir=args.output_dir,
        input_root=args.input_root,
        subjects=args.subjects,
    )

    subjects = discover_subjects(cfg)
    if not subjects:
        LOGGER.warning("No subjects found under %s", cfg.input_root)
        return

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    predictor = SurfacePredictor(cfg)
    total_time = 0.0
    for subject_dir in tqdm(subjects, desc="Predicting subjects"):
        elapsed = predictor.predict(subject_dir, cfg.output_dir)
        total_time += elapsed

    LOGGER.info("Finished processing %d subjects.", len(subjects))
    if subjects:
        LOGGER.info("Average processing time: %.2f seconds", total_time / len(subjects))


if __name__ == "__main__":
    main()
