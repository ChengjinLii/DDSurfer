"""Convert scanner RAS surfaces into FreeSurfer surface RAS coordinates."""
from pathlib import Path
from itertools import product

import nibabel as nib
from nibabel.freesurfer.mghformat import MGHHeader
from nibabel.processing import resample_from_to
import numpy as np
import SimpleITK as sitk
import trimesh

from inference.meshes import index_triangles, load_mesh

LIA_DIRECTIONS = np.array([[-1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])


def volume_files(path):
    path = Path(path).resolve()
    files = [path]
    if path.suffix.lower() == '.nhdr':
        entry = next((line.split(':', 1)[1].strip() for line in path.read_text().splitlines()
                      if line.lower().startswith('data file:')), None)
        if not entry or entry.upper().startswith('LIST') or '%' in entry:
            raise ValueError('Expected an NHDR header referencing one data file')
        files.append((path.parent / entry).resolve())
    return files


def load_volume(path):
    path = Path(path)
    if path.suffix.lower() in ('.nhdr', '.nrrd'):
        image = sitk.ReadImage(str(path))
        if image.GetDimension() != 3:
            raise ValueError('The native reference must be a scalar 3D image')
        data = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
        affine = np.eye(4)
        flip = np.diag([-1., -1., 1.])
        affine[:3, :3] = flip @ np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
        affine[:3, 3] = flip @ np.asarray(image.GetOrigin())
    else:
        image = nib.load(str(path))
        data = np.asarray(image.dataobj)
        affine = image.affine
        if hasattr(image, 'get_qform'):
            q, qc = image.get_qform(coded=True)
            s, sc = image.get_sform(coded=True)
            if qc and sc and not np.allclose(q, s, atol=1e-4):
                raise ValueError('Native reference qform and sform disagree')
    if data.ndim != 3 or not np.isfinite(data).all() or not np.isfinite(affine).all():
        raise ValueError('The native reference must be a finite scalar 3D image')
    if abs(np.linalg.det(affine[:3, :3])) < 1e-8:
        raise ValueError('Singular reference affine')
    return data, affine


def prepare_brain(source, destination):
    data, affine = load_volume(source)
    positive = data[data > 0]
    if not positive.size:
        raise ValueError('Native reference contains no positive intensities')
    high = float(np.percentile(positive, 99.5))
    scaled = np.clip(data / high * 255., 0., 255.).astype(np.uint8)
    source_image = nib.Nifti1Image(scaled, affine)
    orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(affine),
                                                  nib.orientations.axcodes2ornt('LIA'))
    aligned = source_image.as_reoriented(orientation)
    directions = LIA_DIRECTIONS
    spacing = np.linalg.norm(aligned.affine[:3, :3], axis=0)
    canonical = directions @ np.diag(spacing)
    # Surface registration assumes anatomical RAS axes. For axis-aligned inputs
    # LIA needs only a lossless permutation/flip, not interpolation or registration.
    if not np.allclose(aligned.affine[:3, :3], canonical, atol=1e-5):
        corners = np.array(list(product(*[(-.5, size - .5) for size in source_image.shape])))
        physical = nib.affines.apply_affine(affine, corners)
        local = physical @ directions
        low, high = local.min(axis=0), local.max(axis=0)
        shape = np.ceil((high - low) / spacing).astype(int)
        target = np.eye(4)
        target[:3, :3] = canonical
        target[:3, 3] = directions @ (low + spacing / 2.)
        aligned = resample_from_to(source_image, (tuple(shape), target), order=1)
    image = nib.MGHImage(np.asarray(aligned.dataobj, dtype=np.uint8), aligned.affine)
    nib.save(image, str(destination))
    result = nib.load(str(destination))
    if not np.allclose(result.affine, aligned.affine, atol=1e-4):
        raise ValueError('Writing brain.mgz changed its physical geometry')
    if not np.allclose(result.header.get_vox2ras_tkr()[:3, :3], result.affine[:3, :3], atol=1e-5):
        raise ValueError('FreeSurfer reference axes must match anatomical RAS axes')
    return result


def prepare_surface_reference(meshes, destination):
    """Create a geometry-only LIA reference; no MRI intensity is synthesized."""
    low = np.min([mesh.bounds[0] for mesh in meshes], axis=0) - 10.
    high = np.max([mesh.bounds[1] for mesh in meshes], axis=0) + 10.
    corners = np.array(list(product(*zip(low, high)))) @ LIA_DIRECTIONS
    origin = np.floor(corners.min(axis=0))
    shape = (np.ceil(corners.max(axis=0)) - origin + 1).astype(int)
    if np.any(shape < 2) or np.any(shape > 512):
        raise ValueError('Unexpected surface bounds; provide native scanner RAS coordinates in millimetres')
    affine = np.eye(4)
    affine[:3, :3] = LIA_DIRECTIONS
    affine[:3, 3] = LIA_DIRECTIONS @ origin
    image = nib.MGHImage(np.zeros(tuple(shape), dtype=np.uint8), affine)
    nib.save(image, str(destination))
    return nib.load(str(destination))


def volume_info(image):
    affine = image.header.get_vox2ras()
    spacing = np.asarray(image.header.get_zooms()[:3])
    axes = affine[:3, :3] / spacing
    return dict(head=np.array([2, 0, 20]), valid='1', filename='brain.mgz',
                volume=np.array(image.shape[:3]), voxelsize=spacing,
                xras=axes[:, 0], yras=axes[:, 1], zras=axes[:, 2],
                cras=nib.affines.apply_affine(affine, np.asarray(image.shape[:3]) / 2.))


def load_surface(path):
    path = Path(path)
    if path.suffix.lower() in ('.obj', '.ply', '.stl', '.off'):
        return load_mesh(path, indexed=False)
    vertices, faces, info = nib.freesurfer.read_geometry(str(path), read_metadata=True)
    required = ('volume', 'voxelsize', 'xras', 'yras', 'zras', 'cras')
    if not all(key in info for key in required) or not str(info.get('valid', '')).startswith('1'):
        raise ValueError('FreeSurfer input needs valid volume geometry; export to scanner-RAS OBJ with mris_convert --to-scanner')
    affine = np.eye(4)
    affine[:3, :3] = np.column_stack([info['xras'], info['yras'], info['zras']]) @ np.diag(info['voxelsize'])
    affine[:3, 3] = info['cras'] - affine[:3, :3] @ (np.asarray(info['volume']) / 2.)
    header = MGHHeader()
    header.set_data_shape(tuple(info['volume']))
    header.set_zooms(tuple(info['voxelsize']))
    vertices = nib.affines.apply_affine(affine @ np.linalg.inv(header.get_vox2ras_tkr()), vertices)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def load_pair(white_path, pial_path):
    paths = [Path(white_path), Path(pial_path)]
    meshes = [load_surface(path) for path in paths]
    for mesh in meshes:
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError('Expected a single triangular surface')
        if not np.isfinite(mesh.vertices).all():
            raise ValueError('Nonfinite surface vertices')
    white, pial = meshes
    if any(path.suffix.lower() == '.stl' for path in paths):
        if not all(path.suffix.lower() == '.stl' for path in paths) or white.faces.shape != pial.faces.shape:
            raise ValueError('STL white/pial inputs must both be STL with corresponding triangle order')
        pial_points = pial.vertices[pial.faces].reshape(-1, 3)
        # STL duplicates points per triangle. Weld both surfaces with the same
        # white-surface mapping instead of independently reindexing their vertices.
        white, first, inverse = index_triangles(white.vertices[white.faces])
        if not np.allclose(pial_points, pial_points[first][inverse], atol=1e-4, rtol=0):
            raise ValueError('STL triangle correspondence is inconsistent; use indexed OBJ or FreeSurfer surfaces')
        pial = trimesh.Trimesh(vertices=pial_points[first], faces=white.faces, process=False)
    if white.vertices.shape != pial.vertices.shape or not np.array_equal(white.faces, pial.faces):
        raise ValueError('White and pial surfaces must have matching vertex indices and faces')
    if not white.is_watertight or not pial.is_watertight:
        raise ValueError('Cortical surfaces must be closed triangular meshes')
    return white, pial


def write_surface(mesh, image, destination):
    scanner = image.header.get_vox2ras()
    tkr = image.header.get_vox2ras_tkr()
    scanner_to_surface = tkr @ np.linalg.inv(scanner)
    vertices = nib.affines.apply_affine(scanner_to_surface, mesh.vertices)
    # An orientation change can reflect axes; retain outward-facing triangle winding.
    reflected = np.linalg.det(scanner_to_surface[:3, :3]) < 0
    surface_faces = mesh.faces[:, ::-1] if reflected else mesh.faces
    voxels = nib.affines.apply_affine(np.linalg.inv(scanner), mesh.vertices)
    inside = np.all((voxels >= -.5) & (voxels <= np.array(image.shape[:3]) - .5), axis=1)
    if inside.mean() < .99:
        raise ValueError('Native surface falls outside the native MRI; check space and transform direction')
    nib.freesurfer.write_geometry(str(destination), vertices, surface_faces, volume_info=volume_info(image))
    stored, faces, info = nib.freesurfer.read_geometry(str(destination), read_metadata=True)
    restored = nib.affines.apply_affine(scanner @ np.linalg.inv(tkr), stored)
    error = float(np.max(np.abs(restored - mesh.vertices)))
    if error > 1e-4 or not np.array_equal(faces, surface_faces):
        raise ValueError('FreeSurfer surface coordinate round trip failed')
    return dict(scanner_RAS_roundtrip_max_error_mm=error, fraction_inside_reference=float(inside.mean()),
                vertices=len(vertices), faces=len(faces), triangle_winding_reversed=bool(reflected))


def vertex_area(vertices, faces):
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1) / 2.
    values = np.zeros(len(vertices), dtype=np.float64)
    for column in range(3):
        np.add.at(values, faces[:, column], area / 3.)
    return values.astype(np.float32)
