# Surface Utilities

These standalone helpers are not used by the main DDSurfer pipeline:

- `obj_to_stl.py`: convert OBJ to STL.
- `stl_to_obj.py`: convert STL to OBJ.
- `translate_mesh.py`: apply a fixed mesh translation.

Use `python tools/<script>.py --help` from the repository root for arguments.
Format conversion can change vertex indices. It is not a substitute for
checking white/pial vertex correspondence. Fixed translation is not a
registration or an affine-aware coordinate conversion.
