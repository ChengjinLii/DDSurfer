"""Synthetic CPU tests for the unified CLI and affine-aware deployment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
import trimesh

from inference.coordinates import Geometry
from utils.files import sha256_file
from utils.surface_io import load_mesh
from inference.native import atlas_ras_to_native_ras
from inference.predict import SurfacePredictor, load_checksums, load_model, main, resolve_precision
import run_ddsurfer_pipeline as pipeline

ROOT = Path(__file__).resolve().parents[1]


class DummyModel(torch.nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta=delta
    def forward(self, vertices, volumes):
        return vertices+self.delta


class InferenceTests(unittest.TestCase):
    def test_pipeline_uses_unified_entry_and_current_interpreter(self):
        args=pipeline.parse_args(['--subject','100610'])
        command=pipeline.build_prediction_command(ROOT/'DDSurfer_predict.py',args,'both')
        self.assertEqual(Path(command[1]).name,'DDSurfer_predict.py')
        self.assertEqual(command[command.index('--surf_hemi')+1],'both')
        self.assertIn(str(ROOT/'weights'),command)

    def test_prediction_and_pipeline_share_default_weights_root(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'x').mkdir()
            with patch('inference.predict.SurfacePredictor') as predictor:
                main('both',['--input_root',temp,'--subjects','x','--device','cpu'])
        expected=pipeline.parse_args(['--subject','x']).checkpoint_root
        self.assertEqual(expected,ROOT/'weights')
        self.assertEqual(len(predictor.call_args_list),2)
        for call in predictor.call_args_list:
            self.assertEqual(call.args[0].checkpoint_root,expected)

    def test_custom_checkpoint_root_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'x').mkdir()
            folder=Path(temp)/'custom_weights'
            args=pipeline.parse_args(['--subject','x','--checkpoint-root',str(folder)])
            command=pipeline.build_prediction_command(ROOT/'DDSurfer_predict.py',args,'both')
            self.assertEqual(command[command.index('--checkpoint_root')+1],str(folder))
            with patch('inference.predict.SurfacePredictor') as predictor:
                main('left',['--input_root',temp,'--subjects','x','--device','cpu',
                             '--checkpoint_root',str(folder)])
            self.assertEqual(predictor.call_args.args[0].checkpoint_root,folder)

    def test_pipeline_calls_prediction_once_for_both_sides(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(pipeline,'run_command') as run, patch.object(pipeline,'finalize_outputs'):
                pipeline.main(['--subject','100610','--skip-preprocessing','--output-root',temp])
        names=[Path(call[0][0][1]).name for call in run.call_args_list]
        self.assertEqual(names,['export.py','DDSurfer_predict.py','native.py'])

    def test_native_conversion_command(self):
        args=pipeline.parse_args(['--subject','x'])
        command=pipeline.build_native_conversion_command(args)
        self.assertIn(str(ROOT/'inference/native.py'),command)
        self.assertEqual(Path(command[command.index('--output-dir')+1]),ROOT/'outputs/x/ddsurfer')

    def test_unified_left_right_both_dispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'x').mkdir()
            for option,expected in [('both',['left','right']),('left',['left']),('right',['right'])]:
                calls=[]
                class Predictor:
                    def __init__(self,args):calls.append(args.surf_hemi)
                    def predict(self,folder):pass
                with patch('inference.predict.SurfacePredictor',Predictor):
                    main('both',['--input_root',temp,'--subjects','x','--surf_hemi',option,'--device','cpu'])
                self.assertEqual(calls,expected)

    def test_template_coordinate_conversion(self):
        affine=np.eye(4);affine[:3,3]=[-88.299995,-129,-69]
        for hemisphere,offset in [('left',0),('right',64)]:
            geometry=Geometry(affine,(176,224,176),hemisphere)
            crop=np.array([[10.,70.,50.]])
            expected=crop+np.array([offset-88.299995,-129,-69])
            np.testing.assert_allclose(geometry.crop_to_ras(crop),expected)
            np.testing.assert_allclose(geometry.ras_to_crop(expected),crop)

    def test_native_transform_flips_ras_lps_twice_and_does_not_invert_pull_map(self):
        transform=sitk.TranslationTransform(3,(2.,-3.,.5))
        vertices=np.array([[10.,20.,30.],[-4.,8.,12.]])
        np.testing.assert_allclose(atlas_ras_to_native_ras(vertices,transform),vertices+[-2.,3.,.5])

    def test_cpu_auto_precision_and_bf16_rejection(self):
        self.assertEqual(resolve_precision('auto',torch.device('cpu'),'bf16'),'fp32')
        with self.assertRaises(ValueError):resolve_precision('bf16',torch.device('cpu'),'bf16')

    def test_pipeline_and_prediction_default_to_fp32_on_both_devices(self):
        for device in ('cpu', 'cuda:0'):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as temp:
                args = pipeline.parse_args(['--subject', 'x', '--device', device])
                self.assertEqual(args.precision, 'fp32')
                command = pipeline.build_prediction_command(ROOT/'DDSurfer_predict.py', args, 'both')
                self.assertEqual(command[command.index('--precision') + 1], 'fp32')
                (Path(temp)/'x').mkdir()
                with patch('inference.predict.SurfacePredictor') as predictor:
                    main('left', ['--input_root', temp, '--subjects', 'x', '--device', device])
                self.assertEqual(predictor.call_args.args[0].precision, 'fp32')

    def test_explicit_precision_overrides_are_preserved(self):
        for precision in ('bf16', 'fp32', 'auto'):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as temp:
                args = pipeline.parse_args(['--subject', 'x', '--precision', precision])
                command = pipeline.build_prediction_command(ROOT/'DDSurfer_predict.py', args, 'both')
                self.assertEqual(command[command.index('--precision') + 1], precision)
                (Path(temp)/'x').mkdir()
                with patch('inference.predict.SurfacePredictor') as predictor:
                    main('left', ['--input_root', temp, '--subjects', 'x', '--precision', precision])
                self.assertEqual(predictor.call_args.args[0].precision, precision)

    def test_bundled_auto_precision_is_fp32_without_bf16_support(self):
        config = json.loads((ROOT/'weights/manifest.json').read_text())
        self.assertEqual(config['precision'], 'fp32')
        with patch('torch.cuda.is_bf16_supported', return_value=False):
            for device in ('cpu', 'cuda:0'):
                self.assertEqual(resolve_precision('auto', torch.device(device), config['precision']), 'fp32')

    def test_cuda_precision_requires_supported_bf16_or_explicit_fp32(self):
        device = torch.device('cuda:0')
        with patch('torch.cuda.is_bf16_supported', return_value=True):
            self.assertEqual(resolve_precision('auto', device, 'bf16'), 'bf16')
        with patch('torch.cuda.is_bf16_supported', return_value=False):
            self.assertEqual(resolve_precision('fp32', device, 'bf16'), 'fp32')
            with self.assertRaisesRegex(ValueError, 'choose fp32'):
                resolve_precision('auto', device, 'bf16')

    def test_prediction_chains_wm_to_pial_and_exports_physical_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            predictor=SurfacePredictor.__new__(SurfacePredictor)
            predictor.device=torch.device('cpu');predictor.precision='fp32'
            affine=np.eye(4);affine[:3,3]=[-88.299995,-129,-69]
            predictor.geometry=Geometry(affine,(176,224,176),'right')
            predictor.vertices=np.array([[10.,20.,30.],[11.,20.,30.],[10.,21.,30.]],np.float32)
            predictor.faces=np.array([[0,1,2]],np.int64)
            predictor.args=argparse.Namespace(output_dir=Path(temp),surf_hemi='right',save_debug=True)
            entry=dict(postprocess_iters=0,checkpoint_file='dummy.pt',sha256='test',template_output_sha256='template')
            predictor.entries={'white':dict(entry),'pial':dict(entry)}
            predictor.models={'white':DummyModel(.5),'pial':DummyModel(1.)}
            predictor.load_volumes=lambda folder:torch.zeros(1,5,2,2,2)
            predictor.predict(Path(temp)/'x')
            directory=Path(temp)/'mni/x'
            wm=np.load(directory/'right_wm_crop_voxel.npz')['postprocessed']
            pial=np.load(directory/'right_pial_crop_voxel.npz')['postprocessed']
            np.testing.assert_allclose(wm,predictor.vertices+.5)
            np.testing.assert_allclose(pial,wm+1.)
            mesh=load_mesh(directory/'x_predicted_pial_surface_right.obj')
            np.testing.assert_allclose(mesh.vertices,predictor.geometry.crop_to_ras(pial),atol=1e-5)

    def test_deployment_checksums_and_tensor_only_weights(self):
        folder=ROOT/'weights';config=json.loads((folder/'manifest.json').read_text())
        checksums=load_checksums(folder)
        self.assertEqual(set(config),{'architecture','geometry','precision','postprocess_iters',
                                      'checkpoint_files','templates'})
        for hemisphere,files in config['checkpoint_files'].items():
            template=config['templates'][hemisphere]
            self.assertEqual(sha256_file(ROOT/'template'/template['file']),template['sha256'])
            for name in files.values():
                self.assertEqual(sha256_file(folder/name),checksums[name])
                state=torch.load(folder/name,map_location='cpu',weights_only=True)
                self.assertTrue(state)
                self.assertTrue(all(isinstance(value,torch.Tensor) for value in state.values()))
                model=load_model(folder/name,checksums[name],config['architecture'],torch.device('cpu'))
                self.assertFalse(model.training)
                for key,value in state.items():self.assertTrue(torch.equal(value,model.state_dict()[key]))
                del state,model

    def test_reject_corrupt_checkpoint(self):
        folder=ROOT/'weights';config=json.loads((folder/'manifest.json').read_text())
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):
            load_model(folder/'ddsurfer_lh_wm.pt','0'*64,config['architecture'],torch.device('cpu'))

    def test_checkpoint_root_is_flat(self):
        args=pipeline.parse_args(['--subject','100610'])
        self.assertEqual(args.checkpoint_root,ROOT/'weights')
        self.assertFalse((ROOT/'weights/hcp').exists())
        self.assertFalse(any(p.is_dir() for p in (ROOT/'weights').iterdir()))
        self.assertFalse(any(p.is_dir() for p in (ROOT/'template').iterdir()))
        self.assertFalse(list((ROOT/'template').glob('*.stl')))
        self.assertEqual(len(list((ROOT/'template').glob('*.obj'))), 2)
        self.assertFalse((ROOT/'template/hcp_hemi-left_init_160k.obj').exists())
        self.assertFalse((ROOT/'template/hcp_hemi-right_init_160k.obj').exists())
        self.assertTrue((ROOT/'template/100HCP-population-mean-T2-1mm.nii.gz').is_file())

    def test_missing_manifest_reports_requested_root(self):
        with tempfile.TemporaryDirectory() as temp:
            args=argparse.Namespace(device='cpu',checkpoint_root=Path(temp))
            with self.assertRaisesRegex(FileNotFoundError,str(Path(temp)/'manifest.json')):
                SurfacePredictor(args)

    def test_relative_dti_paths_create_valid_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            cwd=Path(temp)
            inputs=cwd/'raw/x/T1w/Diffusion';inputs.mkdir(parents=True)
            nib.save(nib.Nifti1Image(np.ones((3,4,5,7),np.float32),np.eye(4)), inputs/'data.nii.gz')
            nib.save(nib.Nifti1Image(np.ones((3,4,5),np.uint8),np.eye(4)), inputs/'nodif_brain_mask.nii.gz')
            np.savetxt(inputs/'bvals', np.array([[0,1000,1000,1000,1000,1000,1000]]))
            vectors=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]],float)
            np.savetxt(inputs/'bvecs', vectors.T)
            slicer=cwd/'slicer'
            core=slicer/'lib/Slicer-5.2/cli-modules';core.mkdir(parents=True)
            dmri=slicer/'extensions/SlicerDMRI/lib/Slicer-5.2/cli-modules';dmri.mkdir(parents=True)
            for path in (slicer/'Slicer',core/'BRAINSFit',core/'ResampleScalarVectorDWIVolume',
                         dmri/'DWIToDTIEstimation',dmri/'DiffusionTensorScalarMeasurements'):
                path.write_text('#!/bin/sh\nexit 0\n');path.chmod(0o755)
            helper=cwd/'python-helper'
            helper.write_text('#!/bin/sh\nif [ "$(basename "$1")" = inputs.py ]; then\n'
                              f'  exec "{sys.executable}" "$@"\nfi\nexit 17\n');helper.chmod(0o755)
            env=dict(os.environ,SLICER_PATH='slicer')
            env.pop('REFERENCE_IMAGE',None)
            result=subprocess.run(['bash',str(ROOT/'preprocessing/dti.sh'),
                                   '--subject','x','--input-root','raw','--output-root','out',
                                   '--python-bin','./python-helper'],cwd=cwd,env=env,
                                  stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            self.assertEqual(result.returncode,17,result.stdout)
            for name,target in (('data.nii.gz','x-dwi.nii.gz'),('nodif_brain_mask.nii.gz','x-mask-input.nii.gz')):
                link=cwd/'out/x'/target
                self.assertTrue(link.is_symlink())
                self.assertEqual(link.resolve(strict=True),inputs/name)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
