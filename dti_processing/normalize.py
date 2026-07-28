import argparse
import nibabel as nib
import numpy as np
import scipy.stats as stats
import os

def main():
    # -----------------
    # Parse arguments
    # -----------------
    parser = argparse.ArgumentParser(
        description="",
        epilog="")
    parser.add_argument("-v", "--version",
                        action="version", default=argparse.SUPPRESS,
                        version='1.0',
                        help="Show program's version number and exit")
    parser.add_argument(
        '--input', default="", help='input parameter')
    parser.add_argument(
        '--mask', default="", help='mask name.')
    parser.add_argument(
        '--output', default="",help='output file.')
    parser.add_argument(
        '--flip', type=int, default=1, help='flip mask')

    args = parser.parse_args()

    # args.input = '/rfanfs/pnl-zorro/home/fz040/Projects/DeepSurfer/PPMIdata/3104_S103319_2011/3104_S103319_2011-dti-FractionalAnisotropy.nii.gz'
    # args.mask = '/rfanfs/pnl-zorro/home/fz040/Projects/DeepSurfer/PPMIdata/3104_S103319_2011/3104_S103319_2011-dti-FractionalAnisotropy.nii.gz'
    # args.label = '/rfanfs/pnl-zorro/home/fz040/Projects/DeepSurfer/PPMIdata/3104_S103319_2011/3104_S103319_2011_wmparc_T1w.nii.gz'
    # args.output = '/rfanfs/pnl-zorro/home/fz040/Projects/DeepSurfer/PPMIdata/3104_S103319_2011/3104_S103319_2011-dti-FractionalAnisotropy-NormMasked.nii.gz'

    img = nib.load(args.input)
    img_data = img.get_fdata()
    img_affine = img.affine
    img_header = img.header

    if not os.path.exists(args.mask):
        args.mask = args.input

    mask = nib.load(args.mask)
    mask_data = mask.get_fdata()
    if args.flip == 1:
        mask_data = np.flipud(mask_data) # HCP data is needed
    if args.flip == 2:
        mask_data = np.fliplr(mask_data) # test-test

    mask_data[mask_data > 0] = 1
    mask_data[mask_data <= 0] = 0

    img_data[mask_data == 0] = np.nan

    if args.input.find("dti-MaxEigenvalue") > 0 or args.input.find("dti-MidEigenvalue") > 0 or args.input.find("dti-MinEigenvalue") > 0:
        print("%d voxels are outside the expected range." % (np.sum(img_data[mask_data == 1] > 0.004) + np.sum(img_data[mask_data == 1] < 0.0)))
        img_data[mask_data == 1] = np.clip(img_data[mask_data == 1], a_min=0, a_max=0.004)

    elif args.input.find("dti-Trace") > 0:
        print("%d voxels are outside the expected range." % (np.sum(img_data[mask_data == 1] > 0.012) + np.sum(img_data[mask_data == 1] < 0.0)))
        img_data[mask_data == 1] = np.clip(img_data[mask_data == 1], a_min=0, a_max=0.012)

    elif args.input.find("dti-FractionalAnisotropy") > 0:
        print("%d voxels are outside the expected range." % (np.sum(img_data[mask_data == 1] > 1.0) + np.sum(img_data[mask_data == 1] < 0.0)))
        img_data[mask_data == 1] = np.clip(img_data[mask_data == 1], a_min=0, a_max=1)

    img_data[mask_data == 1] = stats.zscore(img_data[mask_data == 1])
    img_data[mask_data == 0] = -4.0

    print("z score: %f - %f " % (np.nanmin(img_data), np.nanmax(img_data)))

    NormMasked = nib.Nifti1Image(img_data, affine=img_affine, header=img_header)
    nib.save(NormMasked, args.output)

if __name__ == '__main__':
    main()
