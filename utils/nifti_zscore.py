#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import nibabel as nib
import numpy as np
import os
import sys
import traceback

def zscore_normalize_data(data, mask=None, epsilon=1e-6, zero_out_background=True):
    """
    Performs Z-scoring on the input data.
    If data is 4D, Z-scoring is performed channel-wise (along the last dimension).

    Args:
        data (np.ndarray): Input image data (3D or 4D).
        mask (np.ndarray, optional): Boolean brain mask (3D). If provided,
                                     mean and std are computed from masked voxels.
        epsilon (float, optional): Small value to add to std dev if it's zero
                                   to prevent division by zero. Defaults to 1e-6.
        zero_out_background (bool, optional): If True and mask is provided,
                                              voxels outside the mask in the
                                              Z-scored data will be set to 0.
                                              Defaults to True.

    Returns:
        np.ndarray: Z-scored data.
    """
    if data.ndim == 3:
        # Process as a single 3D volume
        data_to_process = [data]
        is_4d = False
    elif data.ndim == 4:
        # Process each channel/volume in the 4th dimension
        data_to_process = [data[..., i] for i in range(data.shape[-1])]
        is_4d = True
        print(f"[INFO] Input is 4D, processing {len(data_to_process)} channels/volumes independently.")
    else:
        raise ValueError(f"Input data must be 3D or 4D, but got {data.ndim}D.")

    if mask is not None:
        if mask.ndim != 3:
            raise ValueError(f"Mask must be 3D, but got {mask.ndim}D.")
        if data_to_process[0].shape != mask.shape:
            raise ValueError(f"Shape of each 3D volume ({data_to_process[0].shape}) "
                             f"must match mask shape ({mask.shape}).")
        print("[INFO] Using provided mask for Z-scoring statistics.")
    else:
        print("[INFO] No mask provided. Z-scoring statistics will be computed from all finite voxels in each channel/volume.")

    zscored_channels = []
    for i, channel_data in enumerate(data_to_process):
        if is_4d:
            print(f"  Processing channel/volume {i+1}...")

        if mask is not None:
            masked_channel_data = channel_data[mask]
            if masked_channel_data.size == 0:
                print(f"  [WARNING] Masked region is empty for channel {i+1}. Skipping Z-scoring for this channel, returning original data or zeros.")
                zscored_channels.append(channel_data.copy()) # Or np.zeros_like(channel_data)
                continue
            mu = np.nanmean(masked_channel_data) # Use nanmean in case mask allows NaNs within it
            sigma = np.nanstd(masked_channel_data)
        else:
            finite_voxels = channel_data[np.isfinite(channel_data)]
            if finite_voxels.size == 0:
                print(f"  [WARNING] No finite voxels in channel {i+1}. Skipping Z-scoring for this channel, returning original data or zeros.")
                zscored_channels.append(channel_data.copy()) # Or np.zeros_like(channel_data)
                continue
            mu = np.mean(finite_voxels)
            sigma = np.std(finite_voxels)

        if sigma < epsilon:
            print(f"  [INFO] Standard deviation for channel {i+1} is very small ({sigma:.2e}). Setting to epsilon ({epsilon}) to avoid division by zero.")
            sigma = epsilon
        
        zscored_channel = (channel_data - mu) / sigma

        if mask is not None and zero_out_background:
            zscored_channel[~mask] = 0
            print(f"  [INFO] Voxels outside the mask in channel {i+1} have been set to 0 after Z-scoring.")
            
        zscored_channels.append(zscored_channel)
        print(f"  Channel {i+1}: Mean={mu:.4f}, Std={sigma:.4f} (used for Z-scoring)")


    if is_4d:
        return np.stack(zscored_channels, axis=-1)
    else:
        return zscored_channels[0]

def main():
    parser = argparse.ArgumentParser(description="Perform Z-scoring on a NIfTI image.")
    parser.add_argument("-i", "--input_file", type=str, required=True,
                        help="Path to the input NIfTI file.")
    parser.add_argument("-o", "--output_file", type=str, required=True,
                        help="Path to save the Z-scored NIfTI file.")
    parser.add_argument("-m", "--mask_file", type=str, default=None,
                        help="(Optional) Path to a 3D NIfTI brain mask file. If provided, "
                             "mean and std for Z-scoring will be computed from within this mask.")
    parser.add_argument("-e", "--epsilon", type=float, default=1e-6,
                        help="Small value to add to standard deviation if it's zero "
                             "to prevent division by zero. Default is 1e-6.")
    parser.add_argument("--no_zero_background", action="store_false", dest="zero_out_background",
                        help="If mask is used, do NOT set voxels outside the mask to 0 after Z-scoring. "
                             "By default, background is zeroed out.")
    parser.set_defaults(zero_out_background=True)

    args = parser.parse_args()

    print(f"--- Z-scoring NIfTI file ---")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    if args.mask_file:
        print(f"Mask file: {args.mask_file}")
        print(f"Zero out background (if mask used): {args.zero_out_background}")
    print(f"Epsilon for std dev: {args.epsilon}")
    print("-----------------------------")

    try:
        # 1. Load input NIfTI image
        if not os.path.exists(args.input_file):
            print(f"[ERROR] Input file not found: {args.input_file}")
            sys.exit(1)
        img_nifti = nib.load(args.input_file)
        img_data = img_nifti.get_fdata(dtype=np.float32) # Load as float32 for calculations
        print(f"[INFO] Loaded input image data with shape: {img_data.shape}")

        # 2. Load mask if provided
        mask_data_boolean = None
        if args.mask_file:
            if not os.path.exists(args.mask_file):
                print(f"[ERROR] Mask file not found: {args.mask_file}")
                sys.exit(1)
            mask_nifti = nib.load(args.mask_file)
            mask_data_boolean = mask_nifti.get_fdata().astype(bool) # Ensure boolean mask
            print(f"[INFO] Loaded mask data with shape: {mask_data_boolean.shape}")
            
            # Check if mask shape is compatible with 3D slices of image data
            img_3d_shape = img_data.shape[:3]
            if mask_data_boolean.shape != img_3d_shape:
                print(f"[ERROR] Mask shape {mask_data_boolean.shape} does not match "
                      f"image 3D slice shape {img_3d_shape}.")
                sys.exit(1)
        
        # 3. Perform Z-scoring
        zscored_data = zscore_normalize_data(img_data, 
                                             mask=mask_data_boolean, 
                                             epsilon=args.epsilon,
                                             zero_out_background=args.zero_out_background)
        
        # 4. Create new NIfTI image with Z-scored data
        # Preserve header and affine from original image
        # Ensure output data type is float32, which is common for processed images
        output_nifti = nib.Nifti1Image(zscored_data.astype(np.float32), img_nifti.affine, img_nifti.header)
        output_nifti.header.set_data_dtype(np.float32) # Explicitly set data type in header

        # 5. Save the output NIfTI image
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"[INFO] Created output directory: {output_dir}")
            
        nib.save(output_nifti, args.output_file)
        print(f"[SUCCESS] Z-scored image saved to: {args.output_file}")

    except Exception as e:
        print(f"[ERROR] An error occurred during Z-scoring: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()