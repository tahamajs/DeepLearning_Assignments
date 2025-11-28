import numpy as np
import nibabel as nib
import warnings
import copy
import torch
from torch.utils.data import Dataset


def get_slice_data(volume_data, seg_data, axis, slice_idx):

    if axis == 0:
        vol_slice = volume_data[slice_idx, :, :]
        seg_slice = seg_data[slice_idx, :, :]
    elif axis == 1:
        vol_slice = volume_data[:, slice_idx, :]
        seg_slice = seg_data[:, slice_idx, :]
    else:
        vol_slice = volume_data[:, :, slice_idx]
        seg_slice = seg_data[:, :, slice_idx]

    vol_slice = np.squeeze(vol_slice)
    seg_slice = np.squeeze(seg_slice)

    vol_slice_rotated = np.rot90(vol_slice, k=1)
    seg_slice_rotated = np.rot90(seg_slice, k=1)

    return vol_slice_rotated, seg_slice_rotated


def pad_slice(vol_slice, seg_slice, target_dims=(256, 256)):

    h, w = vol_slice.shape
    th, tw = target_dims

    pad_h_top = (th - h) // 2
    pad_h_bottom = th - h - pad_h_top
    pad_w_left = (tw - w) // 2
    pad_w_right = tw - w - pad_w_left

    padding = (
        (max(0, pad_h_top), max(0, pad_h_bottom)),
        (max(0, pad_w_left), max(0, pad_w_right)),
    )

    vol_padded = np.pad(vol_slice, padding, mode="constant", constant_values=0)
    seg_padded = np.pad(seg_slice, padding, mode="constant", constant_values=0)

    vol_padded = vol_padded[:th, :tw]
    seg_padded = seg_padded[:th, :tw]

    return vol_padded, seg_padded


class IBSRPatchDataset(Dataset):

    def __init__(self, volume_files, segmentation_files):
        self.patches = []
        self.masks = []
        self.patch_size = 128
        self.target_dims = (256, 256)
        self.slice_start = 10
        self.slice_stride = 3
        self.num_slices_to_extract = 48

        print("Loading and processing dataset (with rotation)...")
        for vol_path, seg_path in zip(volume_files, segmentation_files):
            try:
                vol_img = nib.load(vol_path)
                seg_img = nib.load(seg_path)

                vol_data = vol_img.get_fdata()
                seg_data = seg_img.get_fdata()

                for axis in range(3):
                    num_slices_in_axis = vol_data.shape[axis]

                    slice_indices = list(
                        range(self.slice_start, num_slices_in_axis, self.slice_stride)
                    )[: self.num_slices_to_extract]

                    for slice_idx in slice_indices:
                        vol_slice_rotated, seg_slice_rotated = get_slice_data(
                            vol_data, seg_data, axis, slice_idx
                        )

                        vol_padded, seg_padded = pad_slice(
                            vol_slice_rotated, seg_slice_rotated, self.target_dims
                        )

                        for x in [0, self.patch_size]:
                            for y in [0, self.patch_size]:
                                vol_patch = vol_padded[
                                    x : x + self.patch_size, y : y + self.patch_size
                                ]
                                seg_patch = seg_padded[
                                    x : x + self.patch_size, y : y + self.patch_size
                                ]

                                self.patches.append(vol_patch)
                                self.masks.append(seg_patch)
            except Exception as e:
                warnings.warn(f"Warning: Skipping file {vol_path} due to error: {e}")

        print(f"Dataset loaded. Total patches: {len(self.patches)}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        vol_patch = self.patches[idx].astype(np.float32)
        seg_patch = self.masks[idx].astype(np.int64)

        # Binarize segmentation mask: convert any non-zero value to 1
        # This ensures masks are binary (0 = background, 1 = foreground)
        # which is required for 2-class segmentation and prevents CUDA errors
        seg_patch = (seg_patch > 0).astype(np.int64)

        vol_patch_normalized = vol_patch

        vol_tensor = torch.tensor(vol_patch_normalized, dtype=torch.float32).unsqueeze(
            0
        )
        mask_tensor = torch.tensor(seg_patch, dtype=torch.long)

        return vol_tensor, mask_tensor
