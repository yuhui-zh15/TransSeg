import os
import sys
from shutil import copyfile
from PIL import Image
import cv2
import nibabel as nib
import numpy as np
import json

N = 30

val_slc = [1, 2, 3, 4, 8, 22, 25, 29, 32, 35, 36, 38]

basedir = "RawData/Training/"
outputdir = f"processed"
file_idxs = list(range(1, 11)) + list(range(21, 41))


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path


print("# Data:", len(file_idxs))


json_metadata = {"training": [], "validation": []}
for i, idx in enumerate(file_idxs):
    if idx not in val_slc:
        splitdir = "training"
    elif idx in val_slc:
        splitdir = "validation"
    else:
        raise

    img = nib.load(f"{basedir}/img/img{idx:04d}.nii.gz").get_data()
    label = nib.load(f"{basedir}/label/label{idx:04d}.nii.gz").get_data()
    n_slices = img.shape[-1]
    # img = np.array([cv2.resize(img[:, :, i], (224, 224), interpolation=cv2.INTER_LANCZOS4) for i in range(n_slices)]).transpose(1, 2, 0)
    # label = np.array([cv2.resize(label[:, :, i], (224, 224), interpolation=cv2.INTER_NEAREST) for i in range(n_slices)]).transpose(1, 2, 0)

    print(
        splitdir,
        i,
        idx,
        img.shape,
        label.shape,
        (img.mean(), img.std(), img.min(), img.max()),
        (label.mean(), label.std(), label.min(), label.max()),
    )
    for slice_idx in range(n_slices):
        slice_idxs = [
            max(slice_idx - 2, 0),
            max(slice_idx - 1, 0),
            slice_idx,
            min(slice_idx + 1, n_slices - 1),
            min(slice_idx + 2, n_slices - 1),
        ]
        img_slices = img[:, :, slice_idxs]
        label_slices = label[:, :, slice_idxs]

        img_new = nib.Nifti1Image(img_slices, np.eye(4))
        nib.save(
            img_new,
            ensure_dir(
                f"{outputdir}/images/{splitdir}/img{idx:04d}_{slice_idx:03d}.nii.gz"
            ),
        )

        label_new = nib.Nifti1Image(label_slices, np.eye(4))
        nib.save(
            label_new,
            ensure_dir(
                f"{outputdir}/annotations/{splitdir}/label{idx:04d}_{slice_idx:03d}.nii.gz"
            ),
        )
        json_metadata[splitdir].append(
            {
                "image": f"images/{splitdir}/img{idx:04d}_{slice_idx:03d}.nii.gz",
                "label": f"annotations/{splitdir}/label{idx:04d}_{slice_idx:03d}.nii.gz",
            }
        )

basedir = "RawData/Testing/"
file_idxs = list(range(61, 81))

print("# Data:", len(file_idxs))


json_metadata["test"] = []
for i, idx in enumerate(file_idxs):
    splitdir = "test"

    img = nib.load(f"{basedir}/img/img{idx:04d}.nii.gz").get_data()
    n_slices = img.shape[-1]

    print(splitdir, i, idx, img.shape, (img.mean(), img.std(), img.min(), img.max()))
    for slice_idx in range(n_slices):

        slice_idxs = [
            max(slice_idx - 2, 0),
            max(slice_idx - 1, 0),
            slice_idx,
            min(slice_idx + 1, n_slices - 1),
            min(slice_idx + 2, n_slices - 1),
        ]
        img_slices = img[:, :, slice_idxs]

        img_new = nib.Nifti1Image(img_slices, np.eye(4))
        nib.save(
            img_new,
            ensure_dir(
                f"{outputdir}/images/{splitdir}/img{idx:04d}_{slice_idx:03d}.nii.gz"
            ),
        )

        json_metadata[splitdir].append(
            f"images/{splitdir}/img{idx:04d}_{slice_idx:03d}.nii.gz"
        )

json_metadata["labels"] = {
    "0": "background",
    "1": "spleen",
    "2": "rkid",
    "3": "lkid",
    "4": "gall",
    "5": "eso",
    "6": "liver",
    "7": "sto",
    "8": "aorta",
    "9": "IVC",
    "10": "veins",
    "11": "pancreas",
    "12": "rad",
    "13": "lad",
}
json.dump(json_metadata, open(f"{outputdir}/dataset_5slices.json", "w"))
