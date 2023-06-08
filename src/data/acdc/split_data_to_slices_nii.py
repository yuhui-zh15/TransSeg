import os
import sys
from shutil import copyfile
from PIL import Image
import cv2
import nibabel as nib
import numpy as np
import json

train_filenames = json.load(open("ACDC_dataset.json"))["training"]
train_filenames = [
    name["image"].split("/")[-1].replace("imagesTr", "training")
    for name in train_filenames
]
test_filenames = json.load(open("ACDC_dataset.json"))["test"]
test_filenames = [name.split("/")[-1] for name in test_filenames]

leaderboard_filenames = []
for dirpath, dirnames, filenames in os.walk("testing"):
    for filename in [f for f in filenames if f.endswith(".nii.gz") and "frame" in f]:
        leaderboard_filenames.append(filename)

print(len(train_filenames), len(test_filenames), len(leaderboard_filenames))
print(test_filenames, leaderboard_filenames)
input()

basedir = "./training/"
outputdir = f"processed"


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path


all_filenames = train_filenames + test_filenames
json_metadata = {"training": [], "local_test": []}
for i, filename in enumerate(all_filenames):
    if filename in train_filenames:
        splitdir = "training"
    elif filename in test_filenames:
        splitdir = "local_test"
    else:
        raise

    filename_noext = filename.split(".")[0]
    patient_id = filename.split("_")[0]

    img = nib.load(f"{basedir}/{patient_id}/{filename}").get_data()
    label = nib.load(
        f'{basedir}/{patient_id}/{filename.replace(".nii.gz", "_gt.nii.gz")}'
    ).get_data()

    n_slices = img.shape[-1]

    img = np.array(
        [
            cv2.resize(img[:, :, i], (512, 512), interpolation=cv2.INTER_LANCZOS4)
            for i in range(n_slices)
        ]
    ).transpose(1, 2, 0)
    label = np.array(
        [
            cv2.resize(label[:, :, i], (512, 512), interpolation=cv2.INTER_NEAREST)
            for i in range(n_slices)
        ]
    ).transpose(1, 2, 0)

    print(
        splitdir,
        filename,
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
                f"{outputdir}/images/{splitdir}/{filename_noext}_{slice_idx:03d}.nii.gz"
            ),
        )

        label_new = nib.Nifti1Image(label_slices, np.eye(4))
        nib.save(
            label_new,
            ensure_dir(
                f"{outputdir}/annotations/{splitdir}/{filename_noext}_{slice_idx:03d}.nii.gz"
            ),
        )

        json_metadata[splitdir].append(
            {
                "image": f'images/{splitdir}/{filename.replace(".nii.gz", "")}_{slice_idx:03d}.nii.gz',
                "label": f"annotations/{splitdir}/{filename_noext}_{slice_idx:03d}.nii.gz",
            }
        )

basedir = "./testing/"
all_filenames = leaderboard_filenames
json_metadata["test"] = []
file2size = {}
for i, filename in enumerate(all_filenames):
    splitdir = "test"

    filename_noext = filename.split(".")[0]
    patient_id = filename.split("_")[0]

    img = nib.load(f"{basedir}/{patient_id}/{filename}").get_data()
    h, w, d = img.shape[0], img.shape[1], img.shape[2]
    file2size[filename_noext] = (h, w, d)

    n_slices = img.shape[-1]

    img = np.array(
        [
            cv2.resize(img[:, :, i], (512, 512), interpolation=cv2.INTER_LANCZOS4)
            for i in range(n_slices)
        ]
    ).transpose(1, 2, 0)

    print(splitdir, filename, img.shape, (img.mean(), img.std(), img.min(), img.max()))

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
                f"{outputdir}/images/{splitdir}/{filename_noext}_{slice_idx:03d}.nii.gz"
            ),
        )

        json_metadata[splitdir].append(
            f'images/{splitdir}/{filename.replace(".nii.gz", "")}_{slice_idx:03d}.nii.gz'
        )

json.dump(file2size, open("file2size.json", "w"))
json_metadata["labels"] = {
    "0": "background",
    "1": "the right ventricular cavity",
    "2": "myocardium",
    "3": "the left ventricular cavity",
}
json_metadata["validation"] = json_metadata["local_test"]
del json_metadata["local_test"]
json.dump(json_metadata, open(f"{outputdir}/dataset_5slices.json", "w"))
