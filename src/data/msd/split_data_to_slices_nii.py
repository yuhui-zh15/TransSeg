import os
import sys
import nibabel as nib
import numpy as np
import json
import random
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

random.seed(1234)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    return file_path


def process(filename, inputdir):
    split = filename["split"]
    filename_noext = filename["image"].split("/")[-1].replace(".nii.gz", "")

    img = nib.load(f"{inputdir}/{filename['image']}").get_fdata()
    label = (
        nib.load(f"{inputdir}/{filename['label']}").get_fdata()
        if filename["label"] is not None
        else None
    )
    n_dim = len(img.shape)
    assert n_dim in [3, 4]
    n_slices = img.shape[-1] if n_dim == 3 else img.shape[-2]

    print(
        split,
        filename,
        img.shape,
        label.shape if label is not None else "-",
        (img.mean(), img.std(), img.min(), img.max()),
        (label.mean(), label.std(), label.min(), label.max())
        if label is not None
        else "-",
    )

    if n_dim == 4:
        img = img.transpose(3, 0, 1, 2)

    for slice_idx in range(n_slices):
        slice_idxs = [
            max(slice_idx - 2, 0),
            max(slice_idx - 1, 0),
            slice_idx,
            min(slice_idx + 1, n_slices - 1),
            min(slice_idx + 2, n_slices - 1),
        ]

        img_slices = img[..., slice_idxs]

        img_new = nib.Nifti1Image(img_slices, np.eye(4))
        nib.save(
            img_new,
            ensure_dir(
                f"processed/{inputdir}/images/{split}/{filename_noext}_{slice_idx:03d}.nii.gz"
            ),
        )

        if label is not None:
            label_slices = label[..., slice_idxs]
            label_new = nib.Nifti1Image(label_slices, np.eye(4))
            nib.save(
                label_new,
                ensure_dir(
                    f"processed/{inputdir}/annotations/{split}/{filename_noext}_{slice_idx:03d}.nii.gz"
                ),
            )


def process_metadata(filename, inputdir, json_metadata):
    split = filename["split"]
    filename_noext = filename["image"].split("/")[-1].replace(".nii.gz", "")

    img = nib.load(f"{inputdir}/{filename['image']}").get_fdata()

    n_dim = len(img.shape)
    assert n_dim in [3, 4]
    n_slices = img.shape[-1] if n_dim == 3 else img.shape[-2]

    for slice_idx in range(n_slices):
        json_metadata[split].append(
            {
                "image": f"images/{split}/{filename_noext}_{slice_idx:03d}.nii.gz",
                "label": f"annotations/{split}/{filename_noext}_{slice_idx:03d}.nii.gz",
            }
            if filename["label"] is not None
            else f"images/{split}/{filename_noext}_{slice_idx:03d}.nii.gz"
        )


def main():
    random.seed(1234)

    for inputdir in sorted(
        [
            "Task01_BrainTumour",
            "Task03_Liver",
            "Task05_Prostate",
            "Task07_Pancreas",
            "Task09_Spleen",
            "Task02_Heart",
            "Task04_Hippocampus",
            "Task06_Lung",
            "Task08_HepaticVessel",
            "Task10_Colon",
        ]
    ):
        dataset = json.load(open(f"{inputdir}/dataset.json"))

        all_filenames = dataset["training"]
        random.shuffle(all_filenames)
        train_filenames = all_filenames[: int(0.8 * len(all_filenames))]
        val_filenames = all_filenames[
            int(0.8 * len(all_filenames)) : int(0.95 * len(all_filenames))
        ]
        test_filenames = all_filenames[int(0.95 * len(all_filenames)) :]
        leaderboard_filenames = [
            {"image": image, "label": None} for image in dataset["test"]
        ]

        # add split information
        for filename in train_filenames:
            filename["split"] = "training"
        for filename in val_filenames:
            filename["split"] = "validation"
        for filename in test_filenames:
            filename["split"] = "local_test"
        for filename in leaderboard_filenames:
            filename["split"] = "test"

        print(
            inputdir,
            len(train_filenames),
            len(val_filenames),
            len(test_filenames),
            len(leaderboard_filenames),
        )

        pool = Pool(48)
        pool.map(
            partial(process, inputdir=inputdir),
            train_filenames + val_filenames + test_filenames + leaderboard_filenames,
        )
        pool.close()
        pool.join()

        json_metadata = {"training": [], "validation": [], "local_test": [], "test": []}
        for filename in (
            train_filenames + val_filenames + test_filenames + leaderboard_filenames
        ):
            print(filename)
            process_metadata(filename, inputdir, json_metadata)
        json_metadata["labels"] = dataset["labels"]
        json.dump(
            json_metadata, open(f"processed/{inputdir}/dataset_5slices.json", "w")
        )


if __name__ == "__main__":
    main()
