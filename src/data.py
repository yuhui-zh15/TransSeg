import os
import json
import pytorch_lightning as pl
import torch
from functools import partial
import random
import copy
import numpy as np

from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    RandShiftIntensityd,
    NormalizeIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    Resized,
    RandZoomd,
    RandSpatialCropd,
    SpatialPadd,
    MapTransform,
    Randomizable,
)
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


class NIIDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/bcv30/bcv18-12-5slices/",
        split_json="dataset_5slices.json",
        img_size: tuple = (512, 512, 5),
        in_channels: int = 1,
        clip_range: tuple = (-175, 250),
        mean_std: tuple = None,
        train_batch_size: int = 2,
        eval_batch_size: int = 2,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split_json = split_json
        self.img_size = img_size
        self.in_channels = in_channels
        self.clip_range = clip_range
        self.mean_std = mean_std # TODO: remove
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.train_transforms, self.val_transforms = self.create_transforms()

    def create_transforms(self):
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"])
                if self.in_channels == 1
                else AddChanneld(keys=["label"]),
                Resized(
                    keys=["image", "label"],
                    spatial_size=self.img_size,
                    mode=["area", "nearest"],
                ),
                RandZoomd(
                    keys=["image", "label"],
                    min_zoom=0.5,
                    max_zoom=2.0,
                    prob=1,
                    mode=["area", "nearest"],
                    keep_size=False,
                ),
                RandSpatialCropd(
                    keys=["image", "label"], roi_size=self.img_size, random_size=False
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=self.clip_range[0],
                    a_max=self.clip_range[1],
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                # NormalizeIntensityd(
                #     keys=["image"],
                #     subtrahend=None if self.mean_std is None else [self.mean_std[0]],
                #     divisor=None if self.mean_std is None else [self.mean_std[1]],
                #     nonzero=False,
                #     channel_wise=True,
                # ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                SpatialPadd(keys=["image", "label"], spatial_size=self.img_size),
                ToTensord(keys=["image", "label"]),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"])
                if self.in_channels == 1
                else AddChanneld(keys=["label"]),
                Resized(
                    keys=["image", "label"],
                    spatial_size=self.img_size,
                    mode=["area", "nearest"],
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=self.clip_range[0],
                    a_max=self.clip_range[1],
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                # NormalizeIntensityd(
                #     keys=["image"],
                #     subtrahend=None if self.mean_std is None else [self.mean_std[0]],
                #     divisor=None if self.mean_std is None else [self.mean_std[1]],
                #     nonzero=False,
                #     channel_wise=True,
                # ),
                ToTensord(keys=["image", "label"]),
            ]
        )
        return train_transforms, val_transforms

    def setup(self, stage=None):
        data_config_file = f"{self.data_dir}/{self.split_json}"
        data_config = json.load(open(data_config_file))
        print(f"Loading data config from {data_config_file}...")

        train_files = load_decathlon_datalist(
            data_config_file, data_list_key="training"
        )
        val_files = load_decathlon_datalist(
            data_config_file, data_list_key="validation"
        )
        if "local_test" in data_config:
            test_files = load_decathlon_datalist(
                data_config_file, data_list_key="local_test"
            )
        else:
            test_files = []

        self.train_ds = CacheDataset(
            data=train_files,
            transform=self.train_transforms,
            num_workers=6,
            cache_num=64,
        )
        self.val_ds = CacheDataset(
            data=val_files, transform=self.val_transforms, num_workers=3, cache_num=64
        )
        self.test_ds = CacheDataset(
            data=test_files, transform=self.val_transforms, num_workers=3, cache_num=64
        )
        print(
            f"# Train: {len(self.train_ds)}, # Val: {len(self.val_ds)}, # Test: {len(self.test_ds)}..."
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=6,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=3,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=3,
            pin_memory=True,
        )


if __name__ == "__main__":
    dm = NIIDataLoader(data_dir="jsons/", split_json="dataset.json")
    dm.setup()
    # print(dm.train_ds[0]['image'].shape, dm.train_ds[0]['label'].shape)
    print(dm.val_ds[0]["image"].shape, dm.val_ds[0]["label"].shape)
    input("To be continued...")
    for batch in dm.train_dataloader():
        print([key for key in batch])
        print(
            [
                (key, batch[key].shape)
                for key in batch
                if isinstance(batch[key], torch.Tensor)
            ]
        )
        break
