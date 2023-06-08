import logging
import sys
import json
from argparse import ArgumentParser
import pytorch_lightning as pl
from data import NIIDataLoader
from model import SegmentationModel
import utils
import torch
from torchprofile import profile_macs


def parse_args(args=None):
    parser = ArgumentParser()

    ## Required parameters for model module
    parser.add_argument("--force_2d", default=0, type=int)
    parser.add_argument("--use_pretrained", default=0, type=int)
    parser.add_argument("--bootstrap_method", default="centering", type=str)

    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--out_channels", default=14, type=int)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--img_size", default=(512, 512, 5), type=int, nargs="+")

    parser.add_argument("--encoder", default="beit", type=str)
    parser.add_argument("--decoder", default="upernet", type=str)
    parser.add_argument("--loss_type", default="dicefocal", type=str)

    parser.add_argument("--dropout_rate", default=0.0, type=float)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--warmup_steps", default=20, type=int)
    parser.add_argument("--max_steps", default=25000, type=int)

    args = parser.parse_args(args)
    return args


def compute_flops_2d(args):
    model = SegmentationModel(
        force_2d=1,
        use_pretrained=args.use_pretrained,
        bootstrap_method=args.bootstrap_method,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        patch_size=args.patch_size,
        img_size=args.img_size,
        encoder=args.encoder,
        decoder=args.decoder,
        loss_type=args.loss_type,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )
    model.eval()
    inputs = torch.randn(1, 1, 1, 512, 512)
    macs = profile_macs(model.encoder, inputs)
    print("2D FLOPS:", macs * 2)


def compute_flops_3d(args):
    model = SegmentationModel(
        force_2d=0,
        use_pretrained=args.use_pretrained,
        bootstrap_method=args.bootstrap_method,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        patch_size=args.patch_size,
        img_size=args.img_size,
        encoder=args.encoder,
        decoder=args.decoder,
        loss_type=args.loss_type,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )
    model.eval()
    inputs = torch.randn(1, 1, 5, 512, 512)
    macs = profile_macs(model.encoder, inputs)
    print("3D FLOPS:", macs * 2)


if __name__ == "__main__":
    args = parse_args()
    compute_flops_2d(args)
    compute_flops_3d(args)
