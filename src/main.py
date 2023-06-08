import logging
import sys
import json
from argparse import ArgumentParser
import pytorch_lightning as pl
from data import NIIDataLoader
from model import SegmentationModel


def parse_args(args=None):
    parser = ArgumentParser()

    ## Required parameters for data module
    parser.add_argument("--data_dir", default="jsons/", type=str)
    parser.add_argument("--split_json", default="dataset.json", type=str)
    parser.add_argument("--train_batch_size", default=1, type=int)
    parser.add_argument("--eval_batch_size", default=1, type=int)
    parser.add_argument("--clip_range", default=(-175, 250), type=int, nargs="+")
    parser.add_argument("--mean_std", default=None, type=float, nargs="+")

    ## Required parameters for model module
    parser.add_argument("--force_2d", default=0, type=int)
    parser.add_argument("--use_pretrained", default=1, type=int)
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

    ## Required parameters for trainer module
    parser.add_argument("--default_root_dir", default=".", type=str)
    parser.add_argument("--gpus", default=-1, type=int)
    parser.add_argument("--val_check_interval", default=1.0, type=float)
    parser.add_argument("--check_val_every_n_epoch", default=100, type=int)
    parser.add_argument("--gradient_clip_val", default=1.0, type=float)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--log_every_n_steps", default=1, type=int)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--accelerator", default="ddp", type=str)
    parser.add_argument("--seed", default=1234, type=int)

    ## Require parameters for evaluation
    parser.add_argument("--evaluation", default=0, type=int)
    parser.add_argument("--model_path", default=None, type=str)

    args = parser.parse_args(args)
    return args


def train(args):
    wandb_logger = pl.loggers.WandbLogger(
        project="MedicalSegmentation", config=vars(args), log_model=False
    )

    pl.seed_everything(args.seed)

    dm = NIIDataLoader(
        data_dir=args.data_dir,
        split_json=args.split_json,
        img_size=args.img_size,
        in_channels=args.in_channels,
        clip_range=args.clip_range,
        mean_std=args.mean_std,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    if args.model_path is None:
        model = SegmentationModel(
            force_2d=args.force_2d,
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
    else:
        model = SegmentationModel.load_from_checkpoint(args.model_path)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1, monitor="val/mdice", mode="max"
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        default_root_dir=args.default_root_dir,
        gpus=args.gpus,
        val_check_interval=args.val_check_interval,
        # check_val_every_n_epoch=args.check_val_every_n_epoch,
        max_steps=args.max_steps,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator=args.accelerator,
        logger=wandb_logger,
        # limit_train_batches=1, # TODO: uncomment for debugging
        # limit_val_batches=1, # TODO: uncomment for debugging
        # limit_test_batches=1, # TODO: uncomment for debugging
    )
    trainer.fit(model, datamodule=dm)
    trainer.validate(datamodule=dm)
    trainer.test(datamodule=dm)


def evaluate(args):
    wandb_logger = pl.loggers.WandbLogger(
        project="MedicalSegmentation", config=vars(args), log_model=False
    )

    pl.seed_everything(args.seed)

    dm = NIIDataLoader(
        data_dir=args.data_dir,
        split_json=args.split_json,
        img_size=args.img_size,
        in_channels=args.in_channels,
        clip_range=args.clip_range,
        mean_std=args.mean_std,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )

    model = SegmentationModel.load_from_checkpoint(args.model_path)
    model.hparams.save_preds = True

    trainer = pl.Trainer(
        default_root_dir=args.default_root_dir,
        gpus=args.gpus,
        precision=args.precision,
        accelerator=args.accelerator,
        logger=wandb_logger,
        num_sanity_val_steps=-1,
        # limit_val_batches=1, # TODO: uncomment for debugging
        # limit_test_batches=1, # TODO: uncomment for debugging
    )
    trainer.validate(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    args = parse_args()
    if not args.evaluation:
        train(args)
    else:
        evaluate(args)
