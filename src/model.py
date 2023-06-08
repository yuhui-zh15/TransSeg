import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import datetime
import pickle
from utils import (
    eval_metrics,
    eval_metrics_per_img,
    get_img_num_slices,
    to_list,
    get_linear_schedule_with_warmup,
)

from monai.losses import DiceCELoss, DiceFocalLoss
from monai.networks.nets.vit import ViT

from backbones.encoders.beit3d import BEiT3D
from backbones.encoders.swin_transformer import SwinTransformer
from backbones.encoders.swin_transformer_3d import SwinTransformer3D
from backbones.encoders.dino3d import VisionTransformer3D
from backbones.decoders.upernet import UPerHead
from backbones.decoders.setrpup import SetrPupHead
from backbones.decoders.convtrans import ConvTransHead
from backbones.decoders.unetr import UnetrHead


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        force_2d: bool = False,  # if set to True, the model will be trained on 2D images by only using the center slice as the input
        use_pretrained: bool = True,  # whether to use pretrained backbone (only applied to BEiT)
        bootstrap_method: str = "centering",  # whether to inflate or center weights from 2D to 3D
        in_channels: int = 1,
        out_channels: int = 14,  # number of classes
        patch_size: int = 16,  # no depthwise
        img_size: tuple = (512, 512, 5),
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        encoder: str = "beit",
        decoder: str = "upernet",
        loss_type: str = "ce",
        save_preds: bool = False,
        dropout_rate: float = 0.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 500,
        max_steps: int = 20000,
        adam_epsilon: float = 1e-8,
    ):
        super().__init__()
        self.modified_loss = (
            True  # TODO: set True to debug (need to modify MONAI codes)
        )
        self.save_hyperparameters()
        self.feat_size = (img_size[0] // patch_size, img_size[1] // patch_size, 1)

        if encoder == "vit":
            self.encoder = ViT(
                in_channels=in_channels,
                img_size=img_size if not force_2d else (img_size[0], img_size[1], 1),
                patch_size=(patch_size, patch_size, img_size[-1])
                if not force_2d
                else (patch_size, patch_size, 1),
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                pos_embed="perceptron",
                classification=False,
                dropout_rate=dropout_rate,
            )
        elif encoder == "beit":
            self.encoder = BEiT3D(
                img_size=img_size if not force_2d else (img_size[0], img_size[1], 1),
                patch_size=(patch_size, patch_size, img_size[-1])
                if not force_2d
                else (patch_size, patch_size, 1),
                in_chans=in_channels,
                embed_dim=hidden_size,
                depth=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_dim // hidden_size,
                qkv_bias=True,
                init_values=1,
                use_abs_pos_emb=False,
                use_rel_pos_bias=True,
            )
            if use_pretrained:
                self.encoder.init_weights(bootstrap_method=bootstrap_method)
        elif encoder == "swint":
            self.encoder = SwinTransformer(
                pretrain_img_size=(img_size[2], img_size[0], img_size[1])
                if not force_2d
                else (1, img_size[0], img_size[1]),
                patch_size=(img_size[2], 4, 4) if not force_2d else (1, 4, 4),
                in_chans=in_channels,
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=7,
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.3,
                ape=False,
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                use_checkpoint=False,
            )
            if use_pretrained:
                self.encoder.init_weights(bootstrap_method=bootstrap_method)
        elif encoder == "dino":
            self.encoder = VisionTransformer3D(
                img_size=img_size if not force_2d else (img_size[0], img_size[1], 1),
                patch_size=(patch_size, patch_size, img_size[-1])
                if not force_2d
                else (patch_size, patch_size, 1),
                in_chans=in_channels,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            if use_pretrained:
                self.encoder.init_weights(bootstrap_method=bootstrap_method)
        elif encoder == "swint3d":
            self.encoder = SwinTransformer3D(
                pretrained2d=False,
                patch_size=(2, 4, 4),
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=(8, 7, 7),
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.2,
                patch_norm=True,
                in_chans=in_channels,
            )
            if use_pretrained:
                self.encoder.init_weights()
        else:
            raise

        if decoder == "upernet":
            self.decoder = UPerHead(
                layer_idxs=[3, 5, 7, 11],
                in_channels=[hidden_size, hidden_size, hidden_size, hidden_size],
                channels=hidden_size,
                num_classes=out_channels,
                dropout_ratio=0.1,
                fpns=True,
            )
        elif decoder == "upernet-swint":
            self.decoder = UPerHead(
                layer_idxs=[0, 1, 2, 3],
                in_channels=[128, 256, 512, 1024],
                channels=512,
                num_classes=out_channels,
                dropout_ratio=0.1,
                fpns=False,
            )
        elif decoder == "setrpup":
            self.decoder = SetrPupHead(
                channels=hidden_size, num_classes=out_channels, norm_name="instance"
            )
        elif decoder == "convtrans":
            self.decoder = ConvTransHead(
                channels=hidden_size, num_classes=out_channels, norm_name="instance"
            )
        elif decoder == "unetr":
            self.decoder = UnetrHead()
        else:
            raise

        if loss_type == "dicece":
            self.criterion = DiceCELoss(to_onehot_y=True, softmax=True)
        elif loss_type == "dicefocal":
            self.criterion = DiceFocalLoss(to_onehot_y=True, softmax=True)
        elif loss_type == "ce":
            self.criterion = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=0)
        else:
            raise

    def forward(self, inputs):  # inputs: B x Cin x H x W x D
        x = inputs.permute(0, 1, 4, 2, 3).contiguous().float()  # x: B x Cin x D x H x W
        xs = self.encoder(x)  # hiddens: list of B x T x hidden, where T = H/P x W/P
        if self.hparams.encoder not in ["swint", "swint3d"]:
            xs = [
                xs[i]
                .view(inputs.shape[0], self.feat_size[0], self.feat_size[1], -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for i in range(len(xs))
            ]  # xs: list of B x hidden x H/P x W/P
        x = self.decoder(xs)  # x: B x Cout x H x W
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["image"], batch["label"]
        n_slices = inputs.shape[-1]
        assert n_slices == self.hparams.img_size[-1]
        if self.hparams.force_2d:
            inputs = inputs[:, :, :, :, n_slices // 2 : n_slices // 2 + 1].contiguous()
        labels = labels[:, :, :, :, n_slices // 2].contiguous()
        outputs = self(inputs)
        if self.modified_loss:
            loss, (dice_loss, ce_loss) = self.criterion(outputs, labels)
        else:
            loss = self.criterion(outputs, labels)
            dice_loss, ce_loss = torch.tensor(0), torch.tensor(0)
        result = {
            "train/loss": loss.item(),
            "train/dice_loss": dice_loss.item(),
            "train/ce_loss": ce_loss.item(),
        }
        self.log_dict(result)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch["image"], batch["label"]
        n_slices = inputs.shape[-1]
        assert n_slices == self.hparams.img_size[-1]
        if self.hparams.force_2d:
            inputs = inputs[:, :, :, :, n_slices // 2 : n_slices // 2 + 1].contiguous()
        labels = labels[:, :, :, :, n_slices // 2].contiguous()
        outputs = self(inputs)
        if self.modified_loss:
            loss, (dice_loss, ce_loss) = self.criterion(outputs, labels)
        else:
            loss = self.criterion(outputs, labels)
            dice_loss, ce_loss = torch.tensor(0), torch.tensor(0)

        return {
            "loss": loss.item(),
            "dice_loss": dice_loss.item(),
            "ce_loss": ce_loss.item(),
            "labels": to_list(labels.squeeze(dim=1)),
            "preds": to_list(outputs.argmax(dim=1)),
        }

    def validation_epoch_end(self, outputs):
        loss = np.array([x["loss"] for x in outputs]).mean()
        dice_loss = np.array([x["dice_loss"] for x in outputs]).mean()
        ce_loss = np.array([x["ce_loss"] for x in outputs]).mean()

        labels = [label for x in outputs for label in x["labels"]]  # N of image shape
        preds = [pred for x in outputs for pred in x["preds"]]  # N of image shape
        inputs = [None] * len(preds)
        acc, accs, ious, dices = eval_metrics(
            preds, labels, self.hparams.out_channels, metrics=["mIoU", "mDice"]
        )

        result = {
            "val/loss": loss,
            "val/dice_loss": dice_loss,
            "val/ce_loss": ce_loss,
            "val/acc": acc,
            "val/macc": accs.mean(),
            "val/miou": ious.mean(),
            "val/mdice": dices.mean(),
            "val/mdice_nobg": dices[1:].mean(),
        }
        if len(dices) == 14:
            print(dices[[8, 4, 3, 2, 6, 11, 1, 7]])
            dice_8 = dices[[1, 2, 3, 4, 6, 7, 8, 11]].mean()
            result["val/mdice_8"] = dice_8
        self.log_dict(result, sync_dist=True)

        if self.hparams.save_preds:
            cur_time = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
            with open(f"dumps/val-{cur_time}.pkl", "wb") as fout:
                pickler = pickle.Pickler(fout)
                for input, pred, label in zip(inputs, preds, labels):
                    pickler.dump({"input": input, "pred": pred, "label": label})

        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch["image"]
        n_slices = inputs.shape[-1]
        assert n_slices == self.hparams.img_size[-1]
        if self.hparams.force_2d:
            inputs = inputs[:, :, :, :, n_slices // 2 : n_slices // 2 + 1].contiguous()
        outputs = self(inputs)

        if "label" in batch:
            labels = batch["label"]
            labels = labels[:, :, :, :, n_slices // 2].contiguous()
            if self.modified_loss:
                loss, (dice_loss, ce_loss) = self.criterion(outputs, labels)
            else:
                loss = self.criterion(outputs, labels)
                dice_loss, ce_loss = torch.tensor(0), torch.tensor(0)

            return {
                "loss": loss.item(),
                "dice_loss": dice_loss.item(),
                "ce_loss": ce_loss.item(),
                "labels": to_list(labels.squeeze(dim=1)),
                "preds": to_list(outputs.argmax(dim=1)),
            }
        else:
            return {
                "preds": to_list(outputs.argmax(dim=1)),
            }

    def test_epoch_end(self, outputs):
        preds = [pred for x in outputs for pred in x["preds"]]  # N of image shape
        inputs = [None] * len(preds)
        if "labels" in outputs[0]:
            loss = np.array([x["loss"] for x in outputs]).mean()
            dice_loss = np.array([x["dice_loss"] for x in outputs]).mean()
            ce_loss = np.array([x["ce_loss"] for x in outputs]).mean()
            labels = [
                label for x in outputs for label in x["labels"]
            ]  # N of image shape
            acc, accs, ious, dices = eval_metrics(
                preds, labels, self.hparams.out_channels, metrics=["mIoU", "mDice"]
            )

            result = {
                "test/loss": loss,
                "test/dice_loss": dice_loss,
                "test/ce_loss": ce_loss,
                "test/acc": acc,
                "test/macc": accs.mean(),
                "test/miou": ious.mean(),
                "test/mdice": dices.mean(),
                "test/mdice_nobg": dices[1:].mean(),
            }
            if len(dices) == 14:
                print(dices[[8, 4, 3, 2, 6, 11, 1, 7]])
                dice_8 = dices[[1, 2, 3, 4, 6, 7, 8, 11]].mean()
                result["test/mdice_8"] = dice_8
            self.log_dict(result, sync_dist=True)

            if self.hparams.save_preds:
                cur_time = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
                with open(f"dumps/test-{cur_time}.pkl", "wb") as fout:
                    pickler = pickle.Pickler(fout)
                    for input, pred, label in zip(inputs, preds, labels):
                        pickler.dump({"input": input, "pred": pred, "label": label})

            return loss
        else:
            assert self.hparams.save_preds
            cur_time = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
            with open(f"dumps/test-{cur_time}.pkl", "wb") as fout:
                pickler = pickle.Pickler(fout)
                for input, pred in zip(inputs, preds):
                    pickler.dump({"input": input, "pred": pred})

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = nn.ModuleList([self.encoder, self.decoder])
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.max_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


if __name__ == "__main__":
    model = SegmentationModel()
