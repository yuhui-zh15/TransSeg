import json
from collections import OrderedDict

import mmcv
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


# FIXME: This should have been a member var of the model class
# But putting it in utils for now to avoid interface mismatch with old checkpoints
# Format: val/test -> list(int) of number of slices per image
IMG_NUM_SLICES = None


def load_img_num_slices(data_dir, split_json):
    global IMG_NUM_SLICES
    if IMG_NUM_SLICES is not None:
        return IMG_NUM_SLICES

    IMG_NUM_SLICES = dict()

    data_config = json.load(open(f"{data_dir}/{split_json}"))
    for split in ["validation", "local_test"]:

        img_id_to_num_slices = OrderedDict()
        for img_config in data_config[split]:
            img_id, slice_id = (
                img_config["label"]
                .split("/")[-1]
                .replace(".nii.gz", "")
                .split("_")[1:3]
            )
            img_id, slice_id = int(img_id), int(slice_id)
            if img_id not in img_id_to_num_slices:
                img_id_to_num_slices[img_id] = 0
            img_id_to_num_slices[img_id] += 1

        IMG_NUM_SLICES[split] = []
        for img_id, num_slices in img_id_to_num_slices.items():
            IMG_NUM_SLICES[split].append(num_slices)


def get_img_num_slices(split):
    return IMG_NUM_SLICES[split]


def to_list(tensor):
    tensor_np = tensor.detach().cpu().numpy()
    return [tensor_np[i] for i in range(tensor_np.shape[0])]


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def intersect_and_union(
    pred_label,
    label,
    num_classes,
    ignore_index,
    label_map=dict(),
    reduce_zero_label=False,
):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray): Prediction segmentation map.
        label (ndarray): Ground truth segmentation map.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(label, str):
        label = mmcv.imread(label, flag="unchanged", backend="pillow")
    # modify if custom classes
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = label != ignore_index
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(
    results,
    gt_seg_maps,
    num_classes,
    ignore_index,
    label_map=dict(),
    reduce_zero_label=False,
):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes,), dtype=np.float)
    total_area_union = np.zeros((num_classes,), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes,), dtype=np.float)
    total_area_label = np.zeros((num_classes,), dtype=np.float)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(
            results[i],
            gt_seg_maps[i],
            num_classes,
            ignore_index,
            label_map,
            reduce_zero_label,
        )
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return (
        total_area_intersect,
        total_area_union,
        total_area_pred_label,
        total_area_label,
    )


def mean_iou(
    results,
    gt_seg_maps,
    num_classes,
    ignore_index,
    nan_to_num=None,
    label_map=dict(),
    reduce_zero_label=False,
):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category IoU, shape (num_classes, ).
    """

    all_acc, acc, iou = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=["mIoU"],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label,
    )
    return all_acc, acc, iou


def mean_dice(
    results,
    gt_seg_maps,
    num_classes,
    ignore_index,
    nan_to_num=None,
    label_map=dict(),
    reduce_zero_label=False,
):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category dice, shape (num_classes, ).
    """

    all_acc, acc, dice = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=["mDice"],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label,
    )
    return all_acc, acc, dice


def eval_metrics(
    results,
    gt_seg_maps,
    num_classes,
    ignore_index=255,
    metrics=["mIoU"],
    nan_to_num=None,
    label_map=dict(),
    reduce_zero_label=False,
):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category evalution metrics, shape (num_classes, ).
    """

    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ["mIoU", "mDice"]
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError("metrics {} is not supported".format(metrics))
    (
        total_area_intersect,
        total_area_union,
        total_area_pred_label,
        total_area_label,
    ) = total_intersect_and_union(
        results, gt_seg_maps, num_classes, ignore_index, label_map, reduce_zero_label
    )
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    ret_metrics = [all_acc, acc]
    for metric in metrics:
        if metric == "mIoU":
            iou = total_area_intersect / total_area_union
            ret_metrics.append(iou)
        elif metric == "mDice":
            dice = 2 * total_area_intersect / (total_area_pred_label + total_area_label)
            ret_metrics.append(dice)
    if nan_to_num is not None:
        ret_metrics = [np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics]
    return ret_metrics


def eval_metrics_per_img(
    results,
    gt_seg_maps,
    num_classes,
    img_num_slices,
    ignore_index=255,
    metrics=["mIoU"],
    nan_to_num=None,
    label_map=dict(),
    reduce_zero_label=False,
):
    """Calculate evaluation metrics, grouped by each 3D image.
    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        img_num_slices (list[int]): list of number of slices for each patient.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category evalution metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ["mIoU", "mDice"]
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError("metrics {} is not supported".format(metrics))
    if sum(img_num_slices) != len(results):
        raise ValueError(
            "Total number of image slices must be equal to results."
            f"Got {sum(img_num_slices)} != {len(results)}."
        )

    ret_all_acc = []
    ret_acc = []
    ret_iou = []
    ret_dice = []

    idx_start = 0
    for num_slices in img_num_slices:
        idx_end = idx_start + num_slices
        img_results = results[idx_start:idx_end]
        img_gt_seg_maps = gt_seg_maps[idx_start:idx_end]
        (
            img_area_intersect,
            img_area_union,
            img_area_pred_label,
            img_area_label,
        ) = total_intersect_and_union(
            img_results,
            img_gt_seg_maps,
            num_classes,
            ignore_index,
            label_map,
            reduce_zero_label,
        )
        img_all_acc = img_area_intersect.sum() / img_area_label.sum()
        img_acc = img_area_intersect / img_area_label
        img_iou = img_area_intersect / img_area_union
        img_dice = 2 * img_area_intersect / (img_area_pred_label + img_area_label)

        ret_all_acc.append(img_all_acc)
        ret_acc.append(img_acc)
        ret_iou.append(img_iou)
        ret_dice.append(img_dice)

        idx_start = idx_end

    # If an image has NaN metric, then skip it in the global mean.
    ret_metrics = [np.nanmean(ret_all_acc), np.nanmean(ret_acc, axis=0)]
    for metric in metrics:
        if metric == "mIoU":
            ret_metrics.append(np.nanmean(ret_iou, axis=0))
        elif metric == "mDice":
            ret_metrics.append(np.nanmean(ret_dice, axis=0))
    if nan_to_num is not None:
        ret_metrics = [np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics]
    return ret_metrics


if __name__ == "__main__":
    results = [np.ones((3, 3)), np.ones((3, 3))]
    gt_seg_maps = [np.ones((3, 3)), np.ones((3, 3))]
    num_classes = 5
    img_num_slices = [1, 1]
    metrics = eval_metrics_per_img(
        results, gt_seg_maps, num_classes, img_num_slices, metrics=["mIoU", "mDice"]
    )
    # metrics = eval_metrics(results, gt_seg_maps, num_classes, metrics=["mIoU", "mDice"])
    print(metrics)
    load_img_num_slices(
        data_dir="data/msd/processed/Task07_Pancreas/",
        split_json="dataset_5slices.json",
    )
    print(get_img_num_slices("validation"))
    print(get_img_num_slices("local_test"))
