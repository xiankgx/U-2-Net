import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import io, transform
from sklearn.metrics import fbeta_score, mean_absolute_error
from tqdm import tqdm


def iou(gt, pred, thresh=0.5):
    if thresh:
        gt = (gt >= thresh).astype(np.float32)
        pred = (pred >= thresh).astype(np.float32)

    intersection = (gt * pred).sum()
    union = gt.sum() + pred.sum() - intersection

    if union == 0:
        _iou = 1.0
    else:
        _iou = intersection/union

    assert 0 <= _iou <= 1.0
    return _iou


def f_measure(gt, pred, beta_square=0.3):
    return fbeta_score(y_true=gt, y_pred=pred, beta=np.sqrt(beta_square), average="micro")


def evaluate(gt_dir, pred_dir):
    gt_files = glob.glob(gt_dir + "/**/*.png", recursive=True)
    pred_files = list(map(lambda p: p.replace(
        gt_dir, pred_dir), gt_files))
    # pred_files = list(map(lambda p: os.path.splitext(p)[0] + ".jpg", pred_files))

    for f in pred_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"File not found: {f}")

    ious = []
    f_measures = []
    maes = []
    for gt_file, pred_file in tqdm(zip(gt_files, pred_files), desc=f"Evaluating..."):
        gt = io.imread(gt_file)
        pred = io.imread(pred_file)

        if gt.dtype == np.uint8:
            gt = gt/255.0
        if pred.dtype == np.uint8:
            pred = pred/255.0

        if gt.ndim == 3:
            gt = gt[..., 0]
        # RGBA prediction
        if pred.ndim == 3:
            pred = pred[..., -1]

        # Make sure equal dimensions
        if pred.shape[:2] != gt.shape[:2]:
            pred = transform.resize(pred, gt.shape[:2])

        ious.append(iou(gt=gt, pred=pred))
        f_measures.append(f_measure(gt=(gt >= 0.5).astype(np.float32),
                                    pred=(pred >= 0.5).astype(np.float32)))
        maes.append(mean_absolute_error(y_true=gt, y_pred=pred))

    return {
        "gt_dir": gt_dir,
        "pred_dir": pred_dir,

        "iou": np.array(ious).mean(),
        "f_measure": np.array(f_measures).mean(),
        "mae": np.array(maes).mean()
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Salient object detection evaluation script.")
    parser.add_argument("--gt_dir",
                        type=str,
                        help="Ground truth root directory.")
    parser.add_argument("--pred_dir",
                        type=str,
                        help="Prediction root directory.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    metrics = evaluate(args.gt_dir, args.pred_dir)
    print(metrics)

    output_file = os.path.dirname(args.pred_dir) + ".txt"
    # print(output_file)
    with open(output_file, "w") as f:
        f.write(str(metrics))
