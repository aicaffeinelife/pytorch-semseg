import numpy as np
import torch
import torch.nn as nn
import os.path as osp
import os
from PIL import Image
"""Visualization Utils for Segmentation"""
PASCAL_ROOT = '/mnt/DATA/VOC/VOCdevkit/VOC2012'


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (21, 3)
    """
    return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )

# TODO: write the corresponding functions of the other datasets
def apply_pascal_colormap(img,
                          nclass=21,
                          normalized=False):
    cmap = get_pascal_labels()
    r = img.copy()
    g = img.copy()
    b = img.copy()
    rgb = np.zeros((img.shape[0], img.shape[1], 3))
    for ii in range(nclass):
        r[img==ii] = cmap[ii, 0]
        g[img==ii] = cmap[ii, 1]
        b[img==ii] = cmap[ii, 2]

    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    # rgb = rgb.astype(np.uint8)
    return rgb.astype(np.uint8)

def create_tile_vis(images,
                    preds,
                    gts):
    """ Create a tile image with the
    actual image, ground truth and colored
    prediction and saves it to the save_path.
    """
    N = len(images)
    assert(N == len(gts) and N == len(preds)), "More gts/preds than images"

    batch_tiles = []

    for (img, gt, pred) in zip(images, gts, preds):
        vis_tile = np.concatenate([img, gt, pred], axis=1)
        batch_tiles.append(vis_tile)
    tile = np.vstack(np.asarray(batch_tiles))
    return tile


