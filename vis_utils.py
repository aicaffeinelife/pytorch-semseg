import numpy as np
import torch
import torch.nn as nn
import os.path as osp
import os
import cv2

"""Visualization Utils for Segmentation"""
PASCAL_ROOT = '/mnt/DATA/VOC/VOCdevkit/VOC2012'

# code from gist:https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
def pascal_colormap(N=256,
                    normalized=False):
    def bitget(byte, idx):
        return (byte & (1 << idx)) != 0
    dtype = np.float32 if normalized else np.uint8
    cmap = np.zeros((N,3), dtype=dtype)

    # loop over N pixel levels
    for ii in range(N):
        r = g = b = 0
        c = ii
        for j in range(8):  # rgb is packed into a byte
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3 # ii = ii/8
        cmap[ii] = np.array([r,g,b])
    cmap  = cmap / 255 if normalized else cmap
    return cmap


# TODO: write the corresponding functions of the other datasets
def apply_pascal_colormap(img,
                          nclass=21,
                          normalized=False):
    cmap = pascal_colormap()
    dtype = np.float32 if normalized else np.uint8
    rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=dtype)
    for ii in range(nclass):
        rgb[:, :, 0] = cmap[ii, 0]
        rgb[:, :, 1] = cmap[ii, 1]
        rgb[:, :, 2] = cmap[ii, 2]

    if normalized:
        rgb[:, :, 0] /= 255
        rgb[:, :, 1] /= 255
        rgb[:, :, 2] /= 255
    return rgb
    

def create_tile_vis(images,
                    preds,
                    gts,
                    save_path,
                    img_size=(400,400)):
    """ Create a tile image with the
    actual image, ground truth and colored
    prediction and saves it to the save_path.

    images: A single image or a list
    pred: A single associated prediction or list of prediction
    """
    if type(images) == list:
        assert(len(images) == len(preds)), "Lengths don't match"
        N = len(images)
        tile = np.zeros((N, img_size[0]*3))
        for i, img, gt, pred in enumerate(zip(images, preds, gts)):
            img = cv2.resize(img, img_size)
            pred = cv2.resize(pred, img_size)
            gt = cv2.resize(gt, img_size)
            t_img = np.concatenate([img, pred, gt], axis=1)
            tile[i, :] = t_img
        cv2.imwrite(save_path, tile)
    else:
        image = cv2.resize(images, img_size)
        pred = cv2.resize(preds, img_size)
        gt = cv2.resize(gts, img_size)
        tile = np.concatenate([image, pred, gt], axis=1)
        cv2.imwrite(save_path, tile)









