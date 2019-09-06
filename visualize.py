import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import os
import os.path as osp
from vis_utils import apply_pascal_colormap, create_tile_vis
import imageio
import scipy.misc as misc
import argparse
import torchvision.transforms as transforms
import yaml
from PIL import Image
from ptsemseg.models import get_model
from ptsemseg.utils import load_checkpoint
# TODO: Extend for other datasets
parser = argparse.ArgumentParser(description="Segmentation Visualization")
parser.add_argument("--ckpt",
                    type=str,
                    required=True,
                    help='Path to trained model')

parser.add_argument("--config",
                    type=str,
                    required=True,
                    help='Path to config file')


mean_std = {
    "mean": np.array([104.00699, 116.66877, 122.67892]),
    "std": [0.229, 0.224, 0.225]
}


def inference_single(model,
                     image_file,
                     gt_file,
                     imsize,
                     save_path):
    """Perform inference on a single image
    Image file is assumed to be the absolute
    path to the image.
    """
    model.eval()
    image = np.array(imageio.imread(image_file))
    gt_anno = imageio.imread(gt_file)
    gt_anno = np.array(gt_anno[:, :, :-1])
    if isinstance(imsize, int):
        imsize = (imsize, imsize)
    image = misc.imresize(image, imsize)
    gt_anno = misc.imresize(gt_anno, imsize)
    gt_img = image
    image = image.astype(np.float64)
    image -= mean_std['mean']
    image /= 255 # normalize the image

    image = np.transpose(image, (2,0,1))
    image = np.expand_dims(image, 0)
    im_tensor = torch.from_numpy(image).float().cuda()

    im_var = Variable(im_tensor)
    with torch.no_grad():
        output = model(im_var)
    preds = np.squeeze(output.max(1)[1].cpu().numpy(), 0)
    assert(preds.shape == (512, 512))
    decoded_im = apply_pascal_colormap(preds)
    print("Classes found: ", np.unique(preds))
    tile = np.concatenate([gt_img, gt_anno, decoded_im], axis=1)
    imageio.imsave(save_path, tile)




def batch_inference(model,
                    root,
                    image_list,
                    gt_list,
                    imsize,
                    save_path):
    """Perform inference on a list of images"""
    def prepare_example(image):
        image = image.astype(np.float64)
        image = image.astype(np.float64)
        image -= mean_std['mean']
        image /= 255 # normalize the image

        image = np.transpose(image, (2,0,1))
        # image = np.expand_dims(image, 0)
        return image

    images = []
    N = len(image_list)
    batch_preds = []
    gts = [np.array(imageio.imread(gfile)) for gfile in gt_list]
    batch_imgs = np.zeros((len(image_list), -1))
    model.eval()


    for image_f in images:
        img = np.array(imageio.imread(image_f))
        pimg = prepare_example(img)
        images.append(img)
        batch_imgs = np.concatenate([batch_imgs, pimg], axis=0)
    assert(batch_imgs.shape[0] == N)
    print(batch_imgs.shape)
    batch_tensor = torch.from_numpy(batch_imgs).cuda()
    batch_var = Variable(batch_tensor)
    with torch.no_grad():
        batch_out = model(batch_var)
    preds = batch_out.max(1)[1].cpu().numpy() # NxHxW
    # decode maps
    for i in range(N):
        print("Found classes in image {}: {}".format(i, np.unique(preds)))
        decoded = apply_pascal_colormap(preds[i, :, :])
        batch_preds.append(decoded)

    tile = create_tile_vis(images, batch_preds, gts)
    imageio.imsave(save_path, tile)
    print("Saved inference tile to:{}".format(save_path))







if __name__ == '__main__':
    nclasses = 21
    args = parser.parse_args()
    with open(args.config) as cfile:
        cfg = yaml.load(cfile)

    model = get_model(cfg["model"], nclasses).cuda()
    model.eval()
    ckpt = load_checkpoint(args.ckpt)
    model.load_state_dict(ckpt["model_state"])
    print("Loaded from epoch:{}, best_iou:{}".format(ckpt["epoch"],ckpt["best_iou"]))
    root = '/mnt/DATA/VOC/VOCdevkit/VOC2012'
    img_f = '2007_000129'
    image_path = osp.join(root, 'JPEGImages', img_f+'.jpg')
    gt_path = osp.join(root, 'SegmentationClass', img_f+'.png')
    save_path = 'fcn8s_single_inference.png'
    inference_single(model, image_path, gt_path, (512, 512), save_path)
    # img_list = ['2007_000129']

    # image = imageio.imread(osp.join(root, 'JPEGImages', img_list[0]+'.jpg'))
    # image = np.array(image)
    # assert(np.all(image) == 0), "Image is corrupted" 

    # image = misc.imresize(image, (512, 512))
    # gt_img = image
    # image = image.astype(np.float64)
    # image -= mean_std['mean']
    # image /= 255 # normalize the image

    # image = np.transpose(image, (2,0,1))
    # image = np.expand_dims(image, 0)
    # im_tensor = torch.from_numpy(image).float().cuda()

    # im_var = Variable(im_tensor)
    # with torch.no_grad():
    #     output = model(im_var)
    # preds = np.squeeze(output.max(1)[1].cpu().numpy(), 0)
    # assert(preds.shape == (512, 512))
    # decoded_im = apply_pascal_colormap(preds)
    # print("Classes found: ", np.unique(preds))
    # tile = np.concatenate([gt_img, decoded_im], axis=1)
    # imageio.imsave("test_fcn8s_vis.png", tile)


