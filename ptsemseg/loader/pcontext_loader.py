###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

from PIL import Image, ImageOps, ImageFilter
import os
import os.path as osp
import math
import random
import numpy as np
from tqdm import trange
# from ptsemseg.augmentations.augmentations import RandomHorizontallyFlip, RandomRotate, \
#     Compose
import random
import torch
import torch.utils.data as data


# TODO: Refactor the transforms into respective augmentations.
class PascalContext(data.Dataset):
    """Pascal Context Dataset"""
    def __init__(self, root, split, mode='train', transform=None,
                 target_transform=None, base_size=520,
                 crop_size=480):
        from detail import Detail
        self.root = root
        self.mode = mode
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.base_size = base_size
        self.crop_size = crop_size

        # TODO: forego "detail" api and write an ImDB DS to hold all the image metadata
        self.num_classes = 59
        annoFile = osp.join(root, 'trainval_merged.json')
        imgDir = osp.join(root, 'JPEGImages')
        self.detail = Detail(annoFile, imgDir, split)
        self.ids = self.detail.getImgs()
        self._mapping = np.sort(np.array([
            0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22,
            23, 397, 25, 284, 158, 159, 416, 33, 162, 420, 454, 295, 296,
            427, 44, 45, 46, 308, 59, 440, 445, 31, 232, 65, 354, 424,
            68, 326, 72, 458, 34, 207, 80, 355, 85, 347, 220, 349, 360,
            98, 187, 104, 105, 366, 189, 368, 113, 115]))
        self._key = np.array(range(len(self._mapping))).astype('uint8')
        mask_file = os.path.join(root, self.split+'.pth')
        print('mask_file:', mask_file)
        if os.path.exists(mask_file):
            self.masks = torch.load(mask_file)
        else:
            raise RuntimeError("Masks must be pre-downloaded from PyTorch-Encoding")


    def _class_to_mask(self, mask):
        """Given a (M,N,3) mask return a (M,N)
        mask with the colors encoded with class numbers
        """
        values = np.unique(mask)
        for i in range(len(values)):
            assert(values[i] in self._mapping) # check if we're training for more classes
        index = np.digitize(mask.ravel(), self._mapping, right=True) # (x-l, x+r]
        return self._key[index].reshape(mask.shape)

    def _to_tensor(self, mask):
        return torch.from_numpy(np.array(mask)).long()

    def _val_transform(self, img, mask):
        """Fixed Size Crop after resizing"""
        w, h = img.size
        short_size = self.crop_size
        if w > h:
            oh = short_size
            ow = int(1.0 * w*oh/h)
        else:
            ow = short_size
            oh = int(1.0 * h*ow/w)

        img = img.resize((oh,ow), mode=Image.BILINEAR)
        mask = mask.resize((oh,ow), mode=Image.NEAREST)

        w, h = img.size
        x_1 = int(round(h-self.crop_size)/2)
        y_1 = int(round(w-self.crop_size)/2)
        img = img.crop((x_1, y_1, x_1+self.crop_size, y_1+self.crop_size))
        mask = mask.crop((x_1, y_1, x_1+self.crop_size, y_1+self.crop_size))
        mask = self._to_tensor(mask)
        return img, mask

    def _train_transform(self, img, mask):
        """Random Sized Crop with random predictions"""
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        w, h = img.size
        long_size = random.randint(int(self.base_size*0.5),int(self.base_size*2.0))
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        return img, self._to_tensor(mask)

    def __getitem__(self, index):
        img_id = self.ids[index]
        path = img_id['file_name']
        iid = img_id['image_id']
        img = Image.open(os.path.join(self.detail.img_folder, path)).convert('RGB')
        mask = self.masks[iid]
        if self.mode == 'train':
            img, mask = self._train_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_transform(img, mask)
        # additional transforms if needed
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.ids)

