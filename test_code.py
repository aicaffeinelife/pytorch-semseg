from ptsemseg.loader import get_loader
from ptsemseg.augmentations import Compose, Scale, RandomRotate, RandomHorizontallyFlip
from ptsemseg.models import DANet
import os
import os.path as osp
from torch.autograd import Variable
import torch

"""File to test code in various subfolders because Python is a GOD DAMN IDIOT"""

if __name__ == '__main__':

    model = DANet('resnet50',
                  19,
                  pretrain_root='ptsemseg/models/pre_trained',
                  dilated=True).cuda()
    print(model)
    print(sum([p.data.nelement() for p in model.parameters()]))

    y = Variable(torch.randn(1, 3, 512, 512).cuda())
    with torch.no_grad():
        out = model(y)

    for o in out:
        print(o.size())


    # data_root = '/mnt/DATA/datasets/Cityscapes'
    # augmentations = Compose([
    #     Scale(2048),
    #     RandomRotate(10),
    #     RandomHorizontallyFlip(0.5)
    # ])

    # loader = get_loader("cityscapes")
    # cityscapes = loader(data_root, split="train", is_transform=False)
    # print(len(cityscapes.files['train']))
    # print(len(cityscapes.files['val']))
    # cityscapes.compute_stats()

