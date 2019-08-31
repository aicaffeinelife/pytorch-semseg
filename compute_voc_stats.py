import os
import os.path as osp
import numpy as np
import cv2

VOC_ROOT='/mnt/DATA/VOC/VOCdevkit/VOC2012'


def compute_voc_class_stats(root, nclasses=21):
    """ Computes a sort of histogram mapping
    each class to it's frequency in the training
    set.
    """
    gt_dir = osp.join(root, 'ImageSets/Segmentation/trainval.txt')
    label_path = osp.join(root, 'SegmentationClass/pre_encoded')
    files = [line.rstrip('\n') for line in open(gt_dir).readlines()]

    histogram = np.zeros(nclasses, dtype=np.int32)
    for fname in files:
        img = cv2.imread(osp.join(label_path, fname+'.png'), 0)
        histogram[np.unique(img)] += 1

    print("Finished computing stats")
    print(histogram)


if __name__ == '__main__':
    compute_voc_class_stats(VOC_ROOT)



