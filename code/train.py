from data.dataset import IterativeSegmentationDataset
import code.utils.fast_marching as fm

import numpy as np
import torch

def generate_initial_label(mask, bboxs) -> np.ndarray:
    """
    Initializes the labels for the iterative segmentation.
    The 0th label is the intersection of the bounding boxes (inclusive) and the original mask.
    Class index/pixel value of:
        - background: 0
        - artery: 1
        - vein: 2
    """

    mask_copy = np.zeros_like(mask, np.uint32)

    for bbox in bboxs:
        mask_copy[tuple(bbox)] = mask[tuple(bbox)]

    label = np.zeros_like(mask_copy)
    label[mask_copy > 0] = 1 # Voxel that belongs to artery/vein

    return label