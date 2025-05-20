from data.dataset import IterativeSegmentationDataset
import numpy as np

def init_label(artery_mask: np.ndarray, vein_mask: np.ndarray, bboxs: np.ndarray):
    """
    Initializes the labels for the iterative segmentation.
    The 0th label is the intersection of the bounding boxes (inclusive) and the original masks.
    """

    artery_mask_copy = np.zeros_like(artery_mask, np.uint32)
    vein_mask_copy = np.zeros_like(vein_mask, np.uint32)

    for bbox in bboxs:
        artery_mask_copy[tuple(bbox)] = artery_mask[tuple(bbox)]
        vein_mask_copy[tuple(bbox)] = vein_mask[tuple(bbox)]

    return artery_mask_copy, vein_mask_copy