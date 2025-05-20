from data.dataset import IterativeSegmentationDataset
import numpy as np

def init_label(artery_mask: np.ndarray, vein_mask: np.ndarray, bboxs: np.ndarray):
    """
    Initializes the labels for the iterative segmentation.
    The 0th label is the intersection of the bounding boxes (inclusive) and the original masks.
    """

    