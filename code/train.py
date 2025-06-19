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

def generate_next_label(index, previous_label,
                        mask, skeleton, paths,
                        deskeleton_map):
    """
    Generates the next label during the iteration process.
    Done by taking the intersection of the label in the i-1th step 
    and the reconstructed mask from the ith traversed path.

    NOTE: Indexing should be handled <=> len(artery_paths) <? >? =? len(vein_paths)
    NOTE: Should it be previous_label or previous_prediction? Latter probably.
    
    None is returned if no more labels can be generated <=> traversal has been completed

    The function also returns the first coordinate of the ith step of the traversal.
    This point will be utilized during training - the coordinate is the center of a cube
    that will be cropped from the CT scan for more efficient training. If the traversal has been
    finished, None is returned instead of a coordinate.
    """

    if index >= len(paths):
        return None, None # Traversal has been finished

    next_label = np.copy(previous_label)

    skeleton_segment = fm.segment_to_volume(np.array(paths[index]), skeleton.shape)
    deskeletonized = fm.deskeletonize(skeleton_segment, mask, deskeleton_map)
    next_label[deskeletonized > 0] = 1

    return next_label, paths[index][0]
