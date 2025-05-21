from data.dataset import IterativeSegmentationDataset
import code.utils.fast_marching as fm

import numpy as np

def init_label(artery_mask, vein_mask, bboxs) -> np.ndarray:
    """
    Initializes the labels for the iterative segmentation.
    The 0th label is the intersection of the bounding boxes (inclusive) and the original masks.
    Class index/pixel value of:
        - background: 0
        - artery: 1
        - vein: 2
    """

    assert artery_mask.shape == vein_mask.shape, "Mask shapes must be the same"

    artery_mask_copy = np.zeros_like(artery_mask, np.uint32)
    vein_mask_copy = np.zeros_like(vein_mask, np.uint32)

    for bbox in bboxs:
        artery_mask_copy[tuple(bbox)] = artery_mask[tuple(bbox)]
        vein_mask_copy[tuple(bbox)] = vein_mask[tuple(bbox)]

    label = np.zeros_like(artery_mask_copy)
    label[artery_mask_copy > 0] = 1 # class of artery: 1
    label[vein_mask_copy > 0] = 2 # class of vein: 2

    return label

def generate_next_label(index, previous_label,
                         artery_mask, artery_skeleton, artery_paths,
                         vein_mask, vein_skeleton, vein_paths,
                         artery_deskeleton_map, vein_deskeleton_map):
    """
    Generates the next label during the iteration process.
    Done by taking the intersection of the label in the i-1th step 
    and the reconstructed mask from the ith traversed path.

    NOTE: Indexing should be handled <=> len(artery_paths) <? >? =? len(vein_paths)
    NOTE: Should it be previous_label or previous_prediction?
    
    None is returned if no more labels can be generated <=> traversal has been completed
    """

    next_label = np.copy(previous_label)
    artery_paths_exhausted = False
    vein_paths_exhausted = False

    if index > len(artery_paths) - 1:
        artery_paths_exhausted = True
    if index > len(vein_paths) - 1:
        vein_paths_exhausted = True

    if artery_paths_exhausted and vein_paths_exhausted:
        return None

    if not artery_paths_exhausted:
        artery_skeleton_segment = fm.segment_to_volume(np.array(artery_paths[index]), artery_skeleton.shape)
        artery_deskeletonized = fm.deskeletonize(artery_skeleton_segment, artery_mask, artery_deskeleton_map)
        next_label[artery_deskeletonized > 0] = 1

    if not vein_paths_exhausted:
        vein_skeleton_segment = fm.segment_to_volume(np.array(vein_paths[index]), vein_skeleton.shape)
        vein_deskeletonized = fm.deskeletonize(vein_skeleton_segment, vein_mask, vein_deskeleton_map)
        next_label[vein_deskeletonized > 0] = 2
    
    return next_label
