from data.dataset import IterativeSegmentationDataset
import code.utils.fast_marching as fm

import numpy as np

def init_label(artery_mask, vein_mask, bboxs) -> tuple[np.ndarray, np.ndarray]:
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

def generate_next_label(index, previous_label,
                         artery_mask, artery_skeleton, artery_paths,
                         vein_mask, vein_skeleton, vein_paths,
                         artery_deskeleton_map, vein_deskeleton_map):
    """
    Generates the next label during the iteration process.
    Done by taking the intersection of the label in the i-1th step 
    and the reconstructed mask from the ith traversed path.

    NOTE: Indexing should be handled <=> len(artery_paths) <? >? =? len(vein_paths)
    """

    artery_skeleton_segment = fm.segment_to_volume(np.array(artery_paths[index]), artery_skeleton.shape)
    vein_skeleton_segment = fm.segment_to_volume(np.array(vein_paths[index]), vein_skeleton.shape)

    artery_deskeletonized = fm.deskeletonize(artery_skeleton_segment, artery_mask, artery_deskeleton_map)
    vein_deskeletonized = fm.deskeletonize(vein_skeleton_segment, vein_mask, vein_deskeleton_map)