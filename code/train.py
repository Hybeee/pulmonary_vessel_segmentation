from data.dataset import IterativeSegmentationDataset
import code.utils.fast_marching as fm

import numpy as np
import torch

# axaxa kivagasa kell a ct-nek tanitas soran -> maskent nem ferne a memoriaba

def generate_initial_label(artery_mask, vein_mask, bboxs) -> np.ndarray:
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

    if index >= len(artery_paths):
        artery_paths_exhausted = True
    if index >= len(vein_paths):
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

def train(device, epochs, 
          model, optimizer, loss_fn, 
          train_dataset: IterativeSegmentationDataset, val_dataset: IterativeSegmentationDataset = None, 
          verbose=False):
    """
    TODO: Validation related code.
    """
    
    train_loss = list() # will be used for plotting
    val_loss = list()

    for epoch in range(epochs):
        running_loss = 0.0

        model.to(device)
        model.train()

        for idx in range(len(train_dataset)):
            datapoint = train_dataset[idx]

            artery_deskeleton_map = fm.create_deskeleton_map(skeleton=datapoint.artery_skeleton,
                                                             mask_orig=datapoint.artery_mask,
                                                             pixel_spacing=datapoint.spacing)
            vein_deskeleton_map = fm.create_deskeleton_map(skeleton=datapoint.vein_skeleton,
                                                           mask_orig=datapoint.vein_mask,
                                                           pixel_spacing=datapoint.spacing)

            init_label = generate_initial_label(artery_mask=datapoint.artery_mask, vein_mask=datapoint.vein_mask,
                                                bboxs=datapoint.bbox_pair)
            input = init_label

            for i in range(0, max(len(datapoint.artery_paths), len(datapoint.vein_paths))):
                gt_label = generate_next_label(index=i, previous_label=input,
                                               artery_mask=datapoint.artery_mask, artery_skeleton=datapoint.artery_skeleton, artery_paths=datapoint.artery_paths,
                                               vein_mask=datapoint.vein_mask, vein_skeleton=datapoint.vein_skeleton, vein_paths=datapoint.vein_paths,
                                               artery_deskeleton_map=artery_deskeleton_map, vein_deskeleton_map=vein_deskeleton_map)
                
                if gt_label is None:
                    break

                # NOTE: Maybe transform needed here to make gt_label_tensor. Probably... because of normalization. idk. for now only torch.from_numpy
                input_tensor = torch.from_numpy(input)
                gt_label_tensor = torch.from_numpy(gt_label)

                input_tensor = input_tensor.to(device)
                gt_label_tensor = gt_label_tensor.to(device)

                # NOTE: Should gradients be calculated here, or summed up for the path traversal and perform backprop/optimizer.step() based on the
                # loss of the entire traversal/processing of the current data point?
                # For now: opt.step() is called for each prediction during traversal

                output = model(input)

                loss = loss_fn(output, gt_label_tensor)
                running_loss += loss

                loss.backward()
                optimizer.step()

                # Input for the next iteration is the ground truth of the current iteration
                input = gt_label
        
        running_loss /= len(train_dataset)

        print(f"Completed epoch: {epoch}")
        if verbose:
            print(f"\tTraining loss: {running_loss}")
        
        train_loss.append(running_loss)