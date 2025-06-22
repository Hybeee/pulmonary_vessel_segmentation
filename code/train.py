from data.dataset import IterativeSegmentationDataset, DataPoint
import utils.fast_marching as fm

import numpy as np
import torch
import SimpleITK as sitk
from utils.scan_plotter import view_scan

class DataPointLoader:
    def __init__(self, datapoint: DataPoint):
        self.datapoint = datapoint
        self.artery_deskeleton_map = fm.create_deskeleton_map(skeleton=datapoint.artery_skeleton,
                                                         mask_orig=datapoint.artery_mask,
                                                         pixel_spacing=datapoint.spacing)
        self.vein_deskeleton_map = fm.create_deskeleton_map(skeleton=datapoint.vein_skeleton,
                                                            mask_orig=datapoint.vein_mask,
                                                            pixel_spacing=datapoint.spacing)
        self.init_artery_label = generate_initial_label(mask=datapoint.artery_mask, bboxs=datapoint.bbox_pair)
        self.init_vein_label = generate_initial_label(mask=datapoint.vein_mask, bboxs=datapoint.bbox_pair)

    def get_current_data(self, index):
        if index == 0:
            return (self.init_artery_label, self.artery_deskeleton_map,
                    self.datapoint.artery_mask, self.datapoint.artery_skeleton,
                    self.datapoint.artery_paths)
        else:
            return (self.init_vein_label, self.vein_deskeleton_map,
                    self.datapoint.vein_mask, self.datapoint.vein_skeleton,
                    self.datapoint.vein_paths)

def generate_initial_label(mask, bboxs) -> np.ndarray:
    """
    Initializes the labels for the iterative (binary) segmentation.
    The 0th label is the intersection of the bounding boxes (inclusive) and the original mask.
    Class index/pixel value of:
        - background: 0
        - segmentation: 1
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

def pad_image(image, pad_width):
    """
    Pads the 3D image with zeros on all sides by pad_width voxels.
    """

    image_copy = np.copy(image)

    padded_image = np.pad(image_copy,
                          pad_width=((pad_width, pad_width),
                                     (pad_width, pad_width),
                                     (pad_width, pad_width)),
                                     mode='constant', constant_values=0)
    
    return padded_image


def get_3d_patch(image, center, a=40):
    """
    Extracts a 3D patch from the input image.
    The extracted patch is a cube with side lengths of a.
    The parameter center is the center of the extracted cube/patch.
    center's shape is [z, y, x]
    """

    half = a // 2 # floor rounding

    padded_image = pad_image(image=image, pad_width=half)
    adjusted_center = center + half

    x_start = adjusted_center[2] - half
    x_end = adjusted_center[2] + half + (1 if a % 2 != 0 else 0)
    y_start = adjusted_center[1] - half
    y_end = adjusted_center[1] + half + (1 if a % 2 != 0 else 0)
    z_start = adjusted_center[0] - half
    z_end = adjusted_center[0] + half + (1 if a % 2 != 0 else 0)

    patch = padded_image[z_start:z_end, y_start:y_end, x_start:x_end]

    return patch


def train(device, epochs,
          model, optimizer, loss_fn,
          train_dataset: IterativeSegmentationDataset, val_dataset: IterativeSegmentationDataset,
          verbose=False):
    """
    TODO: Validation related code. Until training's logic is not finalized, no need to implement it.
    """
    train_loss = list() # will be used for plotting
    val_loss = list()

    if verbose:
        print(f"Starting training for {epochs} epoch(s)")
        print("-----")

    for epoch in range(epochs):
        running_loss = 0.0

        model.to(device)
        model.train()

        for data_idx in range(len(train_dataset)):
            datapoint = train_dataset[data_idx]
            datapoint_loader = DataPointLoader(datapoint=datapoint)
            input = datapoint.ct

            for i in range(2):
                (init_label, deskeleton_map, mask, skeleton, paths) = datapoint_loader.get_current_data(i)

                # NOTE
                # Itt kerdeses: Mi legyen a halo alap bemenete? Az elvart kimenete valszeg az amit hozzadok generate_next_label-ben(?)
                # Lehet majd eszembe jut DE: Hogy birjuk itt ra a halot, hogy jo iranyba induljon el?
                # init_label = az eredeti teljes maszk azon resze, ami a megadott bounding box-on belul van.
                current_label = init_label

                for path_idx in range(len(paths)):
                    gt_label, curr_path_start = generate_next_label(index=path_idx, previous_label=current_label,
                                                   mask=mask, skeleton=skeleton, paths=paths,
                                                   deskeleton_map=deskeleton_map)

                    if gt_label is None:
                        break

                    input_patch = get_3d_patch(image=input, center=curr_path_start)
                    gt_label_patch = get_3d_patch(image=gt_label, center=curr_path_start)

                    input_tensor = torch.from_numpy(input_patch)
                    gt_label_tensor = torch.from_numpy(gt_label_patch)

                    input_tensor = input_tensor.to(device)
                    gt_label_tensor = gt_label_tensor.to(device)

                    output = model(input_tensor)

                    loss = loss_fn(output, gt_label_tensor)
                    running_loss += loss

                    loss.backward()
                    optimizer.step()

                    current_label = gt_label

        running_loss /= len(train_dataset)

        if verbose:
            print(f"Completed epoch: {epoch}")
            print(f"\tTraining loss: {running_loss}")

        train_loss.append(running_loss)
        running_loss = 0

def main():
    ct = sitk.ReadImage("dataset/HiPaS/ct_scan_nii/005.nii.gz")
    ct = sitk.GetArrayFromImage(ct)
    vein_mask = sitk.ReadImage("dataset/HiPaS/annotation/vein_nii/005.nii.gz")
    vein_mask = sitk.GetArrayFromImage(vein_mask)

    segment_coords = [np.array([103, 257, 213], dtype=np.int64),
                  np.array([101, 257, 214], dtype=np.int16),
                  np.array([100, 256, 214], dtype=np.int16)]
    
    print(f"CT shape: {ct.shape}")

    ct_patch = get_3d_patch(image=ct, center=segment_coords[0])
    vein_mask_patch = get_3d_patch(image=vein_mask, center=segment_coords[0])

    print(f"CT patch shape: {ct_patch.shape}")
    view_scan([ct_patch, vein_mask_patch])

if __name__ == "__main__":
    main()