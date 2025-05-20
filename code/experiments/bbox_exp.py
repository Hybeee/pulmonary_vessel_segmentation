import SimpleITK as sitk
import numpy as np
from scipy.ndimage import find_objects
from code.utils.scan_plotter import scan_to_np_array_with_slice
from code.utils.scan_plotter import view_scan
import matplotlib.pyplot as plt

# image shape is slice x height x width <=> (z, y, x)!

# Splits a given mask into two sides, assigns each pixel/voxel in the mask a label.
# The label is 1 for left - pixel/voxel belongs to the left side -, and the label is 2 for right - pixel/voxel belongs to the right side.
def separate_entry_mask(vein_entry_mask: np.ndarray) -> np.ndarray:
    # calculate connected components
    connected_components = sitk.GetArrayFromImage(
        sitk.ConnectedComponent(sitk.GetImageFromArray(vein_entry_mask))) # each pixel/voxel(?) gets a unique ID -> stored in the numpy array
    component_labels = np.unique(connected_components) # Array of unique IDs
    # select LU, LL, RU, RL points
    sagittal_profile = vein_entry_mask.any(axis=(0, 1)) # does a given sagittal slice contain a mask -> given slice contains a vein
    leftmost_slice = np.argmax(sagittal_profile) # speaks for itself
    rightmost_slice = len(sagittal_profile) - 1 - np.argmax(sagittal_profile[::-1]) # 
    # determine whether it is left or right
    side_labeled_entry_points = np.zeros_like(connected_components) # empty zeros array of shape connected_components
    for l in component_labels[1:]:
        current_entry_mask = connected_components == l
        current_entry_mask_sagittal_profile = current_entry_mask.any(axis=(0, 1))
        leftmost_point = np.argmax(current_entry_mask_sagittal_profile)
        rightmost_point = (len(current_entry_mask_sagittal_profile) - 1 -
                           np.argmax(current_entry_mask_sagittal_profile[::-1]))

        if (leftmost_point - leftmost_slice) < (rightmost_slice - rightmost_point):
            side_label = 1
        else:
            side_label = 2

        side_labeled_entry_points[current_entry_mask] = side_label

    return side_labeled_entry_points

def create_bboxes(side_labeled_entry_points, pixel_spacing):
    bboxes_padded, bboxes = [], []

    expansion_amounts = {
        1: {
            'start_mm': [40., 20., 15.],
            'start_pad_mm': [5., 5., 5.],
            'stop_mm': [10., 10., -5.],
            'stop_pad_mm': [5., 5., 0.],
        },
        2: {
            'start_mm': [30., 25., -5.],
            'start_pad_mm': [5., 5., 0.],
            'stop_mm': [10., 15., 15.],
            'stop_pad_mm': [5., 5., 5.],
        }
    }

    for side_l in [1, 2]: # left and right sides
        side_mask = side_labeled_entry_points == side_l # mask of the given side
        bounding_box = find_objects(side_mask)[0] # retrieves the bounding box of each mask on the side?
        # print(f"Find objects output: {find_objects(side_mask)}")
        # print(f"Got bounding box: {bounding_box}")
        bb_orig_start, bb_orig_stop = [s.start for s in bounding_box], [s.stop for s in bounding_box]

        bbox_start_expand_mm = np.array(expansion_amounts[side_l]['start_mm']) # retrieves expansion values for start
        bbox_stop_expand_mm = np.array(expansion_amounts[side_l]['stop_mm']) # retrieves expansion values for stop
        bbox_padded_start_expand_mm = bbox_start_expand_mm + np.array(expansion_amounts[side_l]['start_pad_mm']) # retrieves padding and adds it to start
        bbox_padded_stop_expand_mm = bbox_stop_expand_mm + np.array(expansion_amounts[side_l]['stop_pad_mm']) # retrieves padding and adds it to end

        bbox_start_expand_px = (bbox_start_expand_mm / pixel_spacing).round().astype(int) # convert start mm to px
        bbox_stop_expand_px = (bbox_stop_expand_mm / pixel_spacing).round().astype(int) # convert stop mm to px
        bbox_padded_start_expand_px = (bbox_padded_start_expand_mm / pixel_spacing).round().astype(int) # convert padding start mm to px
        bbox_padded_stop_expand_px = (bbox_padded_stop_expand_mm / pixel_spacing).round().astype(int) # convert padding stop mm to px

        bb_start = np.clip(np.array(bb_orig_start) - bbox_start_expand_px, 0, side_labeled_entry_points.shape) # clip start values -> stays within valid index range
        bb_stop = np.clip(np.array(bb_orig_stop) + bbox_stop_expand_px, 0, side_labeled_entry_points.shape) # clip end values -> stays within valid index range
        bb_padded_start = np.clip(np.array(bb_orig_start) - bbox_padded_start_expand_px, 0, side_labeled_entry_points.shape) # clip padding start values -> stays within valid index range
        bb_padded_stop = np.clip(np.array(bb_orig_stop) + bbox_padded_stop_expand_px, 0, side_labeled_entry_points.shape) # clip padding end values -> stays within valid index range

        bounding_box = [slice(start, stop) for start, stop in zip(bb_start, bb_stop)]
        bounding_box_padded = [slice(start, stop) for start, stop in zip(bb_padded_start, bb_padded_stop)]

        bboxes.append(bounding_box)
        bboxes_padded.append(bounding_box_padded)

    return bboxes_padded, bboxes

def get_bbox(vein_mask, spacing):
    ts_vein_entry_mask_side_labeled = separate_entry_mask(vein_mask)
    _, bboxes_padded = create_bboxes(ts_vein_entry_mask_side_labeled, spacing)

    return bboxes_padded

def main():
    # skeleton = sitk.ReadImage("dataset/skeleton/005_vein_mask_skeleton.nii.gz")
    # skeleton = sitk.GetArrayFromImage(skeleton)

    # skeleton_slice = np.argwhere(skeleton[103, :, :])

    # print(skeleton_slice)

    # view_scan(skeleton)

    (ct, ct_image_spacing) = scan_to_np_array_with_slice('dataset/HiPaS/ct_scan_nii/005.nii')
    (vein_mask, _) = scan_to_np_array_with_slice('code/resources/result_masks/005_pulmonary_result_masked.nii')
    (vein_and_heart_mask, spacing) = scan_to_np_array_with_slice('code/resources/result_masks/005_heart_and_pulmonary_result_masked.nii')

    print(ct.shape)
    print(vein_mask.shape)

    ct_image_spacing =  [0.7109375, 0.7109375, 1.0] # from metadata.xlsx -> sitk's builtin method didnt work :(

    ct_image_spacing = ct_image_spacing[::-1]

    ts_vein_entry_mask_side_labeled = separate_entry_mask(vein_mask)
    crop_bboxes_padded, crop_bboxes  = create_bboxes(ts_vein_entry_mask_side_labeled, ct_image_spacing)

    mask_copy = np.zeros_like(ct, np.uint32)

    for side_i in crop_bboxes:
        print(f"bbox: {side_i}")
        mask_copy[tuple(side_i)] = vein_mask[tuple(side_i)]

    print(mask_copy.shape)

    view_scan([mask_copy])

    # # mask_copy = sitk.GetImageFromArray(mask_copy)

    # # sitk.WriteImage(mask_copy, "code/result_masks/vessel_entry.nii.gz")

if __name__ == "__main__":
    main()