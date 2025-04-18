import SimpleITK as sitk
import numpy as np
import cv2
from scan_plotter import view_scan
from scipy.ndimage import sobel
from utils.bbox_code import get_bbox
from scan_plotter import scan_to_np_array


def get_intersection(image_1, image_2):
    intersections = np.bitwise_and(image_1, image_2)

    return intersections

def detect_edges_2d(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    
    edges = cv2.filter2D(image, -1, kernel)

    return edges

def detect_edges_3d(image):
    dx = sobel(image, axis=2)
    dy = sobel(image, axis=1)
    dz = sobel(image, axis=0)

    magnitude = np.sqrt(dx**2 + dy**2 + dz**2)

    threshold = 5

    edges = (magnitude > threshold).astype(np.uint8)

    return edges

def detect_edges_3d_sitk(image):
    image = sitk.GetImageFromArray(image)

    gradient = sitk.GradientMagnitude(image)

    gradient = sitk.GetArrayFromImage(gradient)

    threshold = 0
    edges = (gradient > threshold).astype(np.uint8)

    return edges

def get_intersections_bbox_helper(skeleton, bboxs):
    results = []
    # bbox is z y x
    for bbox in bboxs:
        z_start, z_end = bbox[0].start, bbox[0].stop
        y_start, y_end = bbox[1].start, bbox[1].stop
        x_start, x_end = bbox[2].start, bbox[2].stop # x_end is not needed...? that's the middle side of the bounding box

        # print(f"z_start: {z_start} and z_end: {z_end}")
        # print(f"y_start: {y_start} and y_end: {y_end}")
        # print(f"x_start: {x_start} and x_end: {x_end}")

        z_start_intersections = np.argwhere(skeleton[z_start, :, :])
        z_end_intersections = np.argwhere(skeleton[z_end, :, :])
        y_start_intersections = np.argwhere(skeleton[:, y_start, :])
        y_end_intersections = np.argwhere(skeleton[:, y_end, :])
        x_start_intersections = np.argwhere(skeleton[:, :, x_start])

        for ys, xs in z_start_intersections:
            results.append([z_start, ys, xs])
        for ye, xe in z_end_intersections:
            results.append([z_end, ye, xe])

        for zs, xs in y_start_intersections:
            results.append([zs, y_start, xs])
        for ze, xe in y_end_intersections:
            results.append([ze, y_end, xe])

        for zs, ys in x_start_intersections:
            results.append([zs, ys, x_start])

    return np.array(results)

def get_intersections_bbox(vein_mask, skeleton, spacing):
    bbox = get_bbox(vein_mask, spacing)
    print(bbox)
    intersections = get_intersections_bbox_helper(skeleton, bbox)

    print(intersections.shape)

    intersections_mask = np.zeros_like(vein_mask, dtype=np.uint16)

    for coords in intersections:
        intersections_mask[tuple(coords)] = 1

    return intersections_mask

def get_intersections_bbox_2(vein_mask, skeleton, spacing):
    bboxes = get_bbox(vein_mask, spacing)
    intersections = np.zeros_like(vein_mask)

    for index, bbox in enumerate(bboxes):
        z_start, z_end = bbox[0].start, bbox[0].stop
        y_start, y_end = bbox[1].start, bbox[1].stop
        x_start, x_end = bbox[2].start, bbox[2].stop

        z_start_intersections = np.array([
            (y, x) for (y, x) in np.argwhere(skeleton[z_start, :, :])
            if (y_start <= y <= y_end) and (x_start <= x <= x_end)
        ])
        z_end_intersections = np.array([
            (y, x) for (y, x) in np.argwhere(skeleton[z_end, :, :])
            if (y_start <= y <= y_end) and (x_start <= x <= x_end)
        ])
        y_start_intersections = np.array([
            (z, x) for (z, x) in np.argwhere(skeleton[:, y_start, :])
            if (z_start <= z <= z_end) and (x_start <= x <= x_end)
        ])
        y_end_intersections = np.array([
            (z, x) for (z, x) in np.argwhere(skeleton[:, y_end, :])
            if (z_start <= z <= z_end) and (x_start <= x <= x_end)
        ])
        if index == 0:
            x_intersections = np.array([
                (z, y) for (z, y) in np.argwhere(skeleton[:, :, x_start])
                if (z_start <= z <= z_end) and (y_start <= y <= y_end)
            ])
        else:
            x_intersections = np.array([
                (z, y) for (z, y) in np.argwhere(skeleton[:, :, x_end])
                if (z_start <= z <= z_end) and (y_start <= y <= y_end)
            ])

        intersections[z_start, z_start_intersections[:, 0], z_start_intersections[:, 1]] = 1
        intersections[z_end, z_end_intersections[:, 0], z_end_intersections[:, 1]] = 1
        intersections[y_start_intersections[:, 0], y_start, y_start_intersections[:, 1]] = 1
        intersections[y_end_intersections[:, 0], y_end, y_end_intersections[:, 1]] = 1
        if index == 0:
            intersections[x_intersections[:, 0], x_intersections[:, 1], x_start] = 1
        else:
            intersections[x_intersections[:, 0], x_intersections[:, 1], x_end] = 1

    return intersections


def main():
    vessel_entry = sitk.ReadImage("code/result_masks/vessel_entry.nii.gz")
    vessel_entry = sitk.GetArrayFromImage(vessel_entry)
    vessel_skeleton = sitk.ReadImage("dataset/skeleton/005_vein_mask_skeleton.nii.gz")
    vessel_skeleton = sitk.GetArrayFromImage(vessel_skeleton)

    print(vessel_entry.shape)
    print(vessel_skeleton.shape)

    vein_mask = scan_to_np_array('code/result_masks/005_pulmonary_result_masked.nii')
    ct_image_spacing =  [0.7109375, 0.7109375, 1.0]
    ct_image_spacing = ct_image_spacing[::-1]

    intersections = get_intersections_bbox_2(vein_mask, vessel_skeleton, ct_image_spacing)

    # non_zero_coords = np.where(intersections > 0)
    # print(non_zero_coords)

    # print(np.unique(intersections))

    intersections_sitk = sitk.GetImageFromArray(intersections)
    sitk.WriteImage(intersections_sitk, "dataset/intersections/005_vessel_intersections_bbox.nii.gz")

    # NO EDGE DETECTION DONE
    # intersections_ne = get_intersection(vessel_entry, vessel_skeleton)

    # intersections_ne_sitk = sitk.GetImageFromArray(intersections_ne)
    # sitk.WriteImage(intersections_ne_sitk, "dataset/intersections/005_vessel_intersections_ne.nii.gz")

    # EDGE DETECTION ON 2D SLICES
    # vessel_entry_2ded = detect_edges_2d(vessel_entry)

    # view_scan([vessel_entry_2ded])

    # vessel_entry_2ded_sitk = sitk.GetImageFromArray(vessel_entry_2ded)
    # sitk.WriteImage(vessel_entry_2ded_sitk, 'dataset/skeleton/005_vessel_edge_2d.nii.gz')

    # intersections_2ded = get_intersection(vessel_entry_2ded, vessel_skeleton)

    # print(np.unique(intersections_2ded))

    # view_scan([intersections_2ded])

    # intersections_2ded_sitk = sitk.GetImageFromArray(intersections_2ded)
    # sitk.WriteImage(intersections_2ded_sitk, 'dataset/intersections/005_vessel_intersections_2ded.nii.gz')

    # EDGE DETECTION 3D
    # vessel_entry_3ded = detect_edges_3d_sitk(vessel_entry)
    # print(np.unique(vessel_entry_3ded))

    # vessel_entry_3ded_sitk = sitk.GetImageFromArray(vessel_entry_3ded)
    # sitk.WriteImage(vessel_entry_3ded_sitk, "dataset/skeleton/005_vessel_edge_3d_sitk.nii.gz")

    # intersections_3ded = get_intersection(vessel_entry_3ded, vessel_skeleton)
    # print(intersections_3ded.shape)
    # intersections_3ded_sitk = sitk.GetImageFromArray(intersections_3ded)
    # sitk.WriteImage(intersections_3ded_sitk, "dataset/intersections/005_vessel_intersections_3ded_sitk.nii.gz")


if __name__ == "__main__":
    main()