import numpy as np
import SimpleITK as sitk
from numba import jit

# ONLY FOR TESTING
from utils.scan_plotter import view_scan


def neighbors(shape):
    """
    Given a pixel in the 2D/3D space returns an array containing its neighbouring pixel/voxel indices.
    """

    dim = len(shape)
    block = np.ones([3] * dim)
    block[tuple([1] * dim)] = 0
    idx = np.where(block > 0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx - [1] * dim)
    acc = np.cumprod((1,) + shape[::-1][:-1])
    return np.dot(idx, acc[::-1])

def create_deskeleton_map(skeleton, mask_orig, pixel_spacing):
    assert skeleton.shape == mask_orig.shape, 'All arrays must have the same shape'
    skeleton = skeleton.astype(np.uint8)
    fast_marching = sitk.FastMarchingImageFilter()
    seed_points = np.argwhere(skeleton == 1)[:, ::-1]  # shape: (3, -1) | seed_point <=> fast marching starts from here, needed for trial points.
    fast_marching.SetTrialPoints(seed_points.tolist())
    speed_image = mask_orig.astype(np.uint8)
    speed_image_sitk = sitk.GetImageFromArray(speed_image)
    speed_image_sitk.SetSpacing(pixel_spacing[::-1])  # sitk needs it reversed
    fmm_res = fast_marching.Execute(speed_image_sitk)
    travel_times = sitk.GetArrayFromImage(fmm_res)

    travel_times = np.pad(travel_times, (1, 1), mode='constant', constant_values=0)
    mask_orig = np.pad(mask_orig, (1, 1), mode='constant', constant_values=0)
    nbs = neighbors(mask_orig.shape)

    n_dim = skeleton.ndim
    min_neighbor_map = find_min_neighbor(travel_times, mask_orig, nbs)
    min_neighbor_map = np.where(mask_orig==0, -1, min_neighbor_map) # -1 outside the mask
    # unravel
    min_neighbor_map = min_neighbor_map[tuple([slice(1, -1)] * n_dim)]
    min_neighbor_map = np.where(skeleton==1, -1, min_neighbor_map) # -1 along the skeleton

    return min_neighbor_map

@jit(nopython=True, cache=True)
def find_min_neighbor(times, mask, nbs):
    shape_orig = times.shape
    times = times.ravel()
    mask = mask.ravel()
    min_neighbor_map = np.zeros(mask.shape, dtype=np.int8)

    for p in range(len(times)):
        min_val = np.inf
        min_pos = -1
        if mask[p] == 0: continue
        for ni in range(len(nbs)):
            dp = nbs[ni]
            cp = p + dp
            if mask[cp] == 0: continue
            nb_val = times[cp]
            if (nb_val < times[p]) and (nb_val < min_val):
                min_val = nb_val
                min_pos = ni

        if min_pos > -1:
            min_neighbor_map[p] = min_pos

    return min_neighbor_map.reshape(shape_orig)

def deskeletonize(skeleton, mask_orig, deskeleton_map):
    assert skeleton.shape == deskeleton_map.shape == mask_orig.shape, 'All arrays must have the same shape'
    skeleton = np.pad(skeleton, (1, 1), mode='constant', constant_values=0)
    mask_orig = np.pad(mask_orig, (1, 1), mode='constant', constant_values=0)
    deskeleton_map = np.pad(deskeleton_map, (1, 1), mode='constant', constant_values=0)
    nbs = neighbors(skeleton.shape)

    n_dim = skeleton.ndim
    deskeletonized = deskeletonize_helper(skeleton, mask_orig, deskeleton_map, nbs)
    # unravel
    deskeletonized = deskeletonized[tuple([slice(1, -1)] * n_dim)]

    return deskeletonized

@jit(nopython=True, cache=True)
def deskeletonize_helper(skeleton, mask_orig, deskeleton_map, nbs):
    shape_orig = skeleton.shape
    skeleton = skeleton.astype(np.uint8).ravel()
    mask_orig = mask_orig.astype(np.uint8).ravel()
    deskeleton_map = deskeleton_map.astype(np.uint8).ravel()
    deskeletonized = skeleton.copy()

    n_lists = 2
    current_list = 0
    to_update_list = (current_list + 1) % n_lists
    to_update_idx = 0
    lists = [np.zeros(skeleton.shape[0], dtype=np.int64),
             np.zeros(skeleton.shape[0], dtype=np.int64)]

    for p in range(len(skeleton)):
        if skeleton[p] == 0: continue
        lists[to_update_list][to_update_idx] = p
        to_update_idx += 1

    while True:
        current_list = (current_list + 1) % n_lists
        to_update_list = (current_list + 1) % n_lists
        current_list_size = to_update_idx
        to_update_idx = 0

        if current_list_size == 0:
            break
        # print(current_list_size)

        for current_idx in range(current_list_size):
            p = lists[current_list][current_idx]
            skeleton_value = deskeletonized[p]
            for nb_i in range(len(nbs)):
                p_nb = p + nbs[nb_i] # index of neighbour
                if mask_orig[p_nb] == 0: continue # not part of mask
                if deskeleton_map[p_nb] == -1: continue # pixel is part of skeleton
                if deskeletonized[p_nb] != 0: continue # TODO optimize: we can use mask orig for this purpose (it's a copy), also ravel is just a view | neighbour is already part of reconstructed skeleton
                nb_source_arg_d = deskeleton_map[p_nb]
                nb_source_d = nbs[nb_source_arg_d]
                nb_source = p_nb + nb_source_d
                if nb_source != p: continue
                deskeletonized[p_nb] = skeleton_value
                lists[to_update_list][to_update_idx] = p_nb
                to_update_idx += 1

    return deskeletonized.reshape(shape_orig)

def segment_to_volume(coords: np.ndarray, volume_shape):
    """
    Convert a list of coordinates to a binary volume.
    
    coords: list or array of coordinates (each coordinate is a 3-element array)
    volume_shape: tuple with the shape of the volume (e.g. skeleton.shape)
    """

    volume = np.zeros(volume_shape, dtype=np.uint8)
    for c in coords:
        volume[tuple(c)] = 1
    return volume

def main():
    ct = sitk.ReadImage("dataset/HiPaS/ct_scan_nii/005.nii.gz")
    ct = sitk.GetArrayFromImage(ct)
    vein_mask = sitk.ReadImage("dataset/HiPaS/annotation/vein_nii/005.nii.gz")
    vein_mask = sitk.GetArrayFromImage(vein_mask)

    spacing = [0.7109375, 0.7109375, 1.0]

    skeleton = sitk.ReadImage('dataset/skeleton/005_vein_mask_skeleton.nii.gz')
    skeleton = sitk.GetArrayFromImage(skeleton)

    segment_coords = [np.array([103, 257, 213], dtype=np.int64),
                  np.array([101, 257, 214], dtype=np.int16),
                  np.array([100, 256, 214], dtype=np.int16)]

    segment_coords = np.array(segment_coords)

    deskeleton_map = create_deskeleton_map(skeleton, vein_mask, spacing)

    skeleton = segment_to_volume(segment_coords, skeleton.shape)

    deskeletonized = deskeletonize(skeleton, vein_mask, deskeleton_map)

    view_scan([deskeletonized])

if __name__ == "__main__":
    main()