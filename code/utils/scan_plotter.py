import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import SimpleITK as sitk
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.nifti_ext_header import load_multilabel_nifti

def scan_to_np_array(scan_path):
    """
    Returns the read scan in the following shape: (z, y, x).
    """
    scan = sitk.ReadImage(scan_path) # shape: z, y, x
    print(f"scan spacing: {scan.GetSpacing()}")
    scan = sitk.GetArrayFromImage(scan)
    return scan

def scan_to_np_array_with_slice(scan_path):
    scan = sitk.ReadImage(scan_path) # shape: slice, width, height
    print(f"scan spacing: {scan.GetSpacing()}")
    spacing = scan.GetSpacing()
    scan = sitk.GetArrayFromImage(scan)
    return (scan, spacing)

def single_slider_update(fig, ax, slider_value, images):
    new_scan_slice = images[0][int(slider_value), :, :]
    ax.images[0].set_data(new_scan_slice)
    ax.images[0].norm.autoscale(new_scan_slice) # Usually needed for masks

    for i in range(len(images[1:])):
        i += 1
        new_mask_slice = images[i][int(slider_value), :, :]
        ax.images[i].set_data(new_mask_slice)
        ax.images[i].norm.autoscale(new_mask_slice)

    fig.canvas.draw_idle()

def alpha_slider_update(fig, ax, slider_value, images):
    for image in ax.images[1:]:
        image.set_alpha(slider_value)

    fig.canvas.draw_idle()

def combine_masks(images):
    combined_mask = np.max(np.stack(images[1:], axis=-1), axis=-1)

    return np.array([images[0], combined_mask])

def view_scan(images: list[np.ndarray] | np.ndarray, slice_slider_update=single_slider_update, alpha_slider_update=alpha_slider_update):
    
    if isinstance(images, np.ndarray):
        images = [images]

    if len(images) > 2:
        print('Combining masks!')
        images = combine_masks(images)

    initial_slice = 0
    initial_alpha = 0.5

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    axslice = fig.add_axes([0.2, 0.01, 0.65, 0.03])
    slice_slider = Slider(
        ax=axslice,
        label='Scan slider',
        valmin=0,
        valmax=images[0].shape[0] - 1, # scan has the shape of (height, width, num_of_slice)
        valinit=initial_slice,
        valstep=1
    )

    slice_slider.on_changed(
        lambda val : slice_slider_update(fig, ax, val, images)
    )

    if len(images) > 1:
        axalpha = fig.add_axes([0.2, 0.04, 0.65, 0.03])
        alpha_slider = Slider(
            ax=axalpha,
            label='Alpha slider',
            valmin=0.0,
            valmax=1.0,
            valinit=initial_alpha,
            valstep=0.05
        )

        alpha_slider.on_changed(
            lambda val : alpha_slider_update(fig, ax, val, images)
        )
    
    plt.axis('off')
    
    ax.imshow(images[0][initial_slice, :, :], cmap='gray')
    for image in images[1:]:
        ax.imshow(image[initial_slice, :, :], alpha=0.3)

    
    plt.show()


def main():
    # ct = np.load("dataset/HiPaS/ct_scan/005.npz", allow_pickle=True)["data"]
    # artery = np.load("dataset/HiPaS/annotation/artery/005.npz", allow_pickle=True)["data"]
    # vein = np.load("dataset/HiPaS/annotation/vein/005.npz", allow_pickle=True)["data"]

    # ct = ct.transpose(2, 0, 1)
    # # vein = vein.transpose(2, 0, 1)

    vessel_mask = scan_to_np_array('dataset/HiPaS/annotation/vein_nii/005.nii')
    vein_skeleton = scan_to_np_array('dataset/skeleton/005_vein_mask_skeleton.nii.gz')


    # slicer like plotting
    # vessel_mask = vessel_mask.transpose(2, 0, 1)[:, ::-1, :]
    # vein_skeleton = vein_skeleton.transpose(2, 0, 1)[:, ::-1, :]

    # print(vessel_mask.shape)
    # vessel_skeleton = scan_to_np_array('dataset/005_vessel_skeleton.nii')

    view_scan([vessel_mask, vein_skeleton])


if __name__ == "__main__":
    main()