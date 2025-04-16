import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk

def convert_npz_to_nii_image(folder, out_folder):
    """
    .npz to .nii converter.
    """
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    for name in os.listdir(folder):
        image = np.load(f"{folder}/{name}", allow_pickle=True)["data"]
        image = image.transpose(2, 0, 1)
        image = sitk.GetImageFromArray(image)
        sitk.WriteImage(image, f"{out_folder}/{name[:3]}.nii.gz")

def main():
    print("Hello world!")

    convert_npz_to_nii_image('dataset/HiPaS/ct_scan', 'dataset/HiPaS/ct_scan_nii')
    print("ct done")
    convert_npz_to_nii_image('dataset/HiPaS/annotation/artery', 'dataset/HiPaS/annotation/artery_nii')
    print("artery done")
    convert_npz_to_nii_image('dataset/HiPaS/annotation/vein', 'dataset/HiPaS/annotation/vein_nii')
    print("vein done")


if __name__ == "__main__":
    main()