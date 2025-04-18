from utils.data_processor import read_images_from_files, resample_image
import pandas as pd
import numpy as np
import ast
import SimpleITK as sitk

def resample_tester():
    data_root = "dataset/HiPaS"
    metadata = pd.read_excel(f'{data_root}/metadata.xlsx')

    spacings = np.array([ast.literal_eval(spacing) for spacing in metadata['Resolution']])

    ct = sitk.ReadImage("dataset/HiPaS/ct_scan_nii/005.nii.gz")
    ct = sitk.GetArrayFromImage(ct)

    print(ct.shape)
    ct = resample_image(ct, spacings[4], (225, 341, 341))
    print(ct.shape)

def main():
    resample_tester()
    # print("Reading data...")
    # data_root = "dataset/HiPaS"
    # metadata = pd.read_excel(f'{data_root}/metadata.xlsx')

    # spacings = np.array([ast.literal_eval(spacing) for spacing in metadata['Resolution']])

    # ct = sitk.ReadImage("dataset/HiPaS/ct_scan_nii/005.nii.gz")
    # ct = sitk.GetArrayFromImage(ct)

    # print(ct.shape)
    # ct = resample_image(ct, spacings[4], (225, 341, 341))
    # print(ct.shape)

    # cts = read_images_from_files(f"{data_root}/ct_scan_nii", spacings)
    # print("cts read!")
    # vein_masks = read_images_from_files(f"{data_root}/annotations/vein_nii", spacings)
    # print("vein read!")
    # artery_masks = read_images_from_files(f"{data_root}/annotations/artery_nii", spacings)
    # print("artery read!")

    # print(f"Shape of ct scans: {cts.shape}")
    # print(f"Shape of vein masks: {vein_masks.shape}")
    # print(f"Shape of artery masks: {artery_masks}")


if __name__ == "__main__":
    main()