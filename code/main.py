from utils.data_processor import read_images_from_files, resample_image
from utils.scan_plotter import view_scan
from utils.ts_util import get_pulmonary_mask
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
    view_scan(ct)
    ct = resample_image(ct, spacings[4], (225, 341, 341))
    view_scan(ct)
    print(ct.shape)

def make_pulmonary_masks():
    data_root = "dataset/HiPaS"
    metadata = pd.read_excel(f'{data_root}/metadata.xlsx')
    spacings = np.array([ast.literal_eval(spacing) for spacing in metadata['Resolution']])

    cts = read_images_from_files(f"{data_root}/ct_scan_nii", spacings, verbose=True)

    # print(cts.shape)

    # mask = get_pulmonary_mask(cts[4])

    # print(mask.shape)

    # mask_sitk = sitk.GetImageFromArray(mask)
    # sitk.WriteImage(mask_sitk, "code/temp_images/temp_mask.nii.gz")

    pulmonary_masks = []

    for ct in cts:
        pulmonary_mask = get_pulmonary_mask(ct)
        pulmonary_masks.append(pulmonary_mask)
    
    pulmonary_masks = np.array(pulmonary_masks)

    i = 1
    for mask in pulmonary_masks:
        mask_sitk = sitk.GetImageFromArray(mask)
        sitk.WriteImage(mask_sitk, f"code/temp_images/temp_pulmonary_masks/00{i}_pulmonary_mask.nii.gz")
        i += 1

def main():
    # resample_tester()
    make_pulmonary_masks()
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