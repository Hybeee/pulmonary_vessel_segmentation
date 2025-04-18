from utils.data_processor import read_images_from_files
import pandas as pd
import numpy as np
import ast

def main():
    print("Reading data...")
    data_root = "dataset/HiPaS"
    metadata = pd.read_excel(f'{data_root}/metadata.xlsx')

    spacings = np.array([ast.literal_eval(spacing) for spacing in metadata['Resolution']])

    # cts = read_images_from_files(f"{data_root}/ct_scan_nii")
    # print("cts read!")
    # vein_masks = read_images_from_files(f"{data_root}/annotations/vein_nii")
    # print("vein read!")
    # artery_masks = read_images_from_files(f"{data_root}/annotations/artery_nii")
    # print("artery read!")

    # print(f"Shape of ct scans: {cts.shape}")
    # print(f"Shape of vein masks: {vein_masks.shape}")
    # print(f"Shape of artery masks: {artery_masks}")


if __name__ == "__main__":
    main()