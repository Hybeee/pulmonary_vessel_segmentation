from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
import numpy as np
import SimpleITK as sitk

def extract_binary_mask(class_indices):
    """
    Converts a multi-class mask into a binary mask for the specified class index or indices.
    CURRENTLY NOT NEEDED, FOUND AN EASIER WAY TO MAKE THIS WORK!
    """
    seg_nifti_img, _ = load_multilabel_nifti("code/resources/temp_images/temp_ts/output_mask.nii")
    mask = seg_nifti_img.get_fdata()

    if not isinstance(class_indices, np.ndarray):
        class_indices = np.array(class_indices)
    
    binary_mask = (np.isin(mask, class_indices)).astype(int)
    return binary_mask

def get_pulmonary_mask(ct):
    """
    Runs the 'total' segmentation task from TS on the given CT scan,
    and returns a binary mask corresponding to class index 53 (pulmonary vein).
    """
    print("Generating pulmonary_mask...")
    ct_sitk = sitk.GetImageFromArray(ct)
    ct_path = "code/resources/temp_images/temp_ct.nii.gz"
    sitk.WriteImage(ct_sitk, ct_path)

    # output_mask = totalsegmentator(ct_path, task='total', output="code/resources/temp_images/temp_ts_output")
    pulmonary_mask = sitk.ReadImage("code/resources/temp_images/temp_ts_output/pulmonary_vein.nii.gz")
    pulmonary_mask = sitk.GetArrayFromImage(pulmonary_mask)
    return pulmonary_mask