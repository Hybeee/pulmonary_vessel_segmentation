import nibabel as nib
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
import numpy as np
import torch

def mask_result(save_file_name, result_path, class_indices):
    """
    Masking a segmentation result with given class indices - the resulting .nii image will only contain the classes whose index is in class_indices. 
    """
    if class_indices is not np.array:
        class_indices = np.array(class_indices)
    seg_nifti_img, _ = load_multilabel_nifti(result_path)

    segmentation_mask = seg_nifti_img.get_fdata() 

    mask = (np.isin(segmentation_mask, class_indices)).astype(int)
    print(f'Saving segmentation mask to: code/result_masks/{save_file_name}.nii')
    nib.save(nib.Nifti1Image(mask, None), f'code/result_masks/{save_file_name}.nii')

def run_TS_on_image(scan_name, scan_path, task="total", class_indices=None):
    """
    Runs TotalSegmentator on a given scan - at scan_path -, saves the segmentation and if class_indices is not None, then it will further mask the segmentation.
    """
    input_img = nib.load(scan_path)
    output_img = totalsegmentator(input_img, task=task, verbose=True)

    output_path = f"code\\demo_results\\{scan_name}_{task}_output.nii"
    print(f"Segmentation result saved to: {output_path}")
    nib.save(output_img, output_path)

    if class_indices:
        mask_result(f'{scan_name}_result_masked', output_path, class_indices)


def main():

    scan_name = "mytemp"
    scan_path = 'code/temp_images/temp_ct.nii.gz'

    # scan_name = "bbox_scan"
    # scan_path = "dataset/bbox/ct_for_bbox.nii.gz"

    run_TS_on_image(scan_name, scan_path, task="total")

if __name__ == "__main__":
    # print(torch.cuda.is_available())
    main()
    # seg_nifti_img, label_map_dict = load_multilabel_nifti("C:\\BME\\mester\\1_felev\\onlab_1\\code\\demo_results\\005_total_output.nii")

    # print(seg_nifti_img)
    # print("------------------")
    # print(label_map_dict) # contains {index: class_name} key-value pairs