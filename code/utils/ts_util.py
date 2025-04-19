from totalsegmentator.python_api import totalsegmentator
import numpy as np

def extract_binary_mask(mask, class_indices):
    """
    Converts a multi-class mask into a binary mask for the specified class index or indices.
    """
    if not isinstance(class_indices, np.ndarray):
        class_indices = np.array(class_indices)
    
    binary_mask = (np.isin(mask, class_indices)).astype(int)
    return binary_mask

def get_pulmonary_mask(ct):
    """
    Runs the 'total' segmentation task from TS on the given CT scan,
    and returns a binary mask corresponding to class index 53 (pulmonary vein).
    """
    output_mask = totalsegmentator(ct, task='total')
    return extract_binary_mask(output_mask, [53])