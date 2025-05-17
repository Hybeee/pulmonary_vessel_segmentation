from skimage.morphology import skeletonize
import sknw
from utils.bbox_code import get_bbox
import numpy as np
import os
import SimpleITK as sitk
from utils.ts_util import get_pulmonary_mask

class DataHandler:
    """
    DataHandler class that continuously creates the labels during the iterative training.
    Expects ct scan tensors with a batch dimensions - input should have the shape of [batch, z, y, x], where z, y and x are the shape of the ct scans/images.
    Input arrays will be tensors on GPU hence .cpu().numpy() is called at several points of the class. NOTE: needed?? will see
    Initialized by the following steps:\n
        - Creates the skeleton to the original mask of the CT image (meaning vein and artery masks)
        - For each ct scan determines the intersection points of a given bounding box and the skeleton

    TODO: The class will probably need to be rewritten into a Dataset class (torch.utils.data.Dataset). This is because this class will have
          features that will be used during training. Thus it makes more sense to integrate Dataset related features here rather than splitting it
          accross multiple classes.

    PARAMETERS
    ----------
    cts:
        input ct images - will probably be a tensor
    masks:
        input masks - will probably be a tensor
    spacing:
        spacing of the ct images - will probably be an ndarray
    pulmonary_masks = None:
        Pulmonary vein binary masks for the ct scans - if provided by caller, will probably be an ndarray
    """
    def __init__(self, cts: np.ndarray, artery_masks: np.ndarray, vein_masks: np.ndarray, spacings: np.ndarray, pulmonary_masks: np.ndarray = None):
        # NORMAL ATTRIBUTES
        self.cts = cts
        self.artery_masks = artery_masks
        self.vein_masks = vein_masks
        self.spacings = spacings
        if pulmonary_masks is None:
            self.pulmonary_masks = self.make_pulmonary_masks()
        else:
            self.pulmonary_masks = pulmonary_masks

        # DEPENDENT ATTRIBUTES

        # bbox init - same for both type of masks
        self.bboxs = self.get_bboxs()

        # skeleton init
        self.artery_skeletons = self.get_skeletons(self.artery_masks)
        self.vein_skeletons = self.get_skeletons(self.vein_masks)

        # intersection init
        self.artery_intersections = self.get_intersections(self.artery_masks, self.artery_skeletons, self.bboxs)
        self.vein_intersections = self.get_intersections(self.vein_masks, self.vein_skeletons, self.bboxs)

    def make_pulmonary_masks(self):
        """
        Retrieves the pulmonary masks for the ct images in cts attribute
        """
        pulmonary_masks = []

        for ct in self.np_cts:
            pulmonary_mask = get_pulmonary_mask(ct)
            pulmonary_masks.append(pulmonary_mask)
        
        return np.array(pulmonary_masks)

    def get_bboxs(self):
        """
        Retrieves the bounding boxes of the ct images.\n
        Needed later to determine intersection points.
        """
        bboxs = []
        for mask, spacing in zip(self.np_pulmonary_masks, self.np_spacings):
            bbox = get_bbox(mask, spacing)
            bboxs.append(bbox)

        return np.array(bboxs)

    def get_skeletons(self, masks):
        """
        Creates the skeletons for each input mask in masks.
        Uses skimage.morpoholy.skeletonize(...).
        """
        skeletons = []
        for mask in masks:
            skeleton = skeletonize(mask).astype(np.uint16)
            skeletons.append(skeleton)

        return np.array(skeletons)
    
    def get_intersections(self, masks, skeletons, bboxs):
        """
        Calculates the intersections for each skeleton and bounding box pair..
        """
        intersections = []

        for mask, skeleton, mask_bbox in zip(masks, skeletons, bboxs):
            curr_intersections = np.zeros_like(mask)

            for index, bbox in enumerate(mask_bbox):
                z_start, z_end = bbox[0].start, bbox[0].stop
                y_start, y_end = bbox[1].start, bbox[1].stop
                x_start, x_end = bbox[2].start, bbox[2].end

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
                    if(z_start <= z <= z_end) and (x_start <= x <= x_end)
                ])
                if index == 0:
                    x_intersections = np.array([
                        (z, y) for (z, y) in np.argwhere(skeleton[:, :, x_start])
                        if (z_start <= z <= z_end) and (y_start <= y <= y_end)
                    ])
                else:
                    x_intersections = np.array([
                        (z, y) for(z, y) in np.argwhere(skeleton[:, :, x_end])
                        if (z_start <= z <= z_end) and (y_start <= y <= y_end)
                    ])

            curr_intersections[z_start, z_start_intersections[:, 0], z_start_intersections[:, 1]] = 1
            curr_intersections[z_end, z_end_intersections[:, 0], z_end_intersections[:, 1]] = 1
            curr_intersections[y_start_intersections[:, 0], y_start, y_start_intersections[:, 1]] = 1
            curr_intersections[y_end_intersections[:, 1], y_end, y_end_intersections[:, 1]] = 1
            if index == 0:
                curr_intersections[x_intersections[:, 0], x_intersections[:, 1], x_start] = 1
            else:
                curr_intersections[x_intersections[:, 0], x_intersections[:, 1], x_end] = 1
        
        intersections.append(curr_intersections)

        return np.array(intersections)

def resample_image(image, original_spacing, target_shape):
    """
    Resamples a ct image to a given target_shape.
    New image's spacing will be calculated based on targetshape with the following formula:
    original_spacing * (original_size / target_shape) // each being a 3dim vector in current project
    """
    original_size = np.array(image.shape)
    new_spacing = original_spacing * (original_size / np.array(target_shape))

    sitk_image = sitk.GetImageFromArray(image)
    sitk_image.SetSpacing(original_spacing[::-1]) # by default spacing is set to [1, 1, 1]

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize([int(size) for size in target_shape[::-1]]) # expects x, y, z
    resample_filter.SetOutputSpacing(new_spacing[::-1].tolist()) # expects x, y, z
    resample_filter.SetOutputOrigin(sitk_image.GetOrigin())
    resample_filter.SetOutputDirection(sitk_image.GetDirection())

    resample_filter.SetInterpolator(sitk.sitkLinear)

    resampled_image = resample_filter.Execute(sitk_image)

    return sitk.GetArrayFromImage(resampled_image)

def read_images_from_files(folder_path, spacings, verbose=False) -> np.ndarray:
    """
    Input is a folder containing .nii/.nii.gz or .npz files.
    The function reads those files and returns them in a numpy array.
    """
    # Resampling: 341x341x225 or 225x341x341?

    if verbose:
        print(f"Reading scans from: {folder_path}")

    images = []
    i = 0
    for name, spacing in zip(os.listdir(folder_path), spacings):
        if i == 5:
            break
        file_path = f"{folder_path}/{name}"
        if name[-3:] == "npz":
            image = np.load(file_path, allow_pickle=True)["data"]
            image = image.transpose(2, 0, 1) # for HiPaS dataset at least image is stored as [y, x, z] -> [z, y, x]
        else:
            image = sitk.ReadImage(file_path)
            image = sitk.GetArrayFromImage(image)
        image = resample_image(image, spacing, (225, 341, 341))
        images.append(image)
        i += 1
    
    return np.array(images)