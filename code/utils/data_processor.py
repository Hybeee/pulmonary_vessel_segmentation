import numpy as np
import os
import SimpleITK as sitk
from skimage.morphology import skeletonize
from utils.bbox_code import get_bbox
from utils.ts_util import get_pulmonary_mask
import utils.skeleton_traverser as traverser


class DataPreparer:
    """
    DataHandler class that prepares data for training a model to iteratively predict pulmonary arteries and veins. The inputs are the parameters described below.\n
    The preparation consists of the following stages - for each input CT scan and their respective artery/vein mask, spacing values and optionally provided pulmonary mask:
        1. If no pulmonary masks were provided the class creates them for each input CT scan by utilizing a TS (TotalSegmentator) model
        2. Based on the provided/generated pulmonary mask, bounding boxes are generated that contain the exit points of pulmonary veins and arteries from the heart.
        3. Based on the input artery and vein masks, their skeletons are generated
        4. The intersection points of the skeletons and the bounding boxes are generated
        5. Both skeletons are traversed from each intersection point - thus each traversal consists of at most n amount of graph component traversals
           where each component equals to the subcomponent of the entire skeleton's graph's that contains the intersection point. The traversal consists of multiple step and
           the skeleton bit that was traversed during a given step will be used as ground truth value during training - the original mask will be reconstructed from the skeleton
           using Fast Marching Method (FMM).
    
    NOTE: As mentioned above this process is done for each input CT scan, artery/vein mask, spacing value and optionally provided pulmonary mask tuple. Thus the class expects
    inputs in the form of np.ndarray where the ith element in each input array (CT scan, artery mask, vein mask, spacing, etc.) belongs to the same data point. 

    PARAMETERS
    ----------
    cts: np.ndarray
        input ct images
    artery_masks: np.ndarray
        input artery masks
    vein_masks: np.ndarray
        input vein masks
    spacing: np.ndarray
        spacing of the ct images
    pulmonary_masks: np.ndarray = None:
        Pulmonary vein binary masks for the ct scans. If not given the class creates it by utilizing TotalSegmentator
    verbose: bool = False:
        If set to True, print statements inform the caller about the process'/pipeline's state
    """
    def __init__(self, cts: np.ndarray,
                 artery_masks: np.ndarray,
                 vein_masks: np.ndarray,
                 spacings: np.ndarray,
                 pulmonary_masks: np.ndarray = None,
                 verbose: bool =False):
        # NORMAL ATTRIBUTES
        self.cts = cts
        self.artery_masks = artery_masks
        self.vein_masks = vein_masks
        self.spacings = spacings
        if pulmonary_masks is None:
            if verbose:
                print("No pulmonary masks were given. Utilizing TotalSegmentator to create them.")
            self.pulmonary_masks = self.make_pulmonary_masks(self.cts)
        else:
            self.pulmonary_masks = pulmonary_masks

        # DEPENDENT ATTRIBUTES

        # bbox init - same for both type of masks
        if verbose:
            print("Creating bounding boxes.")
        self.bboxs = self.get_bboxs(self.pulmonary_masks, self.spacings)

        # skeleton init
        if verbose:
            print("Creating skeletons.")
        self.artery_skeletons = self.get_skeletons(self.artery_masks)
        self.vein_skeletons = self.get_skeletons(self.vein_masks)

        # intersection init
        if verbose:
            print("Creating intersections")
        self.artery_intersections = self.get_intersections(self.artery_masks, self.artery_skeletons, self.bboxs)
        self.vein_intersections = self.get_intersections(self.vein_masks, self.vein_skeletons, self.bboxs)

        # creating interseciton objects for graph traversal
        if verbose:
            print("Traversing graphs and creating traversed paths.")
        self.traversed_artery_paths = self.traverse_graph(self.artery_skeletons,
                                                             self.artery_intersections,
                                                             self.bboxs)
        self.traversed_vein_paths = self.traverse_graph(self.vein_skeletons,
                                                        self.vein_intersections,
                                                        self.bboxs)

    def make_pulmonary_masks(self, cts: np.ndarray) -> np.ndarray:
        """
        Retrieves the pulmonary masks for the ct images in cts parameter
        """
        pulmonary_masks = []

        for ct in cts:
            pulmonary_mask = get_pulmonary_mask(ct)
            pulmonary_masks.append(pulmonary_mask)
        
        return np.array(pulmonary_masks)

    def get_bboxs(self, pulmonary_masks: np.ndarray, spacings: np.ndarray) -> np.ndarray:
        """
        Retrieves the bounding boxes of the ct images.\n
        Needed later to determine intersection points.
        """
        bboxs = []
        for mask, spacing in zip(pulmonary_masks, spacings):
            bbox = get_bbox(mask, spacing)
            bboxs.append(bbox)

        return np.array(bboxs)

    def get_skeletons(self, masks: np.ndarray) -> np.ndarray:
        """
        Creates the skeletons for each input mask in masks.
        Uses skimage.morpoholy.skeletonize(...).
        """
        skeletons = []
        for mask in masks:
            skeleton = skeletonize(mask).astype(np.uint16)
            skeletons.append(skeleton)

        return np.array(skeletons)
    
    def get_intersections(self, masks: np.ndarray, skeletons: np.ndarray, bboxs: np.ndarray) -> np.ndarray:
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
    
    def traverse_graph(self, skeletons: np.ndarray, intersections: np.ndarray, bboxs: np.ndarray) -> np.ndarray:
        """
        Utilizes code/utils/skeleton_traverser.py to traverse the skeletons and generate the traversed paths which will later be used to create labels for training.
        """
        traversed_paths = []
        for skeleton, curr_intersections in zip(skeletons, intersections):
            graph = traverser.get_graph(skeleton=skeleton)
            intersection_obj_list = traverser.create_intersection_objects(intersections=curr_intersections, graph=graph, bboxs=bboxs)
            _, curr_traversed_paths = traverser.traverse_graph(graph=graph,
                                                           intersection_obj_list=intersection_obj_list, 
                                                           bboxs=bboxs) 
            # returns traversed nodes too, currently not needed, but won't remove that functionality yet

            traversed_paths.append(curr_traversed_paths)
        
        return traversed_paths

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