from skimage.morphology import skeletonize
import sknw
from bbox_code import get_bbox
import numpy as np

class DataHandler:
    """
    DataHandler class that continuously creates the labels during the iterative training.
    Expects ct scan tensors with a batch dimensions - input should have the shape of [batch, z, y, x], where z, y and x are the shape of the ct scans/images.
    Input arrays will be tensors on GPU hence .cpu().numpy() is called at several points of the class. NOTE: needed?? will see
    Initialized by the following steps:\n
        - Creates the skeleton to the original mask of the CT image (meaning vein and artery masks)
        - For each ct scan determines the intersection points of a given bounding box and the skeleton
    NOTE: should bbox be created here? should bbox be given as a parameter\n
    NOTE: Implementation of current class will probably change a lot in the future, the important part currently is the core.

    PARAMETERS
    ----------
    cts:
        input ct images
    masks:
        input masks
    spacing:
        spacing of the ct images
    """
    def __init__(self, cts, masks, spacings):
        # Initializing attributes
        self.cts = cts
        self.masks = masks
        self.spacings = spacings
        self.np_cts = cts.cpu().numpy()
        self.np_masks = masks.cpu().numpy()
        self.np_spacings = spacings.cpu().numpy()

        # Initializing dependent attributes
        self.bboxs = self.get_bboxs()
        self.skeletons = self.get_skeletons()
        self.intersections = self.get_intersections()

    def get_bboxs(self):
        """
        Retrieves the bounding boxes of the ct images.\n
        Needed later to determine intersection points.
        """
        bboxs = []
        for mask, spacing in zip(self.np_masks, self.np_spacings):
            bbox = get_bbox(mask, spacing)
            bboxs.append(bbox)

        return np.array(bboxs)

    def get_skeletons(self):
        """
        Creates the skeletons for each input mask.
        Uses skimage.morpoholy.skeletonize(...).
        """
        skeletons = []
        for mask in self.np_masks:
            skeleton = skeletonize(mask).astype(np.uint16)
            skeletons.append(skeleton)

        return np.array(skeletons)
    
    def get_intersections(self):
        """
        Calculates the intersections for each skeleton and bounding box pair..
        """
        intersections = []

        for mask, skeleton, mask_bbox in zip(self.np_masks, self.skeletons, self.bboxs):
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