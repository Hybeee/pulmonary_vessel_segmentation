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
    NOTE: should bbox be created here? should bbox be given as a parameter

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

        self.skeletons = self.get_skeletons(masks)

    def get_skeletons(self, masks):
        """
        Creates the skeletons for each input mask.
        Uses skimage.morpoholy.skeletonize(...).
        """
        np_masks = masks.cpu().numpy()
        skeletons = []
        for mask in np_masks:
            skeleton = skeletonize(mask).astype(np.uint16)
            skeletons.append(skeleton)

        return np.array(skeletons)