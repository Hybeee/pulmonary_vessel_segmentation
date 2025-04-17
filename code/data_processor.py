from skimage.morphology import skeletonize
import sknw


class DataHandler:
    """
    DataHandler class that continuously creates the labels during the iterative training.
    Expects ct scan tensors with a batch dimensions - input should have the shape of [batch, z, y, x], where z, y and x are the shape of the ct scans/images
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
    def __init__(self, cts, masks, spacing):
        print("DataHandler initialized")
        self.init_cts = cts
        self.init_masks = masks