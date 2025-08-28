from torch.utils.data import Dataset
from data.data_preparer import DataPreparer


class DataPoint:
    """
    Class that represents a single training data sample/point.

    A training sample consists of the following attributes/data structures:

        - The original CT scan
        - The mask of the artery network
        - The artery network's skeleton
        - The path segments of the artery skeleton on which the traversal is done
        - The mask of the vein network
        - The vein network's skeleton
        - The path segments of the vein skeleton on which the traversal is done
        - A tuple of bounding box pairs
            - A bounding box that covers the left side of the heart. Used to detect left side exit points of the artery/vein networks
            - A bounding box that covers the right side of the heart. Used to detect right side exit points of the artery/vein networks
        - The spacing meta information of the CT scan
    """
    def __init__(self, ct,
                 artery_mask, artery_skeleton, artery_paths,
                 vein_mask, vein_skeleton, vein_paths,
                 bbox_pair, spacing):
        self.ct = ct
        self.artery_mask = artery_mask
        self.artery_skeleton = artery_skeleton
        self.artery_paths = artery_paths
        self.vein_mask = vein_mask
        self.vein_skeleton = vein_skeleton
        self.vein_paths = vein_paths
        self.bbox_pair = bbox_pair
        self.spacing = spacing

class IterativeSegmentationDataset(Dataset):
    """
    Dataset class.

    Handles how training samples are accessed during training.

    When the ith data sample accessed it returns a DataPoint data class instance
    that contains all the relevant information/data structures to the given sample.
    """
    def __init__(self, dataPreparer: DataPreparer, transform):
        super(IterativeSegmentationDataset, self).__init__()
        self.cts = dataPreparer.cts

        self.artery_masks = dataPreparer.artery_masks
        self.artery_skeletons = dataPreparer.artery_skeletons
        self.traversed_arteries = dataPreparer.traversed_arteries

        self.vein_masks = dataPreparer.vein_masks
        self.vein_skeletons = dataPreparer.vein_skeletons
        self.traversed_veins = dataPreparer.traversed_veins

        self.bboxs = dataPreparer.bboxs
        self.spacings = dataPreparer.spacings

        self.transform = transform

    def __len__(self):
        return len(self.cts)
    
    def __getitem__(self, index) -> DataPoint:
        # Is transform even needed? only for cts?
        ct = self.cts[index]
        
        artery_mask = self.artery_masks[index]
        artery_skeleton = self.artery_skeletons[index]
        artery_paths = self.traversed_arteries[index]

        vein_mask  =self.vein_masks[index]
        vein_skeleton = self.vein_skeletons[index]
        vein_paths = self.traversed_veins[index]

        bbox_pair = self.bboxs[index]
        spacing = self.spacings[index]

        return DataPoint(ct=ct,
                         artery_mask=artery_mask, artery_skeleton=artery_skeleton, artery_paths=artery_paths,
                         vein_mask=vein_mask, vein_skeleton=vein_skeleton, vein_paths=vein_paths,
                         bbox_pair=bbox_pair, spacing=spacing)