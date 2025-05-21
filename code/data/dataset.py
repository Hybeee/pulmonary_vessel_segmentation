from torch.utils.data import Dataset
from data_preparer import DataPreparer


class DataPoint:
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