from torch.utils.data import Dataset
from data_preparer import DataPreparer


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

        self.spacings = dataPreparer.spacings

        self.transform = transform

    def __len__(self):
        return len(self.cts)
    
    def __geitem__(self, index):
        # Is transform even needed? only for cts?
        ct = self.cts[index]
        
        artery_mask = self.artery_masks[index]
        artery_skeleton = self.artery_skeletons[index]
        traversed_artery_paths = self.traversed_arteries[index]

        vein_mask  =self.vein_masks[index]
        vein_skeleton = self.vein_skeletons[index]
        traversed_vein_paths = self.traversed_veins[index]

        spacing = self.spacings[index]

        return self.transform(ct), artery_mask, artery_skeleton, traversed_artery_paths, vein_mask, vein_skeleton, traversed_vein_paths, spacing