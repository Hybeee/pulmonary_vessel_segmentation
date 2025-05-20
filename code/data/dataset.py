from torch.utils.data import Dataset
from data_preparer import DataPreparer


class IterativeSegmentationDataset(Dataset):
    def __init__(self, dataPreparer: DataPreparer, transform):
        super(IterativeSegmentationDataset, self).__init__()
        self.cts = dataPreparer.cts

        self.artery_masks = dataPreparer.artery_masks
        self.artery_skeletons = dataPreparer.artery_skeletons
        self.traversed_artery_paths = dataPreparer.traversed_artery_paths

        self.vein_masks = dataPreparer.vein_masks
        self.vein_skeletons = dataPreparer.vein_skeletons
        self.traversed_vein_paths = dataPreparer.traversed_vein_paths

        self.spacings = dataPreparer.spacings

        self.transform = transform