from unittest import TestCase

from torch.utils.data import DataLoader
import sys
sys.path.insert(0, "../")
from deepclustering.dataloader.sampler import InfiniteRandomSampler
from dataloader import H5Dataset

class TestDataLoader(TestCase):
    def setUp(self):
        super().setUp()
        self.train_set = H5Dataset(root_path="../data_train_nocut", crop_size=(64, 64, 64), mode="train")

    def test_dataset(self):
        for i in range(len(self.train_set)):
            img, target = self.train_set[i]

    def test_dataloader(self):
        dataloader = DataLoader(dataset=self.train_set, batch_size=4)
        for i, data in enumerate(dataloader):
            img, target = data
            assert img.shape[0] == 4
            assert target.shape[0] == 4
        dataloader = DataLoader(dataset=self.train_set, batch_size=4, num_workers=4)
        for i, data in enumerate(dataloader):
            img, target = data
            assert img.shape[0] == 4
            assert target.shape[0] == 4

        dataloader = DataLoader(dataset=self.train_set, batch_size=4, num_workers=4, sampler=InfiniteRandomSampler(self.train_set))
        for i, data in enumerate(dataloader):
            img, target = data
            assert img.shape[0] == 4
            assert target.shape[0] == 4
            if i==1000:
                break


