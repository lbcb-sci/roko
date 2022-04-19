#dataset.py
import torch
import os
import h5py
import numpy as np
import pytorch_lightning as pl
from datasets import TrainDataset, InMemoryTrainDataset, TrainToTensor
#from inference import InferenceDataset, ToTensor
from torch.utils.data import Dataset, DataLoader


#DATAModule
#prepare_data (how to download(), tokenize, etc)
#setup (how to split, etc)
#train_dataloader
#val_dataloader(s)
#test_dataloader(s)

class RokoDataModule(pl.LightningDataModule):
    
    def __init__(self, 
                 data_dir: str, 
                 val_path: str,
                 batch_size: int = 32, 
                 mem: bool = False,
                 workers: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.mem = mem
        self.val_path = val_path
        self.workers = workers

    def prepare_data(self): # check valid paths
        assert(os.path.exists(self.data_dir))
        assert(os.path.exists(self.val_path))


    def setup(self, stage = None):
        if self.mem:
            self.data_class = InMemoryTrainDataset     
        else:
            self.data_class = TrainDataset

        self.train_ds = self.data_class(self.data_dir, transform=TrainToTensor())
        self.val_ds = self.data_class(self.val_path, transform=TrainToTensor())

    def train_dataloader(self):
        return DataLoader(self.train_ds, self.batch_size, True, num_workers=self.workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, self.batch_size, num_workers=self.workers, pin_memory=True)

    def test_dataloader(self):
        pass
        #return DataLoader(self.test_ds, self.batch_size, shuffle=False, num_workers= self.workers)
