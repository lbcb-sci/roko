import torch
from torch.utils.data import Dataset
import os
import h5py
import numpy as np
from abc import abstractmethod


def get_filenames(path):
    if os.path.isdir(path):
        filenames = []
        for f in os.listdir(path):
            if f.endswith('.hdf5'):
                filenames.append(os.path.join(path, f))
        for name in filenames:
            print(name)
        return filenames

    return [path]

class StorageDataset(Dataset):
    def __init__(self, path, transform=None, read_contigs=False):
        self.filenames = get_filenames(path)
        fds = [h5py.File(f, 'r', libver='latest', swmr=True) for f in self.filenames]

        # For torch workers
        self.fds = None

        self.idx = {}
        self.contigs = {}
        self.size = 0

        for i, f in enumerate(fds):
            groups = list(f.keys())

            if 'info' in groups:
                groups.remove('info')
            if 'contigs' in groups:
                groups.remove('contigs')

            for g in groups:
                group_size = f[g].attrs['size']
                for j in range(group_size):
                    self.idx[self.size + j] = (i, g, j)
                self.size += group_size

        for f in fds:
            f.close()

        self.transform = transform

    @abstractmethod
    def get_sample(self, group, offset):
        pass

    def __getitem__(self, idx):
        f_idx, g, p = self.idx[idx]

        # Open fds for the first time
        if not self.fds:
            self.fds = [h5py.File(f, 'r', swmr=True, libver='latest') for f in self.filenames]

        f = self.fds[f_idx]
        group = f[g]

        sample = self.get_sample(group, p)

        # permutate the order of the reads
        perm = np.random.permutation(sample[0].shape[0])
        perm = np.delete(perm, np.where(perm == 0))
        perm = np.insert(perm, 0, 0)
        tup = (sample[0][perm], sample[1], sample[2])

        if self.transform:
            tup = self.transform(tup)

        return tup

    def __len__(self):
        return self.size

class TrainDataset(StorageDataset):
    def get_sample(self, group, offset):
        X = group['examples'][offset]
        Y = group['labels'][offset]
        X2 = group['stats'][offset] # 5 90
        
        # del the print statements later: check if before and after are the same data types
        print("before",type(X2[0][0]))
        X2 = X2.astype(np.int16)
        print("after",type(X2[0][0]))

        return X, Y, np.nan_to_num(X2/np.sum(X2,axis=0))


class InMemoryTrainDataset(Dataset):
    def __init__(self, path, transform=None):
        self.filenames = get_filenames(path)

        self.X = []
        self.Y = []
        self.X2 = []

        for filename in self.filenames:
            with h5py.File(filename, 'r') as f:
                groups = list(f.keys())
                if 'info' in groups:
                    groups.remove('info')
                if 'contigs' in groups:
                    groups.remove('contigs')

                for g in groups:
                    X = f[g]['examples'][()]
                    Y = f[g]['labels'][()]
                    X2 = f[g]['stats'][()]

                    self.X.extend(list(X))
                    self.Y.extend(list(Y))
                    self.X2.extend(list(X2))

            print(f'Processed: {filename}')

        assert len(self.X) == len(self.Y) == len(self.X2)
        self.size = len(self.X)

        self.transform = transform

    def __getitem__(self, idx):
        sample = (self.X[idx], self.Y[idx], self.X2[idx])
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.size


class TrainToTensor:
    def __call__(self, sample):
        X, Y, X2 = sample
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(X2)
