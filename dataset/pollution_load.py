# code taken from: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# References: DeepAR (https://arxiv.org/pdf/1704.04110.pdf)
from pandas import read_csv
from datetime import datetime
import os
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler
import torch
import numpy as np


class PollutionDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, dev, ctx_win_len, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        path = os.path.dirname(os.path.realpath(__file__))
        dataset = read_csv(path + csv_file, header=0, index_col=0)

        self.ctx_win_len = ctx_win_len
        values = dataset.values

        # integer encode direction
        self.dir_encoder = LabelEncoder()
        values[:, 4] = self.dir_encoder.fit_transform(values[:, 4])

        # ensure all data is float
        values = values.astype('float32')
        self.scaled = values
        # normalize features
        self.min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled[:, 0:1] = self.min_max_scaler.fit_transform(values[:, 0:1])

        # scale all covariates to be 0 mean, 1 variance (see section 3.4 of deepAR paper)
        self.std_scalar = StandardScaler()
        self.scaled[:, 1:] = self.std_scalar.fit_transform(self.scaled[:, 1:])
        # total data: 43800 = 1825*24
        # training: 0: test_offset
        # test: test_offset: total data len
        self.train_size = 2*365*24
        self.total_data_size = self.scaled.shape[0]
        self.scaled = torch.from_numpy(self.scaled).float().to(dev)

    def get_scale_params(self):
        return self.min_max_scaler.data_min_[0], self.min_max_scaler.data_max_[0]

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        r = torch.from_numpy(np.arange(idx, idx + self.ctx_win_len)).unsqueeze(1).float().cuda()
        out = torch.cat((r, self.scaled[idx:idx + self.ctx_win_len, :]), -1).unsqueeze(0)
        return out.squeeze()


class RandomSampler(Sampler):
    r"""Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source_len, offset, ctx_win_len):
        self.data_source_len = data_source_len
        self.ctx_win_len = ctx_win_len
        # returns random numbers between 0, n.
        self.perm = torch.randperm(self.data_source_len - self.ctx_win_len - offset)
        self.perm = self.perm + offset
        self.idx = 0

    def __iter__(self):
        while self.idx + self.ctx_win_len < self.data_source_len:
            sample_idx = self.perm[self.idx]
            yield sample_idx
            self.idx = self.idx + 1

    def __len__(self):
        return len(self.data_source)
