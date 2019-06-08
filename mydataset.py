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
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
import torch
import numpy as np
import matplotlib.pyplot as plt


class MyDataset(Dataset):
    def get_min_max_scale_params(self):
        return self.min_max_scaler.data_min_[0], 1 / self.min_max_scaler.scale_[0]

    def get_std_scale_params(self):
        return self.std_scaler_target.mean_[0], self.std_scaler_target.scale_[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # relative time from the start of this context window
        rel_time = torch.from_numpy(np.arange(0, self.ctx_win_len) * self.resolution).unsqueeze(1).float().to(self.dev)
        # absolute time from the start of the time series
        abs_time = torch.from_numpy(np.arange(idx, idx + self.ctx_win_len) * self.resolution).unsqueeze(1).float().to(
            self.dev)
        if (self.num_tim_indx == 2):  # use both relative time and absolute time (from the beginning of the series)
            out = torch.cat((rel_time, abs_time, self.scaled[idx: idx + self.ctx_win_len, :]), -1).unsqueeze(0)
        if (self.num_tim_indx == 1):  # relative time only
            out = torch.cat((rel_time, self.scaled[idx: idx + self.ctx_win_len, :]), -1).unsqueeze(0)
        if (self.num_tim_indx == 0):  # no time, only used for the SW example
            out = self.scaled[idx: idx + self.ctx_win_len, :].unsqueeze(0)
        return torch.squeeze(out, 0)

    def get_train_test_samplers(self, train_test_split):
        # Creating data indices for training and test splits:

        indices = list(range(self.dataset_size))
        split = int(np.floor(train_test_split * self.dataset_size))
        # subtract ctx_win_len from the indices because each training sample consists of ctx_win_len data points
        # after the index
        train_indices, test_indices = indices[: split - self.ctx_win_len], indices[split: -self.ctx_win_len]
        return SubsetRandomSampler(train_indices), SubsetRandomSampler(test_indices)

    def plot_corr(self, df, size=10):
        """Function plots a graphical correlation matrix for each pair of columns in the dataframe.
        Input:
            df: pandas DataFrame
            size: vertical and horizontal size of the plot"""
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(size, size))
        ax.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
        plt.yticks(range(len(corr.columns)), corr.columns)
