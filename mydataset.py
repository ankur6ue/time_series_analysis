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
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # relative time from the start of this context window
        rel_time = torch.from_numpy(np.arange(0, self.ctx_win_len) * self.resolution).unsqueeze(1).float().to(self.dev)
        # absolute time from the start of the time series
        abs_time = torch.from_numpy(np.arange(idx, idx + self.ctx_win_len) * self.resolution).unsqueeze(1).float().to(
            self.dev)
        if (self.num_time_indx == 2):  # use both relative time and absolute time (from the beginning of the series)
            out = torch.cat((rel_time, abs_time, self.scaled[idx: idx + self.ctx_win_len, :]), -1).unsqueeze(0)
        if (self.num_time_indx == 1):  # relative time only
            out = torch.cat((rel_time, self.scaled[idx: idx + self.ctx_win_len, :]), -1).unsqueeze(0)
        if (self.num_time_indx == 0):  # no time, only used for the SW example
            out = self.scaled[idx: idx + self.ctx_win_len, :].unsqueeze(0)
        return torch.squeeze(out, 0)

    def get_train_test_samplers(self, train_test_split):
        """
        Create a list of inidces in the time series data and split the list into a training and test partition based on
        the train_test_split ratio. Training and testing RandomSamplers are then created from these partitions which can
        be used to sample batches during training and testing using the Sampler infrastructure provided by PyTorch

        :param train_test_split: ratio of training samples and total number of samples. A value of 0.7 means 70% of the
        time series samples will be used for training and the rest for testing
        :return: Train and Test random samplers.
        """

        indices = list(range(self.dataset_size))
        split = int(np.floor(train_test_split * self.dataset_size))
        # subtract ctx_win_len from the indices because each training sample consists of ctx_win_len data points
        # after the index
        train_indices, test_indices = indices[: split - self.ctx_win_len], indices[split: -self.ctx_win_len]
        return SubsetRandomSampler(train_indices), SubsetRandomSampler(test_indices)

    def plot_corr(self, df, size=10):
        """
        Plots a graphical correlation matrix for each pair of columns in the dataframe.
        Input:
            df: pandas DataFrame
            size: vertical and horizontal size of the plot"""
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(size, size))
        ax.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
        plt.yticks(range(len(corr.columns)), corr.columns)
