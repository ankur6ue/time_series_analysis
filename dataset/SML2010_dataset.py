# References: DeepAR (https://arxiv.org/pdf/1704.04110.pdf)
from pandas import read_csv
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from mydataset import MyDataset
import torch
import numpy as np


class SML2010Dataset(MyDataset):
    """SML2010 dataset (https://archive.ics.uci.edu/ml/datasets/SML2010)"""

    def __init__(self, csv_file, dev, ctx_win_len, num_time_indx=1):
        """
        Reads the dataset file, performs some data visualization (plotting cross-correlation heatmaps) and transfers
        loaded data to the device (CPU/CUDA)
        :param csv_file: csv filename containing preprocessed data.
        :param dev: cpu or cuda, output of torch.device
        :param ctx_win_len: context window length (conditioning + prediction window)
        used by parent MyDataset class to read chunks out of the loaded data during batch generation
        :param num_time_indx: 1 if only relative age is used, 2 if both absolute age and relative age are used
        """
        self.num_time_indx = num_time_indx
        self.dev = dev
        path = os.path.dirname(os.path.realpath(__file__))
        df = read_csv(os.path.join(path, '..\\data\\SML2010', csv_file), header=0, index_col=0)

        self.ctx_win_len = ctx_win_len
        values = df.values

        # ensure all data is float
        values = values.astype('float32')
        self.scaled = values
        ## plot correlation matrix
        self.plot_corr(df)

        # note: code below is commented out, because we do per batch normalization rather than normalizing the entire
        # training data
        # normalize features
        # self.min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        # self.scaled[:, 0:1] = self.min_max_scaler.fit_transform(values[:, 0:1])

        # scale all covariates to be 0 mean, 1 variance (see section 3.4 of deepAR paper)
        # self.std_scalar = StandardScaler()
        # self.scaled[:, 1:] = self.std_scalar.fit_transform(self.scaled[:, 1:])

        self.dataset_size = self.scaled.shape[0]
        self.scaled = torch.from_numpy(self.scaled).float().to(dev)
        # time series resolution
        self.resolution = 1