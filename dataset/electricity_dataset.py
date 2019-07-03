# code taken from: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# References: DeepAR (https://arxiv.org/pdf/1704.04110.pdf)
from pandas import read_csv, DataFrame
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from mydataset import MyDataset
import torch
import numpy as np


class ElectricityDataset(MyDataset):
    """Beijing pollution dataset."""

    def __init__(self, csv_file, dev, ctx_win_len, num_time_indx=1):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.num_tim_indx = num_time_indx
        self.dev = dev
        path = os.path.dirname(os.path.realpath(__file__))
        df = read_csv(os.path.join(path, '..\\data\\Electricity', csv_file), header=0, index_col=0)

        self.ctx_win_len = ctx_win_len
        values = df.values
        # first column is the date, second column: week_day, third column: hours, fourth column: minutes
        # ensure all data is float
        values = values.astype('float32')

        ## plot correlation matrix
        # df2 = DataFrame(values, index=df.index, columns=df.columns)
        # self.plot_corr(df2)

        self.scaled = values
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