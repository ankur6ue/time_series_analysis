# code taken from: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# References: DeepAR (https://arxiv.org/pdf/1704.04110.pdf)
from pandas import read_csv
import os
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from mydataset import MyDataset
import torch
import numpy as np


class NSDQ100Dataset(MyDataset):
    """NSDQ100 dataset."""

    def __init__(self, csv_file, dev, ctx_win_len, col_names, num_time_indx=1):
        """
        Loads the NSDQ100 dataset so samples can be drawn from it during training and testing
        :param csv_file: file name of the csv file containing the data
        :param dev: device (CPU/GPU). Output of torch.device
        :param ctx_win_len: length of the context window (conditioning + prediction window)
        :param col_names: stock ticker symbols of the stocks we want to forecast. eg: ['NDX', 'ALXN']
        :param num_time_indx: number of time indices. Usually a relative time index wrt to the start of a batch,
        but can also include absolute time index from the beginning of the series (age). If the age is included,
        num_time_indx = 2, otherwise 1
        """
        self.num_tim_indx = num_time_indx
        self.dev = dev

        path = os.path.dirname(os.path.realpath(__file__))
        df = read_csv(path + '\\..\\data\\NSDQ100\\' + csv_file, header=0, index_col=0)
        # make the target columns the first columns in the dataframe
        cols = df.columns.tolist()
        for i in range(len(col_names)):
            try:
                idx = cols.index(col_names[i])  # index of strings in col_names, throws exception if element not found
            except ValueError:
                idx = None
            if (idx is not None):  # rearrange columns so col corresponding to idx is in the beginning
                cols = cols[idx:idx+1] + cols[:idx] + cols[idx+1:]
        df = df[cols]


        ## plot correlation matrix
        self.plot_corr(df[cols[0:70]])

        self.ctx_win_len = ctx_win_len
        values = df.values

        # plot some data
        groups = [0, 1, 2, 3, 4, 5]
        i = 1
        # plot each column
        fig = pyplot.figure()
        for group in groups:
            pyplot.subplot(len(groups), 1, i)
            pyplot.plot(values[20000:-1, group])
            pyplot.title(df.columns[group], y=0.5, loc='right')
            i += 1
        # for adjusting vertical spacing between subplots (hspace is for vertical spacing, go figure)
        fig.subplots_adjust(hspace=0.5)
        pyplot.show()

        # ensure all data is float
        values = values.astype('float32')
        self.scaled = values
        # normalize features
        #self.min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        #self.std_scaler_target = StandardScaler()
        #self.scaled[:, 0:1] = self.std_scaler_target.fit_transform(values[:, 0:1])

        # scale all covariates to be 0 mean, 1 variance (see section 3.4 of deepAR paper)
        #self.std_scalar = StandardScaler()
        #self.scaled[:, 1:] = self.std_scalar.fit_transform(self.scaled[:, 1:])

        self.dataset_size = self.scaled.shape[0]
        self.scaled = torch.from_numpy(self.scaled).float().to(dev)
        # time series resolution
        self.resolution = 1