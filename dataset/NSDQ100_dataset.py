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
    """NSDQ100 dataset (http://cseweb.ucsd.edu/~yaq007/NASDAQ100_stock_data.html)."""

    def __init__(self, csv_file, dev, ctx_win_len, col_names, num_time_indx=1):
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
        df = read_csv(os.path.join(path, '..\\data\\NSDQ100', csv_file), header=0, index_col=0)
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
        self.dataset_size = self.scaled.shape[0]
        self.scaled = torch.from_numpy(self.scaled).float().to(dev)
        # time series resolution
        self.resolution = 1