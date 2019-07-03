# References: DeepAR (https://arxiv.org/pdf/1704.04110.pdf)
from pandas import read_csv, DataFrame
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from mydataset import MyDataset
import torch
import numpy as np


class PollutionDataset(MyDataset):
    """Beijing pollution dataset."""

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
        df = read_csv(os.path.join(path, '..\\data\\BeijingPM25', csv_file), header=0, index_col=0)

        self.ctx_win_len = ctx_win_len
        values = df.values

        # integer encode direction
        self.dir_encoder = LabelEncoder()
        values[:, 6] = self.dir_encoder.fit_transform(values[:, 6])

        # first column is the pollution values, second column: week_day, third column: hours
        # convert week_day and hours into sin/cos series.
        # See: https://github.com/drivendataorg/power-laws-forecasting/blob/master/3rd%20Place/Model_Documentation_and_Write_up.pdf
        hours = values[:, 2]
        week_day = values[:, 1]

        time_sin = np.sin(2 * np.pi * hours.astype(float) / 24)
        time_cos = np.cos(2 * np.pi * hours.astype(float) / 24)

        wday_sin = np.sin(2 * np.pi * week_day.astype(float) / 7)
        wday_cos = np.cos(2 * np.pi * week_day.astype(float) / 7)

        # Replace existing columns with sin transform, add new column for cosine
        #values[:, 1] = wday_sin
        #values[:, 2] = time_sin
        #values = np.insert(values, 1, wday_cos, axis=1)
        #values = np.insert(values, 3, time_cos, axis=1)

        # ensure all data is float
        values = values.astype('float32')

        ## plot correlation matrix
        #df2 = DataFrame(values, index=df.index, columns=df.columns)
        #self.plot_corr(df2)

        self.scaled = values
        self.dataset_size = self.scaled.shape[0]
        self.scaled = torch.from_numpy(self.scaled).float().to(dev)
        # time series resolution
        self.resolution = 1