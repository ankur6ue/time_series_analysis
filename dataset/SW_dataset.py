# References: DeepAR (https://arxiv.org/pdf/1704.04110.pdf)
from pandas import read_csv
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from mydataset import MyDataset
import torch
import numpy as np
from matplotlib import pyplot as plt

class SWDataset(MyDataset):
    """Loads the modulaed sinewave dataset. See _time_series function below for the functional form"""

    def __init__(self, dev, ctx_win_len, num_time_indx=1, t_min=0, t_max=40, resolution=0.05):
        """
        Samples a modulated sine function and stores the samples in the dataset array. Batches are drawn out
        of this array during training.
        :param dev: device (CPU/GPU). Output of torch.device
        :param ctx_win_len: length of the context window (conditioning + prediction window)
        :param num_time_indx: number of time indices. Usually a relative time index, but can also include absolute time
        index from the beginning of the series (age). If the age is included, num_time_indx = 2, otherwise 1
        :param t_min: start time index of the modulated SW
        :param t_max: end time index of the modulated SW
        :param resolution: resolution of the modulated SW
        """

        self.num_time_indx = num_time_indx
        self.dev = dev

        self.t_min = t_min
        self.t_max = t_max
        self.resolution = resolution
        self.t_min = self.t_min/self.resolution
        self.t_max = self.t_max/self.resolution
        self.dataset = self._time_series(np.arange(self.t_min,self.t_max)*self.resolution)
        plt.figure()
        plt.title('Time series function')
        plt.xlabel('time')
        plt.ylabel('function')
        plt.plot(np.arange(len(self.dataset)), self.dataset, 'g', linewidth=2.0)
        plt.show()
        # no scaling is applied to the original samples. Instead, scaling is applied to each batch before feeding the
        # batch to the model
        self.scaled = torch.from_numpy(self.dataset).float().to(dev)
        self.dataset_size = self.scaled.shape[0]
        self.scaled = self.scaled.unsqueeze(1)
        self.ctx_win_len = ctx_win_len
        self.resolution = 1

    @staticmethod
    def _time_series(t):
        return t * np.sin(t) / 12 + np.sin(t * 5)
