# References: DeepAR (https://arxiv.org/pdf/1704.04110.pdf)
from pandas import read_csv, DataFrame
import os
from mydataset import MyDataset
import torch

class ElectricityDataset(MyDataset):
    """Electricity dataset."""

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
        df = read_csv(os.path.join(path, '..\\data\\Electricity', csv_file), header=0, index_col=0)

        self.ctx_win_len = ctx_win_len
        values = df.values
        # ensure all data is float
        values = values.astype('float32')

        ## plot correlation matrix
        # df2 = DataFrame(values, index=df.index, columns=df.columns)
        # self.plot_corr(df2)
        self.scaled = values
        self.dataset_size = self.scaled.shape[0]
        self.scaled = torch.from_numpy(self.scaled).float().to(dev)
        # time series resolution
        self.resolution = 1