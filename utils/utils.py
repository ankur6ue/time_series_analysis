import torch
import os
import models.model as model
import json
import numpy as np


class TSUtils:
    """
    Provides utility functions to calculate error metrics, split a batch between input, target and covariates and
    scaling the batch data
    """

    def __init__(self, dev, cfg):
        self.min = 0
        self.max = 0
        self.range = 0
        self.mean = 0
        self.dev = dev
        self.cfg = cfg

    def NRMSE(self, input, target):
        """
        Normalized Root Mean Square Error
        """
        return np.sqrt(torch.nn.functional.mse_loss(input, target).item()) / torch.mean(torch.abs(target))

    def NormDeviation(self, input, target):
        """
        Normalized Deviation
        """
        return torch.mean(torch.abs(input - target)) / torch.mean(torch.abs(target))

    def MAE(self, input, target):
        """
        Mean Absolute Error
        """
        return torch.mean(torch.abs(input - target))

    def split_batch(self, batch):
        """
        takes a batch and splits it into input, target and covariates. A batch has dimensions B x T x D, where D
        is structured as follows:
        | num_time_indx | num_target_series | num_covariates |

        - first num_time_indx cols: consist of relative and absolute time indices.
        Relative time of a sample w.r.t the start of the batch, absolute time of a sample w.r.t the start of the
        time sequence (age). In the current implementation, only the relative time is used, so num_time_indx = 1. If
        absolute age is also provided, num_time_indx = 2

        - next num_target_series cols: number of target series for which we want to perform forecasting

        - remaining cols: covariates, including seasonal (time) covariates (hour, min)

        The function returns the input, target (which is the input shifted by one time step) and covariates.
        covariates consist of the original covariates and time indices concatenated together.

        num_covariates indicates the number of covariates to concatenate. This must be less than the total number
        of covariates in the batch and allows flexibility on choosing the number of covariates we want to consider for
        forecasting.
        Similarly, total_num_targets tells us the total number of target time series, out of which we can pick num_target
        series for forecasting. num_targets must be <= total_num_targets
        """
        num_time_idx = self.cfg['num_time_idx']
        num_targets = self.cfg['num_targets']
        total_num_targets = self.cfg['total_num_targets']
        num_covariates = self.cfg['num_covariates']

        # first get the time indices
        b = num_time_idx
        time_idx = batch[:, 1::, :b].to(self.dev).float()
        # Next, the input and target time series. num_targets is the number of target series
        # we want to perform forecasting on
        e = b + num_targets
        input = batch[:, 0:-1, b: e].to(self.dev).float()
        # target is the input time shifted by one time step
        target = batch[:, 1::, b: e].to(self.dev).float()
        # advance e by total_num_targets because we may only have chosen a subset of the total number of targets
        # to perform forecasting. eg. electricity dataset has total_num_targets = 370, but we may perform forecasting
        # only on the first 100.
        e = e - num_targets + total_num_targets
        # TODO Add logic to check for number of covariates
        total_num_covariates = batch.shape[2] - e + 1
        # assert self.num_covariates <= total_num_covariates

        # lastly, the covariates. Covariates are a concatenation of the sample time indices and other covariates
        # (meteorological variables for the pollution dataset, seasonality variables such as hour, minutes etc)
        covariates = time_idx
        b = e
        e = b + num_covariates
        if (num_covariates > 0):  # if there are any covariates other than the time index (Sinewave example has none)
            covariates = torch.cat((time_idx, batch[:, 1::, b: e]), -1)

        return input, target, covariates

    def scale(self, input, covariates):
        """
        Scales the target time series and the covariates to be zero mean and range = 1.
        Scaling is not applied to the time index.
        :param batch
        :return: scaled batch
        """

        # [0] selects the values, [1] the args.
        # unsqueeze adds another dimension, so we can perform tensor ops with input and covariates
        self.mean = torch.mean(input, dim=1).unsqueeze(1)
        input = input - self.mean
        min = torch.min(input, dim=1)[0].unsqueeze(1)
        max = torch.max(input, dim=1)[0].unsqueeze(1)
        self.range = max - min
        self.range[self.range < 1e-5] = 1e-5
        input = input / self.range

        num_covariates = self.cfg['num_covariates']
        if (num_covariates > 0):
            mean = torch.mean(covariates, dim=1).unsqueeze(1)
            covariates = covariates - mean
            min = torch.min(covariates, dim=1)[0].unsqueeze(1)
            max = torch.max(covariates, dim=1)[0].unsqueeze(1)
            range = max - min
            range[range < 1e-5] = 1e-5
            covariates = covariates / range

        # For doing a sanity check for min/max values
        # min_ = torch.min(torch.min(batch[:, :, covariate_col:], dim=1)[0])
        # max_ = torch.max(torch.max(batch[:, :, covariate_col:], dim=1)[0])

        # scale time index.
        # min = torch.min(batch[:, :, :b], dim=1)[0].unsqueeze(1)
        # max = torch.max(batch[:, :, :b], dim=1)[0].unsqueeze(1)
        # range = max - min
        # batch[:, :, :b] = (batch[:, :, :b] - min) / range
        # mean = torch.mean(batch[:, :, :b], dim=1).unsqueeze(1)
        # batch[:, :, :b] = batch[:, :, :b] - mean_

        return input, covariates

    def invert_scale(self, input, probabalistic=False):
        """
        Inverts the scale applied to the target time series. Useful to calculate the loss functions on the un-scaled
        time series.
        :param input time series
        :return: unscaled time-series
        """
        if probabalistic == False:
            return input * self.range + self.mean
        else:
            mean = input[:, :, :, 0]
            std = input[:, :, :, 1]
            scaled_mean = mean * self.range + self.mean
            scaled_std = std * torch.sqrt(self.range)
            return torch.cat((scaled_mean.unsqueeze(-1), scaled_std.unsqueeze(-1)), -1)


def load_model(file_prefix):
    """
    loads model and associated config file.
    :param file_prefix: file name prefix (without extension) of the model and config file name.
    :return: The model and config files if they exist, None otherwise
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    predicate = os.path.join(dir_path, '..\\data', file_prefix)
    full_path_model = predicate + ".pth"
    full_path_cfg = predicate + ".txt"
    exists = os.path.isfile(full_path_model)
    if exists:
        with open(full_path_cfg) as f:
            cfg = f.read()
            # load config file
            cfg = cfg.replace("'", '"')
            cfg = json.loads(cfg)
            num_covariates = cfg['num_covariates']
            num_time_idx = cfg['num_time_idx']
            num_targets = cfg['num_targets']
            input_dim = num_time_idx + num_targets + num_covariates
            # create model object
            model_ = model.model(num_lstms=cfg['num_lstms'], input_dim=input_dim, output_dim=cfg['num_targets'],
                                 hidden_dim=cfg['hidden_dim'])
            f.close()
            # load weights and populate state dict for the model
            model_.load_state_dict(torch.load(full_path_model))
            return model_, cfg
    else:
        return None


# save model and associated config file so we know what parameters were used to generate a given model
def save_model(model, cfg, file_prefix):
    """
    save model and associated config file so we know what parameters were used to generate a given model
    :param model: model to be saved
    :param cfg: config
    :param file_prefix: file name prefix (without extension) of the model and config file name. Model is saved as a .pth file
    and config as a txt file
    :return: None
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    predicate = os.path.join(dir_path, '..\\data', file_prefix)
    torch.save(model.state_dict(), predicate + ".pth")
    f = open(predicate + ".txt", "w")
    f.write(str(cfg))
    f.close()
