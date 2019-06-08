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
        return np.sqrt(torch.nn.functional.mse_loss(input, target).item())/torch.mean(torch.abs(target))

    def NormDeviation(self, input, target):
        """
        Normalized Deviation
        """
        return torch.mean(torch.abs(input - target))/torch.mean(torch.abs(target))

    def MAE(self, input, target):
        """
        Mean Absolute Error
        """
        return torch.mean(torch.abs(input - target))

    def split_batch(self, batch):
        """
        takes a batch and splits it into input, target and covariates. A batch is structured as follows:
        | num_time_indx | num_target_series | num_covariates |
        - first num_time_indx cols: time indices - relative time of a sample wrt the start of the batch, absolute
        time of a sample wrt the start of the time sequence (age)
        - next num_target_series cols: number of target series over which we want to perform forecasting
        - remaining cols: number of covariates

        The function returns the input, target (which is the input shifted by one time step) and covariates.
        covariates consist of the original covariates and time indices concatenated together.

        self.num_covariates indicates the number of covariates to concatenate. This must be less than the total number
        of covariates in the batch and allows flexibility on number of covariates we want to consider for forecasting.
        """
        num_time_idx        = self.cfg['num_time_idx']
        num_targets         = self.cfg['num_targets']
        total_num_targets   = self.cfg['total_num_targets']
        num_covariates      = self.cfg['num_covariates']

        # first get the time indices
        b = num_time_idx
        time_idx = batch[:, 1::, :b].to(self.dev).float()
        e = b + num_targets
        # Next, the target time series. self.num_target_series is the number of target series
        # we want to perform forecasting on
        input = batch[:, 0:-1, b: e].to(self.dev).float()
        # target is the input time shifted by one time step
        target = batch[:, 1::, b: e].to(self.dev).float()

        e = e - num_targets + total_num_targets
        # TODO Add logic to check for number of covariates
        total_num_covariates = batch.shape[2] - e + 1
        # assert total_num_covariates <= self.num_covariates

        # lastly, the covariates
        covariates = time_idx
        b = e
        e = b + num_covariates
        if (num_covariates > 0):
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

    def scale_(self, batch):
        """
        Scales the target time series and the covariates to be zero mean and range = 1.
        Scaling is not applied to the time index.
        :param batch
        :return: scaled batch
        """
        # scale target time series
        b = self.num_time_indx  # begin index
        e = self.num_time_indx + self.num_target_series  # end index

        # [0] selects the values, [1] the args.
        # unsqueeze adds another dimension, so we can perform tensor ops with input and covariates
        self.min = torch.min(batch[:, :, b:e], dim=1)[0].unsqueeze(1)
        self.max = torch.max(batch[:, :, b:e], dim=1)[0].unsqueeze(1)
        self.range = self.max - self.min
        self.range[self.range < 1e-5] = 1e-5  # divide by zero issues
        batch[:, :, b:e] = (batch[:, :, b:e] - self.min) / self.range
        # verify: torch.min(input, dim=1) is all zeros etc
        self.mean = torch.mean(batch[:, :, b:e], dim=1).unsqueeze(1)
        batch[:, :, b:e] = batch[:, :, b:e] - self.mean

        if (self.num_covariates > 0):
            # column index from where covariates start.
            covariate_col = self.num_time_indx + self.num_target_series
            min = torch.min(batch[:, :, covariate_col:], dim=1)[0].unsqueeze(1)
            max = torch.max(batch[:, :, covariate_col:], dim=1)[0].unsqueeze(1)
            range = max - min
            range[range < 1e-5] = 1e-5
            batch[:, :, covariate_col:] = (batch[:, :, covariate_col:] - min) / range
            mean = torch.mean(batch[:, :, covariate_col:], dim=1).unsqueeze(1)
            batch[:, :, covariate_col:] = batch[:, :, covariate_col:] - mean

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

        return batch

    def invert_scale_(self, input):
        """
        Inverts the scale applied to the target time series. Useful to calculate the loss functions on the un-scaled
        time series.
        :param input time series
        :return: time series with the scale applied in scale (see above) undone.
        """
        return (input + self.mean) * self.range + self.min

    def invert_scale(self, input):
        """
        Inverts the scale applied to the target time series. Useful to calculate the loss functions on the un-scaled
        time series.
        :param input time series
        :return: time series with the scale applied in scale (see above) undone.
        """
        return input * self.range + self.mean

def load_model(file_prefix):
    """
    loads model and associated config file.
    :param file_prefix: file name prefix (without extension) of the model and config file name.
    :return: The model and config files if they exist, None otherwise
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    full_path_model = dir_path + '\\..\\data\\' + file_prefix + ".pth"
    full_path_cfg = dir_path + '\\..\\data\\' + file_prefix + ".txt"
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
    torch.save(model.state_dict(), dir_path + '\\..\\data\\' + file_prefix + ".pth")
    f = open(dir_path + '\\..\\data\\' + file_prefix + ".txt", "w")
    f.write(str(cfg))
    f.close()