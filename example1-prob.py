# This script implements training a probabilistic model that outputs parameters of a gaussian likelihood function
import models.model_prob as model
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data.dataloader import DataLoader
from utils.utils import TSUtils
from utils.utils import save_model

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# set random seeds for reproducibility
torch.manual_seed(0)  # for pytorch
np.random.seed(seed=0)  # for numpy

# Pick the model you want to train
example = 'NSDQ100'

if example == 'Electricity':
    from dataset.electricity_dataset import ElectricityDataset
    from data.Electricity import config

    cfg = config.simple_cfg
    dataset = ElectricityDataset(csv_file=cfg['data_file'], dev=device, ctx_win_len=cfg['ctx_win_len'], num_time_indx=cfg['num_time_idx'])

if example == 'Sinewave':
    from dataset.SW_dataset import SWDataset
    from data.Sinewave import config
    cfg = config.simple_cfg
    dataset = SWDataset(dev=device, num_time_indx=cfg['num_time_idx'], ctx_win_len=cfg['ctx_win_len'])

if example == 'BeijingPM25':
    from dataset.pollution_dataset import PollutionDataset
    from data.BeijingPM25 import config

    cfg = config.full_cfg
    dataset = PollutionDataset(csv_file=cfg['data_file'], dev=device, ctx_win_len=cfg['ctx_win_len'])

if example == 'SML2010':
    from dataset.SML2010_dataset import SML2010Dataset
    from data.SML2010 import config

    cfg = config.full_cfg
    dataset = SML2010Dataset(csv_file=cfg['data_file'], dev=device, ctx_win_len=cfg['ctx_win_len'])

if example == 'NSDQ100':
    from dataset.NSDQ100_dataset import NSDQ100Dataset
    from data.NSDQ100 import config

    cfg = config.full_cfg
    dataset = NSDQ100Dataset(csv_file=cfg['data_file'], dev=device, ctx_win_len=cfg['ctx_win_len'],
                             col_names=['NDX', 'ALXN'])

# Read variables from the corresponding config
num_covariates = cfg['num_covariates']
num_time_idx = cfg['num_time_idx']
num_targets = cfg['num_targets']
input_dim = num_time_idx + num_targets + num_covariates
# Total context window size used for training. The context window consists of 1. conditioning_context where input
# data points are available and network predictions are conditioned on actual input data, and 2. prediction_context,
# where the network predictions are conditioned on network output at the previous step. Covariates are assumed
# to be available for the entire context window
ctx_win_len = cfg['ctx_win_len']
cond_win_len = cfg['cond_win_len']
pred_win_len = ctx_win_len - cond_win_len - 1
batch_size = cfg['batch_size']

model = model.model(num_lstms=cfg['num_lstms'], input_dim=input_dim, output_dim=cfg['num_targets'],
                       hidden_dim=cfg['hidden_dim']).to(device)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), cfg['lr'])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['lr_step_size'], gamma=cfg['lr_gamma'])

# get_train_test_samplers creates Pytorch RandomSamplers for training and testing. The samplers are used to
# generate batches.
train_sampler, test_sampler = dataset.get_train_test_samplers(cfg['train_test_split'])

train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                              shuffle=False, num_workers=0)

test_dataloader = DataLoader(dataset, batch_size=1, sampler=test_sampler,
                             shuffle=False, num_workers=0)
max_batches_per_epoch = cfg['max_batches_per_epoch']
num_epochs = cfg['num_epochs']

losses = []
batch_num = 1
utils = TSUtils(device, cfg)
# Training function
def train():
    for epoch in range(0, num_epochs):
        batch_num = 0
        for i, batch in enumerate(train_dataloader):
            if batch_num > max_batches_per_epoch:
                break
            # split batch into input, target and covariates. Target is simply the input shifted forward by one time step
            input, target, covariates = utils.split_batch(batch)
            # input in the conditioning window. For the prediction window, model output will be conditioned on the
            # model output for the previous step
            input_cond = input[:, 0:cond_win_len, :]
            # Scale the input and oovariates so they are 0 mean and range = 1. Note we apply scaling per batch, not for the
            # entire dataset. Contrast this with training imagenet models, where the scaling params are calculated over
            # the entire training dataset
            input_cond, covariates = utils.scale(input_cond, covariates)
            optimizer.zero_grad()
            # Run the forward pass
            out = model(input_cond, covariates, future=pred_win_len)
            # scale model output and target to the original data scale. Passing probabalistic=True tells the
            # invert_scale function that the likelihood model is being used and the likelihood parameters (mean/var)
            # should be rescaled.
            out = utils.invert_scale(out, probabalistic=True)
            # calculate negative log likelihood loss, run backward pass, update parameters.
            loss = model.NLL(out, target)
            loss.backward()
            scheduler.step()
            optimizer.step()
            loss = loss.item()
            losses.append(loss)
            print('epoch:{0}/{1}, step:{2}, loss:{3}'.format(epoch, num_epochs, batch_num, loss))
            batch_num = batch_num + 1

    return model, losses

## helper function to plot the loss function progression
def plot_loss(losses):
    plt.figure()
    plt.title('Loss progression')
    plt.xlabel('batch no')
    plt.ylabel('loss')
    plt.plot(np.arange(len(losses)), losses, 'g', linewidth=2.0)
    plt.show()

def save_model_(model, file_prefix):
    save_model(model, cfg, file_prefix)

## testing
def predict(model, num_targets=1):
    test_dataloader_iter = iter(test_dataloader)
    # Generate 1 samples with batch size = 1
    for i in range(0, 1):
        test_batch = next(test_dataloader_iter)
        with torch.no_grad():
            input, target, covariates = utils.split_batch(test_batch)
            input, covariates = utils.scale(input, covariates)
            pred = model(input[:, 0:cond_win_len, :], covariates, future=pred_win_len)
            pred = utils.invert_scale(pred, True)
            loss = model.NLL(pred, target)
            print("loss (prediction): {0}".format(loss))
            # convert prediction and target to numpy array on the CPU for plotting
            # B x T x N (B = batch size (1 for prediction), N = number of target series, T= number of time steps in the
            # context window
            preds = pred[0, :, :, :].detach().cpu().numpy()
            targets = target[0, :, :].detach().cpu().numpy()
        # plot the target series on a separate plot
        for j in range(num_targets):
            pred = preds[:, j, 0]  # mean of predicted series j
            std = preds[:, j, 1]  # std. dev of predicted series j
            target = targets[:, j]
            plt.figure()
            plt.title(
                'Red: Groundtruth, Green: Predicted values \n Dashlines: predicted values in the prediction window')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xticks()
            plt.yticks()
            # plot target series in red
            plt.plot(np.arange(len(target)), target, 'r', linewidth=2.0)
            # plot the prediction in the conditioning window in solid green
            plt.plot(np.arange(cond_win_len), pred[0:cond_win_len], 'g', linewidth=2.0)
            # plot the prediction in the conditioning window in dashed green
            plt.plot(np.arange(cond_win_len, cond_win_len + pred_win_len), pred[cond_win_len:], 'g' + ':',
                     linewidth=2.0)
            # plot the +- sigma and +- 2 sigma lines in dark gray and light gray
            plt.fill_between(np.arange(len(target)), pred - std, pred + std, color='gray', alpha=0.5)
            plt.fill_between(np.arange(len(target)), pred - 2 * std, pred + 2 * std, color='gray', alpha=0.2)
            #plt.show()

if __name__ == "__main__":
    model, losses = train()
    file_prefix = "{0}\\model_covariates-{1}_epochs{2}".format(example, cfg['num_covariates'],
                                                                       cfg['num_epochs'])
    #save_model_(model, file_prefix)
    predict(model, num_targets)

