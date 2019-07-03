import torch
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data.dataloader import DataLoader
from utils.utils import TSUtils, load_model

""" 
This file loads models corresponding to three configs varying in number of covariates and epochs for the 
pollution dataset and runs inference on each of those models on the same data so the model results can be 
compared
"""

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# set random seeds for reproducibility
torch.manual_seed(0)  # for pytorch
np.random.seed(seed=0)  # for numpy

example = 'BeijingPM25'

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
                             col_names=['NDX'])

# The config parameters for the 3 models [covariates, num_epochs] for the pollution dataset
cov_epochs = [[2, 50], [9, 50], [9, 100]]
# list of models and configs that we'll be loading
models = []
cfgs = []
# create three separate models/utils corresponding to the three configs
for cov, epoch in cov_epochs:
    # file prefix for each config file
    file_prefix = "{0}\\model_covariates-{1}_epochs{2}".format(example, cov, epoch)
    # Load the pre-trained model and config corresponding to the file prefix
    model, cfg = load_model(file_prefix)
    model.eval()
    model.to(device)
    models.append(model)
    cfgs.append(cfg)

# The configs only differ in the number of covariates and epochs. Other parameters are the same
# use the first config to create the dataloaders
cfg = cfgs[0]
num_covariates = cfg['num_covariates']
num_time_idx = cfg['num_time_idx']
num_targets = cfg['num_targets']
input_dim = num_time_idx + num_targets + num_covariates
ctx_win_len = cfg['ctx_win_len']
cond_win_len = cfg['cond_win_len']
pred_win_len = ctx_win_len - cond_win_len - 1
#pred_win_len = 19
batch_size = cfg['batch_size']
utils = TSUtils(device, cfg)
train_sampler, test_sampler = dataset.get_train_test_samplers(cfg['train_test_split'])
test_dataloader = DataLoader(dataset, batch_size=1, sampler=test_sampler,
                             shuffle=False, num_workers=0)
criterion = torch.nn.MSELoss()
test_dataloader_iter = iter(test_dataloader)
colors = ['g', 'b', 'y']
# errors for different sample windows for each of the 3 models
rmse = [] # Root Mean Square
mae = [] # Mean Average Error
nd = [] # Normalized Deviation
nrmse = [] # Normalized RMSE
legends = ['target series']
for i in range(0,len(cov_epochs)):
    rmse.append([])
    mae.append([])
    nd.append([])
    nrmse.append([])
    legends.append('num covariates: {0}, num epochs: {1}'.format(cov_epochs[i][0], cov_epochs[i][1]))

handles = []
num_trials = 10
for i in range(0, num_trials):
    test_batch = next(test_dataloader_iter)
    # A batch is dimensions B x T x D, B = 1 for testing, T: context window length, and D = num_time_indices (generally
    # 1 for the relative age) + num of target time series + num of covariates
    # We are only interested in the target for now, so ignore the rest
    _, target, _ = utils.split_batch(test_batch)
    targets = target[0, :, :].detach().cpu().numpy()
    # plot the target series. 0 selects the first target series in this batch. The Pollution dataset only has one
    # target series (the PM2.5 index), the other datasets may have more than one.
    target = targets[:, 0]
    plt.figure()
    plt.title('Solid line: conditioning window. Dash line: prediction window')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xticks()
    plt.yticks()
    # plot the target series first so we don't plot it repeatedly for multiple models
    handle, = plt.plot(np.arange(len(target)), target, 'r', linewidth=2.0)
    handles.append(handle)
    # Run multiple models on the same data
    for k in np.arange(0, len(models)):
        cfg = cfgs[k]
        model = models[k]
        num_covariates = cfg['num_covariates']
        num_time_idx = cfg['num_time_idx']
        num_targets = cfg['num_targets']
        input_dim = num_time_idx + num_targets + num_covariates
        ctx_win_len = cfg['ctx_win_len']
        batch_size = cfg['batch_size']
        utils = TSUtils(device, cfg)
        with torch.no_grad():
            # split the batch into input, target and covariates
            input, target, covariates = utils.split_batch(test_batch)
            # input in the conditioning window. For the prediction window, model output will be conditioned on the
            # model output for the previous step
            input_cond = input[:, 0:cond_win_len, :]
            # Scale the input and oovariates so they are 0 mean and range = 1.
            input_cond, covariates = utils.scale(input_cond, covariates)
            # Run the forward pass
            pred = model(input_cond, covariates, future=pred_win_len)
            # scale model output and target to the original data scale.
            pred = utils.invert_scale(pred)
            # calculate loss only for the prediction window
            pred_ = pred[:, cond_win_len:cond_win_len + pred_win_len, :]
            target_ = target[:, cond_win_len:cond_win_len + pred_win_len, :]
            loss = np.sqrt(criterion(pred_, target_).item())
            rmse[k].append(loss)
            mae[k].append(torch.nn.functional.l1_loss(pred_, target_).item())
            nrmse[k].append(utils.NRMSE(pred_, target_).item())
            nd[k].append(utils.NormDeviation(pred_, target_).item())
            print("loss (prediction): {0}".format(loss))
            preds = pred[0, :, :].detach().cpu().numpy()
            targets = target[0, :, :].detach().cpu().numpy()

        # just plot prediction corresponding to first target series
        for j in range(0,1):
            pred = preds[:, j]  # target series j
            handle, = plt.plot(np.arange(cond_win_len), pred[0:cond_win_len], colors[k], linewidth=2.0)
            handles.append(handle)
            plt.plot(np.arange(cond_win_len, cond_win_len + pred_win_len), pred[cond_win_len:], colors[k] + ':',
                     linewidth=2.0)
    plt.legend(handles, legends)
    plt.show()
    for k in np.arange(0, len(models)):
        print("RMSE:{0}, MAE:{1}, NRMSE:{2}, ND:{3}".format(rmse[k][i], mae[k][i], nrmse[k][i], nd[k][i]))

for k in np.arange(0, len(models)):
        print("RMSE:{0}, MAE:{1}, NRMSE:{2}, ND:{3}".format(np.mean(rmse[k]), np.mean(mae[k]), np.mean(nrmse[k]), np.mean(nd[k])))
print('done')
