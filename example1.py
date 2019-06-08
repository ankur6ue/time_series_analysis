import models.model as model
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data.dataloader import DataLoader
from utils.utils import TSUtils
from utils.utils import save_model, load_model

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

num_covariates = cfg['num_covariates']
num_time_idx = cfg['num_time_idx']
num_targets = cfg['num_targets']
input_dim = num_time_idx + num_targets + num_covariates
# Total context window size used for training. The context window consists of 1. conditioning_context where input
# data points are available and network predictions are conditioned on actual input data, and 2. prediction_context,
# where the network predictions are conditioned on network output at the previous step. I use half of the context
# for training and half for prediction. Feel free to try other combinations. Covariates such as time step are assumed
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

def train():
    for epoch in range(0, num_epochs):
        batch_num = 0
        for i, batch in enumerate(train_dataloader):
            if batch_num > max_batches_per_epoch:
                break
            input, target, covariates = utils.split_batch(batch)
            input_cond = input[:, 0:cond_win_len, :]
            input_cond, covariates = utils.scale(input_cond, covariates)
            # cond_ctx_len is partition of the training data over which network output is conditioned on input data. pred_ctx_len is
            # the other part of the partition where the network output is conditioned on previous network outputs.
            optimizer.zero_grad()
            out = model(input_cond, covariates, future=pred_win_len)
            # scale model output and target to the original data scale
            out = utils.invert_scale(out)
            loss = criterion(out[:], target[:])

            loss.backward()
            scheduler.step()
            optimizer.step()
            loss = np.sqrt(loss.item())
            losses.append(loss)
            print('epoch:{0}/{1}, step:{2}, loss:{3}'.format(epoch, num_epochs, batch_num, loss))
            batch_num = batch_num + 1

    return model, losses

######## plot loss ############
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
    rmse = []
    mae = []
    mape = []
    for i in range(0, 10):
        test_batch = next(test_dataloader_iter)
        with torch.no_grad():
            input, target, covariates = utils.split_batch(test_batch)
            input, covariates = utils.scale(input, covariates)
            pred = model(input[:, 0:cond_win_len, :], covariates, future=pred_win_len)
            pred = utils.invert_scale(pred)
            loss = np.sqrt(criterion(pred, target).item())
            rmse.append(loss)
            mae.append(torch.nn.functional.l1_loss(pred, target).item())
            print("loss (prediction): {0}".format(loss))
            preds = pred[0, :, :].detach().cpu().numpy()
            targets = target[0, :, :].detach().cpu().numpy()
        # plot the target series on a separate plot
        for j in range(num_targets):
            pred = preds[:, j]  # target series j
            target = targets[:, j]
            plt.figure()
            plt.title(
                'Red: Groundtruth, Green: Predicted values \n Dashlines: predicted values in the prediction window')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xticks()
            plt.yticks()
            plt.plot(np.arange(len(target)), target, 'r', linewidth=2.0)
            plt.plot(np.arange(cond_win_len), pred[0:cond_win_len], 'g', linewidth=2.0)
            plt.plot(np.arange(cond_win_len, cond_win_len + pred_win_len), pred[cond_win_len:], 'g' + ':',
                     linewidth=2.0)
            #plt.show()
    print("RMSE:{0}, MAE:{1}".format(np.mean(rmse), np.mean(mae)))

if __name__ == "__main__":
    model, losses = train()
    file_prefix = "{0}\\model_covariates-{1}_epochs{2}".format(example, cfg['num_covariates'],
                                                                       cfg['num_epochs'])
    #model = load_model(model, file_prefix)
    #save_model_(model, file_prefix)
    predict(model, num_targets)

