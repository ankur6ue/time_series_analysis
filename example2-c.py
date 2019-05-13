from dataset.time_series import MockTs
import example2.model2 as model2
import torch
import torch.optim as optim
import math
from dataset.pollution_load import DataLoader, PollutionDataset, RandomSampler
from matplotlib import pyplot as plt
import numpy as np

ts = MockTs()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# takes a batch and splits it into input, target and covariates. The implementation depends on the
# type of dataset used
def split_batch(batch):
    # convert from numpy to pytorch tensor
    # target is the input time shifted by one unit
    input = batch[:, 0:-2, 1].to(device).float()
    target = batch[:, 1:-1, 1].to(device).float()
    time_idx = batch[:, :, 0].to(device).float()

    # Add extra dimension to the input and time_idx (covariate) so they can be concatenated
    # along the third dimension.
    input = input.unsqueeze(2)
    time_idx = time_idx.unsqueeze(2)
    covariates = torch.cat((time_idx, batch[:, :, 2:7]), -1)
    return input, target, covariates

def rescale_predictions(pred):
    min, max = pollution_dataset.get_scale_params()
    return torch.mul(torch.add(pred, min), max)

def rescale_target(target):
    min, max = pollution_dataset.get_scale_params()
    return torch.mul(torch.add(target, min), max)

# Total context window size used for training. The context window consists of 1. conditioning_context where input
# data points are available and network predictions are conditioned on actual input data, and 2. prediction_context,
# where the network predictions are conditioned on network output at the previous step. I use half of the context
# for training and half for prediction. Feel free to try other combinations. Covariates such as time step are assumed
# to be available for the entire context window
ctx_win_len = 20
batch_size = 5
model2 = model2.model2(num_lstms=2, input_dim=7).to(device)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model2.parameters(), lr=0.001)
# I tried decreasing the learning rate every 100 epochs - dividing by 2 (gamma = 0.5), 10 (gamma = 0.1) etc., but
# it didn't make a difference in my experiments.
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1)
batch_num = 1
pollution_dataset = PollutionDataset(csv_file='/../data/pollution.csv', dev=device, ctx_win_len=ctx_win_len)
sampler = RandomSampler(pollution_dataset.train_size, 0, ctx_win_len)  #DummySampler(dataset)  #None #
dataloader = DataLoader(pollution_dataset, batch_size=batch_size, sampler=sampler,
                        shuffle=False, num_workers=0)
num_batches = 200
losses = []
for i, batch in enumerate(dataloader):
    if batch_num > num_batches:
        break
    input, target, covariates = split_batch(batch)
    # cond_ctx_len is partition of the training data over which network output is conditioned on input data. pred_ctx_len is
    # the other part of the partition where the network output is conditioned on previous network outputs.
    cond_ctx_len = (int)(input.shape[1])
    pred_ctx_len = (int)(input.shape[1]) - cond_ctx_len

    optimizer.zero_grad()
    out = model2(input[:, 0:cond_ctx_len, :], covariates, future=pred_ctx_len)

    # scale model output and target to the original data scale
    out = rescale_predictions(out)
    target = rescale_target(target)
    loss = criterion(out, target)
    losses.append(loss.item())
    loss.backward()
    scheduler.step()
    optimizer.step()
    print('step: {0}, loss: {1}'.format(batch_num, loss.item()))
    batch_num = batch_num + 1

# plot loss
plt.figure()
plt.title('Red: Groundtruth, Green: Predicted values \n Dashlines: predicted values in the prediction window')
plt.xlabel('batch no')
plt.ylabel('loss')
plt.plot(np.arange(len(losses)), losses, 'g', linewidth=2.0)
plt.show()

## testing
sampler = RandomSampler(pollution_dataset.total_data_size, pollution_dataset.train_size, ctx_win_len) #DummySampler(dataset)  #None #
test_dataloader = DataLoader(pollution_dataset, batch_size=1, sampler = sampler,
                        shuffle=False, num_workers=0)

with torch.no_grad():
    test_data = next(iter(test_dataloader))
    input, target, covariates = split_batch(test_data)
    cond_ctx_len = (int)(input.shape[1]/2)
    pred_ctx_len = (int)(input.shape[1]) - cond_ctx_len

    pred = model2(input[:, 0:cond_ctx_len, :], covariates, future=pred_ctx_len)
    pred = pred[0, :].detach().cpu().numpy()
    target = target[0, :].detach().cpu().numpy()

plt.figure()
plt.title('Red: Groundtruth, Green: Predicted values \n Dashlines: predicted values in the prediction window')
plt.xlabel('x')
plt.ylabel('y')
plt.xticks()
plt.yticks()
plt.plot(np.arange(len(target)), target, 'r', linewidth=2.0)
plt.plot(np.arange(cond_ctx_len), pred[0:cond_ctx_len], 'g', linewidth=2.0)
plt.plot(np.arange(cond_ctx_len,  cond_ctx_len + pred_ctx_len), pred[cond_ctx_len:], 'g'+':', linewidth=2.0)
plt.show()
print('done')