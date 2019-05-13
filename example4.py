from dataset.time_series import MockTs
import example4.model4 as model4
import torch
import torch.optim as optim
import math

from matplotlib import pyplot as plt
import numpy as np

ts = MockTs()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# takes a batch and splits it into input, target and covariates. The implementation depends on the
# type of dataset used. In this example, the covariate is just the time index
def split_batch(batch):
    # convert from numpy to pytorch tensor
    # target is the input time shifted by one unit
    # convert from numpy to pytorch tensor
    input = torch.from_numpy(batch[0]).to(device).float()
    target = torch.from_numpy(batch[1]).to(device).float()
    time_idx = torch.from_numpy(batch[2]).to(device).float()

    # Add extra dimension to the input and time_idx (covariate) so they can be concatenated
    # along the third dimension.
    input = input.unsqueeze(2)
    time_idx = time_idx.unsqueeze(2)
    covariates = time_idx
    return input, target, covariates

def ts_generator(ts_obj, n_steps, batch_size):
    """
    This is a util generator function for Keras
    :param ts_obj: a Dataset child class object that implements the 'next_batch' method
    :param n_steps: parameter that specifies the length of the net's input tensor
    :return:
    """
    while 1:
        batch = ts_obj.next_batch(batch_size, n_steps)
        yield batch[0], batch[1], batch[2]
# Total context window size used for training. The context window consists of 1. conditioning_context where input
# data points are available and network predictions are conditioned on actual input data, and 2. prediction_context,
# where the network predictions are conditioned on network output at the previous step. I use half of the context
# for training and half for prediction. Feel free to try other combinations. Covariates such as time step are assumed
# to be available for the entire context window
context_size = 100
batch_size = 20
batch_gen = ts_generator(ts, context_size, batch_size)
model4 = model4.model4(2, 2).to(device)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model4.parameters(), lr=0.001)
# I tried decreasing the learning rate every 100 epochs - dividing by 2 (gamma = 0.5), 10 (gamma = 0.1) etc., but
# it didn't make a difference in my experiments.
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1)
num_batches = 400
batch_num = 1
for batch in batch_gen:
    if batch_num > num_batches:
        break
    input, target, covariates = split_batch(batch)
    # cond_ctx_len is partition of the training data over which network output is conditioned on input data. pred_ctx_len is
    # the other part of the partition where the network output is conditioned on previous network outputs.
    cond_ctx_len = (int)(input.shape[1])
    pred_ctx_len = (int)(input.shape[1]) - cond_ctx_len

    optimizer.zero_grad()
    out = model4(input[:, 0:cond_ctx_len, :], covariates, future=pred_ctx_len)
    loss = model4.NLL(out, target)
    loss.backward()
    scheduler.step()
    optimizer.step()
    print('step: {0}, loss: {1}'.format(batch_num, loss.item()))
    batch_num = batch_num + 1

with torch.no_grad():
    test_data = ts.next_batch(1, context_size)
    input, target, covariates = split_batch(test_data)
    cond_ctx_len = (int)(input.shape[1]/2)
    pred_ctx_len = (int)(input.shape[1]) - cond_ctx_len
    pred = model4(input[:, 0:cond_ctx_len, :], covariates, future=pred_ctx_len)
    y = pred[0, :].detach().cpu().numpy()
    target = target[0, :].detach().cpu().numpy()

plt.figure()
plt.title('Red: Groundtruth, Green: Predicted values \n Dashlines: predicted values in the prediction window')
plt.xlabel('x')
plt.ylabel('y')
plt.xticks()
plt.yticks()
mean = y[:,0]
std = y[:,1]
xax = np.arange(target.size)
xax_pred = np.arange(cond_ctx_len)
xax_cond = np.arange(cond_ctx_len,  cond_ctx_len + pred_ctx_len)
plt.plot(xax, target, 'r', linewidth=2.0)
plt.plot(xax_pred, mean[0:cond_ctx_len], 'g', linewidth=2.0)
plt.plot(xax_cond, mean[cond_ctx_len:], 'g'+':', linewidth=2.0)
plt.fill_between(xax, mean-std, mean+std, color='gray', alpha=0.5)
plt.fill_between(xax, mean-2*std, mean+2*std, color='gray', alpha=0.2)
plt.show()
print('done')