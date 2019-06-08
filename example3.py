from dataset.time_series import MockTs
import example3.model3 as model3
import torch
import torch.optim as optim
import math

from matplotlib import pyplot as plt
import numpy as np

ts = MockTs()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
    """Build sinusoidal embeddings.

    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb

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
embed_dim = 2
model3 = model3.model3(2, embed_dim+1).to(device)
embeddings = get_embedding(400,5)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model3.parameters(), lr=0.001)
# I tried decreasing the learning rate every 100 epochs - dividing by 2 (gamma = 0.5), 10 (gamma = 0.1) etc., but
# it didn't make a difference in my experiments.
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1)
num_batches = 400
batch_num = 1
for batch in batch_gen:
    if batch_num > num_batches:
        break
    # convert from numpy to pytorch tensor
    input = torch.from_numpy(batch[0]).to(device).float()
    target = torch.from_numpy(batch[1]).to(device).float()
    time_idx = torch.from_numpy(batch[2]).to(device).float()
    # cond_ctx_len is partition of the training data over which network output is conditioned on input data. pred_ctx_len is
    # the other part of the partition where the network output is conditioned on previous network outputs.
    cond_ctx_len = (int)(input.shape[1]/2)
    pred_ctx_len = (int)(input.shape[1]) - cond_ctx_len

    optimizer.zero_grad()
    # convert to indices. Multiplication by 10 because the resolution of time series is 0.1
    time_idx1 = (time_idx * 10.0).type(torch.IntTensor)
    embed_dim = embeddings.shape[1]
    [h, w] = time_idx1.shape
    em = embeddings[time_idx1.view(-1).long(), :].view(h, w, embed_dim)
    em2 = embeddings[(time_idx1+1).view(-1).long(), :].view(h, w, embed_dim)
    sint = torch.unsqueeze((time_idx), 2)
    cost = torch.unsqueeze((time_idx), 2)
    em = torch.cat((sint, cost), 2)
    em = em.to(device)
    input = input[:, 0: cond_ctx_len].unsqueeze(2)
    # Get embedding corresponding to the time index
    out = model3(input, em, future=pred_ctx_len)
    loss = criterion(out, target)
    loss.backward()
    scheduler.step()
    optimizer.step()
    print('step: {0}, loss: {1}'.format(batch_num, loss.item()))
    batch_num = batch_num + 1


with torch.no_grad():
    test_data = ts.next_batch(1, 100)
    input = torch.from_numpy(test_data[0]).cuda().float()
    time_idx = torch.from_numpy(test_data[2]).to(device).float()
    target = test_data[1]
    cond_ctx_len = (int)(input.shape[1] / 2)
    pred_ctx_len = (int)(input.shape[1]) - cond_ctx_len

    pred = model2(input[:, 0:cond_ctx_len], time_idx, future=pred_ctx_len)
    y = pred[0, :].detach().cpu().numpy()

plt.figure()
plt.title('Red: Groundtruth, Green: Predicted values \n Dashlines: predicted values in the prediction window')
plt.xlabel('x')
plt.ylabel('y')
plt.xticks()
plt.yticks()
plt.plot(np.arange(target.size), target[0], 'r', linewidth=2.0)
plt.plot(np.arange(cond_ctx_len), y[0:cond_ctx_len], 'g', linewidth=2.0)
plt.plot(np.arange(cond_ctx_len,  cond_ctx_len + pred_ctx_len), y[cond_ctx_len:], 'g'+':', linewidth=2.0)
plt.show()
print('done')