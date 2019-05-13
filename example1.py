from dataset.time_series import MockTs
import example1.model1 as model1
import torch
import torch.optim as optim
import math

from matplotlib import pyplot as plt
import numpy as np

ts = MockTs()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

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

context_size = 100
batch_size = 20
batch_gen = ts_generator(ts, context_size, batch_size)
model1 = model1.model1(2, 1).to(device)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model1.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
num_batches = 200
batch_num = 1
for batch in batch_gen:
    if batch_num > num_batches:
        break
    input = torch.from_numpy(batch[0]).to(device).float()
    target = torch.from_numpy(batch[1]).to(device).float()
    time_idx = torch.from_numpy(batch[2]).to(device).float()

    optimizer.zero_grad()
    future = (int)(context_size/2)
    out = model1(input[:, 0:-future], future=future)
    loss = criterion(out, target)
    loss.backward()
    scheduler.step()
    optimizer.step()
    print('step: {0}, loss: {1}'.format(batch_num, loss.item()))
    batch_num = batch_num + 1

with torch.no_grad():
    test_data = ts.next_batch(1, 100)
    input = torch.from_numpy(test_data[0][:, 0:-future]).cuda().float()
    target = test_data[1]
    future = 50
    pred = model1(input, future=future)
    y = pred[0, :].detach().cpu().numpy()

plt.figure()
plt.title('Red: Groundtruth, Green: Predicted values \n Dashlines: predicted values in the prediction window')
plt.xlabel('x')
plt.ylabel('y')
plt.xticks()
plt.yticks()
plt.plot(np.arange(target.size), target[0], 'r', linewidth=2.0)
plt.plot(np.arange(input.shape[1]), y[0:input.shape[1]], 'g', linewidth=2.0)
plt.plot(np.arange(input.shape[1],  input.shape[1] + future), y[input.shape[1]:], 'g'+':', linewidth=2.0)
plt.show()
print('done')