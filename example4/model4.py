""" Implements simple time-series prediction using LSTMs
"""
# Import necessary libraries
import torch 
import torch.nn as nn

class model4(nn.Module):
    def __init__(self, num_lstms, input_dim):
        super(model4, self).__init__()
        self.lstm_out = 51
        self.num_lstms = num_lstms
        lstms = []

        lstms.append(nn.LSTMCell(input_dim, self.lstm_out))
        for i in range(1, self.num_lstms):
            lstms.append(nn.LSTMCell(self.lstm_out, self.lstm_out))

        self.lstms = nn.ModuleList(lstms)
        self.linear1 = nn.Linear(self.lstm_out, 2)
        self.linear2 = nn.Linear(self.lstm_out, 2)
        self.criterion = torch.nn.MSELoss()

    def forward(self, input, time_idx, future = 0):
        outputs = []
        dev = input.device
        h_t = []
        c_t = []
        # add extra dimension to input and time_index so we can use torch.cat on them
        cond_ctx_len = input.size(1)
        bsz = input.size(0)
        pred_ctx_len = (int)(time_idx.shape[1]) - cond_ctx_len

        # concatenate input and covariate
        input = torch.cat((input, time_idx[:, 0:cond_ctx_len, :]), 2)
        for i in range(0, self.num_lstms):
            h_t.append(torch.zeros(input.size(0), self.lstm_out, dtype=torch.float).to(dev))
            c_t.append(torch.zeros(input.size(0), self.lstm_out, dtype=torch.float).to(dev))

        outputs = torch.Tensor().to(dev)
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t[0], c_t[0] = self.lstms[0](input_t.squeeze(1), (h_t[0], c_t[0]))
            for n in range(1, self.num_lstms):
                h_t[n], c_t[n] = self.lstms[n](h_t[n - 1], (h_t[n], c_t[n]))
            output = self.linear1(h_t[n])
            outputs = torch.cat((outputs, output), 0)
        # Running softplus on the batch of outputs is a lot faster than running it in the loop
        outputs[:, 1] = self.softplus(outputs[:, 1])
        # must run softplus on the last output before running prediction on it
        output[:, 1] = self.softplus(output[:, 1])
        for i in range(future):  # if we should predict the future
            # sample from the distribution
            sample = self.sample(output)
            output_t = torch.cat((sample, time_idx[:, cond_ctx_len + i, :]), 1)
            h_t[0], c_t[0] = self.lstms[0](output_t, (h_t[0], c_t[0]))
            for n in range(1, self.num_lstms):
                h_t[n], c_t[n] = self.lstms[n](h_t[n - 1], (h_t[n], c_t[n]))
            output = self.linear1(h_t[n])
            # for prediction, we must run softplus on each output because we draw a sample from it
            # to pass to the next prediction step
            output[:, 1] = self.softplus(output[:, 1])
            outputs = torch.cat((outputs, output), 0)
        # convert to dim (bsz, len, 2)
        outputs = torch.stack(outputs.split(bsz, 0), 1).squeeze(2)
        return outputs


    def sample(self, output):
        mean, std = torch.split(output, 1, dim=1)
        normal_dist = torch.distributions.normal.Normal(mean, std)
        return normal_dist.sample()

    def softplus(self, x):
        """ Positivity constraint """
        softplus = torch.log(1+torch.exp(x))
        # Avoid infinities due to taking the exponent
        softplus = torch.where(softplus==float('inf'), x, softplus)
        return softplus

    def NLL(self, outputs, truth):
        mean, std = torch.split(outputs, 1, dim=2)
        mean = mean.squeeze(2)
        std = std.squeeze(2)
        diff = torch.sub(truth, mean)
        #loss =  torch.mean(torch.pow(diff, 2))
        #loss_verify = self.criterion(mean, truth)
        loss = torch.mean(torch.div(torch.pow(diff, 2), torch.pow(std, 2))) + 2*torch.mean(torch.log(std))
        return loss
        