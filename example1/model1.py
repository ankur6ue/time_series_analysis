""" Implements simple time-series prediction using LSTMs
"""
# Import necessary libraries
import torch 
import torch.nn as nn

class model1(nn.Module):
    def __init__(self, num_lstms, input_dim):
        super(model1, self).__init__()
        self.lstm_out = 51
        self.num_lstms = num_lstms
        lstms = []

        lstms.append(nn.LSTMCell(input_dim, self.lstm_out))
        for i in range(1, self.num_lstms):
            lstms.append(nn.LSTMCell(self.lstm_out, self.lstm_out))

        self.lstms = nn.ModuleList(lstms)
        self.linear1 = nn.Linear(self.lstm_out, 1)

    def forward(self, input, future = 0):
        outputs = []
        dev = input.device
        h_t = []
        c_t = []
        train_ctx_size = input.size(1)
        for i in range(0, self.num_lstms):
            h_t.append(torch.zeros(input.size(0), self.lstm_out, dtype=torch.float).to(dev))
            c_t.append(torch.zeros(input.size(0), self.lstm_out, dtype=torch.float).to(dev))

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t[0], c_t[0] = self.lstms[0](input_t, (h_t[0], c_t[0]))
            for n in range(1, self.num_lstms):
                h_t[n], c_t[n] = self.lstms[n](h_t[n-1], (h_t[n], c_t[n]))
            output = self.linear1(h_t[n])
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t[0], c_t[0] = self.lstms[0](output, (h_t[0], c_t[0]))
            for n in range(1, self.num_lstms):
                h_t[n], c_t[n] = self.lstms[n](h_t[n - 1], (h_t[n], c_t[n]))
            output = self.linear1(h_t[n])
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs
        