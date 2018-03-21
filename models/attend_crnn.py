# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class BidirectionalLSTM(nn.Module):
    def __init__(self, num_features, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(num_features, hidden_size, bidirectional=True)
        self.output = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_features):
        recurrent, _ = self.rnn(input_features)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.output(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, relation_aware=False):
        super(AttentionLayer, self).__init__()

        self.output_dim = output_dim
        self.linear_v = nn.Linear(input_dim, output_dim)
        self.linear_q = nn.Linear(input_dim, output_dim)
        self.linear_k = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, num_features = x.size()
        z = Variable(torch.zeros((batch_size, seq_len, self.output_dim)))
        for i in xrange(batch_size):
            xq = self.linear_q(x[i])
            xk = self.linear_k(x[i])
            xv = self.linear_v(x[i])

            e = Variable(torch.zeros((seq_len, seq_len)))
            alpha = Variable(torch.zeros((seq_len, seq_len)))

            for m in xrange(seq_len):
                for n in xrange(seq_len):
                    e[m, n] = torch.exp(xq[m].dot(xk[n]) / np.sqrt(self.output_dim))
                alpha[m] = e[m] / torch.sum(e[m])
                for n in xrange(seq_len):
                    z[i, m] = alpha[m][n]*xv[n]

        return z


def __test__attention_layer():
    atten_layer = AttentionLayer(input_dim=4, output_dim=3)
    x = Variable(torch.randn((2, 3, 4)))
    print(x)
    result = atten_layer(x)
    print(result)


class AttendCRNN(nn.Module):
    def __init__(self):
        super(AttendCRNN, self).__init__()

    def forward(self, x):
        pass

