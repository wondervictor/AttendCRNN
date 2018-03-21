# -*- coding: utf-8 -*-

import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, num_features, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(num_features, hidden_size, bidirectional=True)
        self.output = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_features):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.output(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()


    def forward(self, x):

        pass


class AttendCRNN(nn.Module):

    def __init__(self):
        super(AttendCRNN, self).__init__()

    def forward(self, x):

        pass