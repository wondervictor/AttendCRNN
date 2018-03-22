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
    def __init__(self, input_dim, output_dim, use_cuda, relation_aware=False):
        super(AttentionLayer, self).__init__()

        self.output_dim = output_dim
        self.use_cuda = use_cuda
        self.linear_v = nn.Linear(input_dim, output_dim)
        self.linear_q = nn.Linear(input_dim, output_dim)
        self.linear_k = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, num_features = x.size()
        z = Variable(torch.zeros((batch_size, seq_len, self.output_dim)))
        if self.use_cuda:
            z = z.cuda()
        xq = self.linear_q(x)
        xk = self.linear_k(x)
        xv = self.linear_v(x)
        # 性能瓶颈
        for i in xrange(batch_size):

            e = Variable(torch.zeros((seq_len, seq_len)))
            alpha = Variable(torch.zeros((seq_len, seq_len)))
            if self.use_cuda:
                e = e.cuda()
                alpha = alpha.cuda()
            for m in xrange(seq_len):
                for n in xrange(seq_len):
                    e[m, n] = torch.exp(xq[i, m].clone().dot(xk[i, n].clone()) / np.sqrt(self.output_dim))
                alpha[m] = e[m].clone() / torch.sum(e[m])
                for n in xrange(seq_len):
                    z[i, m] = alpha[m, n].clone()*xv[i, n].clone()

        return z


def __test__attention_layer():
    atten_layer = AttentionLayer(input_dim=4, output_dim=3)
    x = Variable(torch.randn((2, 3, 4)))
    print(x)
    result = atten_layer(x)
    print(result)
    y = Variable(torch.randn((2, 3, 3)))
    loss_criterion = nn.MSELoss()
    loss = loss_criterion(result, y)
    import torch.optim as optimizer
    optimizer = optimizer.Adam(lr=0.001, params=atten_layer.parameters())
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()


# __test__attention_layer()


class AttendCRNN(nn.Module):
    def __init__(self, nc, hidden_size, num_class, use_cuda, leaky_relu=True):
        super(AttendCRNN, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            in_chan = nc if i == 0 else nm[i - 1]
            out_chan = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(in_chan, out_chan, ks[i], ss[i], ps[i]))
            if batch_norm:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(out_chan))
            if leaky_relu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True, inplace=True))

        conv_relu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        conv_relu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        conv_relu(2, True)
        conv_relu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        conv_relu(4, True)
        conv_relu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        conv_relu(6, True)  # 512x1x16

        self.cnn = cnn

        self.attend_layer = AttentionLayer(input_dim=512, output_dim=512, use_cuda=use_cuda)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_class))

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)  # [b, w, c]
        attend = self.attend_layer(conv)
        attend = attend.permute(1, 0, 2)  # [w, b, c]
        output = self.rnn(attend)
        return output


def __test_atten_crnn__():

    atten_crnn = AttendCRNN(1, 256, 27)

    images = Variable(torch.randn((2, 1, 32, 100)))

    atten_crnn(images)

