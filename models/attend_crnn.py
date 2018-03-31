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
    def __init__(self, input_dim, output_dim, use_cuda, seqlen=26, relation_aware=False):
        super(AttentionLayer, self).__init__()

        self.output_dim = output_dim
        self.use_cuda = use_cuda
        self.relation_aware = relation_aware

        self.linear_v = nn.Linear(input_dim, output_dim)
        self.linear_q = nn.Linear(input_dim, output_dim)
        self.linear_k = nn.Linear(input_dim, output_dim)

        if self.relation_aware:
            self.alpha_V = nn.Parameter(torch.zeros((seqlen, seqlen)))
            self.alpha_K = nn.Parameter(torch.zeros((seqlen, seqlen)))

    def forward(self, x):

        batch_size, seq_len, num_features = x.size()
        x_k = self.linear_k(x)
        x_q = self.linear_q(x)
        x_v = self.linear_v(x)

        if not self.relation_aware:
            atten_energies = torch.matmul(x_q, x_k.transpose(2, 1))
            atten_energies = torch.stack([F.softmax(atten_energies[i]) for i in xrange(batch_size)])


            # softmax_atten_energies = Variable(torch.zeros((batch_size, seq_len, seq_len)))
            #
            # if self.use_cuda:
            #     softmax_atten_energies = softmax_atten_energies.cuda()
            # for batch in xrange(batch_size):
            #     for i in xrange(seq_len):
            #         softmax_atten_energies[batch, i] = atten_energies[batch, i] / torch.sum(atten_energies[batch, i])
            z = torch.matmul(atten_energies, x_v)
        else:
            x_m = x_k + self.alpha_K
            atten_energies = torch.matmul(x_q, x_m.transpose(2, 1))
            _v = x_v + self.alpha_V
            z = torch.matmul(atten_energies, _v)

        return z, atten_energies


def __test__attention_layer():
    atten_layer = AttentionLayer(input_dim=5, output_dim=4, use_cuda=True)
    x = Variable(torch.randn((2, 3, 5)))
    print(x)
    result, _ = atten_layer(x)
    print(result)
    y = Variable(torch.randn((2, 3, 4)))
    loss_criterion = nn.MSELoss()
    loss = loss_criterion(result, y)
    import torch.optim as optimizer
    optimizer = optimizer.Adam(lr=0.001, params=atten_layer.parameters())
    optimizer.zero_grad()
    loss.backward()
    print(loss)
    optimizer.step()


__test__attention_layer()


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
        self.attend_layer = AttentionLayer(input_dim=hidden_size, output_dim=hidden_size, use_cuda=use_cuda)
        self.rnn1 = BidirectionalLSTM(512, hidden_size, hidden_size)
        self.rnn2 = BidirectionalLSTM(hidden_size, hidden_size, num_class)
        # self.rnn = nn.Sequential(
        #     BidirectionalLSTM(512, hidden_size, hidden_size),
        #     BidirectionalLSTM(hidden_size, hidden_size, num_class))

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        # conv = conv.permute(0, 2, 1)  # [b, w, c]
        # attend = self.attend_layer(conv)
        # attend = attend.permute(1, 0, 2)  # [w, b, c]
        # output = self.rnn(attend)

        conv = conv.permute(2, 0, 1)  # [w, b, c]
        rnn = self.rnn1(conv)
        rnn = rnn.permute(1, 0, 2)    # [b, w, c]
        attend, attend_energies = self.attend_layer(rnn)
        attend = attend.permute(1, 0, 2)  # [w, b, c]
        output = self.rnn2(attend)
        return output, attend_energies


def __test_atten_crnn__():

    atten_crnn = AttendCRNN(1, 256, 27, False)

    images = Variable(torch.randn((2, 1, 32, 100)))

    atten_crnn(images)

