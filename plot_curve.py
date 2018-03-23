# -*- coding: utf-8 -*-

import argparse
import seaborn as sbn
import numpy as np
import matplotlib.pyplot as plt



def open_log_txt(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()

    def filter_(line):
        if line[0] != '[':
            return False
        return True

    lines = filter(filter_, lines)
    data = map(lambda x: float(x.rstrip('\n\r').split(':')[1][1:]), lines)
    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', required=True, help='log file need to plot')

    opt = parser.parse_args()

    crnn_attention = open_log_txt(opt.log_dir + '/1.txt')
    crnn = open_log_txt(opt.log_dir + '/2.txt')

    sbn.set_style('dark')
    #sbn.regplot(y=data)
    plt.title('Training Loss: 200 Iters')
    plt.plot(crnn_attention, 'r-', label='crnn_attention loss')
    plt.plot(crnn, 'g-', label='crnn loss')
    plt.xlabel('train iter')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


