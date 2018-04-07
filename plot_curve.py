# -*- coding: utf-8 -*-

import argparse
import seaborn as sbn
import numpy as np
import matplotlib.pyplot as plt



def open_log_txt(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()

    def filter_train(line):
        if line[0] != '[':
            return False
        return True

    def filter_test(line):
        if line[0] != 'T':
            return False
        return True

    test_loss = map(lambda x: x.replace(',', ':').split(':'), filter(filter_test, lines))
    test_loss = map(lambda x: (float(x[1][1:]), float(x[3][1:])), test_loss)
    train_loss = map(lambda x: float(x.rstrip('\n\r').split(':')[1][1:]), filter(filter_train, lines))

    return train_loss, test_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', required=True, help='log file need to plot')

    opt = parser.parse_args()
    crnn_attention = open_log_txt(opt.log_dir + '/crnn_attention_log.txt')
    crnn_attention1 = open_log_txt(opt.log_dir + '/crnn_attention_relation_aware_log.txt')
    crnn = open_log_txt(opt.log_dir + '/crnn_log.txt')

    plt.figure(1, figsize=(20, 6))
    sbn.set_style('dark')
    plt.title('Training Loss')
    plt.plot(crnn_attention[0], 'r-', label='CRNN Attention')
    plt.plot(crnn[0], 'g-', label='CRNN')
    plt.plot(crnn_attention1[0], 'y-', label='CRNN Attention Relation Aware')
    plt.xlabel('train iter')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.figure(2, figsize=(20, 6))
    plt.title('Testing Loss')
    plt.plot([x[0] for x in crnn_attention[1]], 'r-', label='CRNN Attention')
    plt.plot([x[0] for x in crnn[1]], 'g-', label='CRNN')
    plt.plot([x[0] for x in crnn_attention1[1]], 'y-', label='CRNN Attention Relation Aware')
    plt.xlabel('test iter')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.figure(3, figsize=(20, 6))
    plt.title('Testing Accuracy')
    plt.plot([x[1] for x in crnn_attention[1]], 'r-', label='CRNN Attention')
    plt.plot([x[1] for x in crnn[1]], 'g-', label='CRNN')
    plt.plot([x[1] for x in crnn_attention1[1]], 'y-', label='CRNN Attention Relation Aware')
    plt.xlabel('test iter')
    plt.ylabel('test accuracy')
    """
    0.410239
    0.409242
    0.439827
    """
    print(max([x[1] for x in crnn_attention1[1]]))
    print(max([x[1] for x in crnn_attention[1]]))
    print(max([x[1] for x in crnn[1]]))
    plt.legend()
    plt.show()

