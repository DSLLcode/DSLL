# -*- coding:utf-8 -*-

# Deep Streaming Label Learning

from helpers import split_label
import arff
import numpy as np

def load_dataset(dataset):
    if dataset == "yeast":
        data_dir = 'datasets/'
        train_data = arff.load(open(data_dir + 'yeast-train.arff', 'rt'))
        train_data = np.array(train_data['data']).astype(np.float)

        test_data = arff.load(open(data_dir + 'yeast-test.arff', 'rt'))
        test_data = np.array(test_data['data']).astype(np.float)

        train_X = train_data[:, :103]
        train_Y_full = train_data[:, 103:]
        train_Y, train_Y_rest = split_label(train_Y_full, 0.5)

        test_X = test_data[:, :103]
        test_Y_full = test_data[:, 103:]
        test_Y, test_Y_rest = split_label(test_Y_full, 0.5)

    return train_X, train_Y, train_Y_rest, test_X, test_Y, test_Y_rest

