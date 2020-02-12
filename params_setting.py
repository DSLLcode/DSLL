# -*- coding:utf-8 -*-

# Deep Streaming Label Learning

from collections import namedtuple
import time
import torch

hyper_params = namedtuple('args', ['classifier_epoch', 'label_mapping_epoch', 'label_representation_epoch', 'KD_epoch',
                                   'classifier_dropout', 'label_mapping_dropout', 'label_representation_dropout', 'KD_dropout',
                                   'classifier_L2', 'label_mapping_L2', 'label_representation_L2', 'KD_L2',
                                   'label_mapping_hidden1', 'label_representation_hidden1', 'classifier_hidden1', 'KD_hidden1',
                                   'label_mapping_hidden2', 'label_representation_hidden2', 'classifier_hidden2', 'KD_hidden2',
                                   'classifier_input_dim', 'classifier_output_dim', 'label_mapping_input_dim', 'KD_input_dim',
                                   'label_mapping_output_dim', 'label_representation_input_dim', 'label_representation_output_dim',
                                   'KD_output_dim', 'dataset_name', 'N', 'D', 'M_full', 'M', 'N_test', 'M_new','time',
                                   'device','batch_size','model_name', 'loss','currentEpoch','batchNorm','algorithm', 'changeloss'])
# default
hyper_params.time = time.strftime("%Y-%m-%d_%H:%M:%S")
hyper_params.device = 'cuda' if torch.cuda.is_available() else 'cpu'
hyper_params.batch_size = 50
hyper_params.batchNorm = False
hyper_params.algorithm = 'DSLL'
hyper_params.currentEpoch = 0
hyper_params.changeloss = False

def get_params(dataset_name):
    hyper_params.dataset_name = dataset_name
    if dataset_name == 'yeast':
        hyper_params.classifier_dropout = 0.1
        hyper_params.label_mapping_dropout = 0.1
        hyper_params.label_representation_dropout = 0.1
        hyper_params.classifier_L2 = 1e-08
        hyper_params.label_mapping_L2 = 1e-08
        hyper_params.label_representation_L2 = 1e-08

        hyper_params.classifier_hidden1 = 200
        hyper_params.classifier_hidden2 = 100
        hyper_params.label_mapping_hidden1 = 64
        hyper_params.label_mapping_hidden2 = 0
        hyper_params.label_representation_hidden1 = 200
        hyper_params.label_representation_hidden2 = 100

        hyper_params.KD_hidden1 = 200
        hyper_params.KD_hidden2 = 100
        hyper_params.KD_epoch = 20
        hyper_params.KD_L2 = 1e-08
        hyper_params.KD_dropout = 0.1

        hyper_params.classifier_epoch = 200
        hyper_params.label_mapping_epoch = 20
        hyper_params.label_representation_epoch = 200
    return hyper_params

