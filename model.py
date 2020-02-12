# -*- coding:utf-8 -*- 

# Deep Streaming Label Learning

import torch.nn as nn
import torch

class KnowledgeDistillation(nn.Module):
    def __init__(self, hyper_params):
        super(KnowledgeDistillation, self).__init__()
        self.W_m = nn.Sequential(
            nn.Linear(hyper_params.KD_input_dim, hyper_params.KD_hidden1),
            torch.nn.Dropout(hyper_params.KD_dropout),
            nn.ReLU(),
        )
    def forward(self, input):
        return self.W_m(input)


class IntegratedModel_3net(nn.Module):
    def __init__(self, hyper_params):
        super(IntegratedModel_3net, self).__init__()
        self.W_m = nn.Sequential(
            nn.Linear(hyper_params.classifier_input_dim, hyper_params.classifier_hidden1),
            torch.nn.Dropout(hyper_params.classifier_dropout),
            nn.ReLU(),
        )
        self.label_mapping = nn.Sequential(
            nn.Linear(hyper_params.label_mapping_input_dim, hyper_params.label_mapping_hidden1),
            torch.nn.Dropout(hyper_params.label_mapping_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.label_mapping_hidden1, hyper_params.label_mapping_output_dim),
        )
        self.representation = nn.Sequential(
            nn.Linear((hyper_params.classifier_hidden1 + hyper_params.label_mapping_output_dim),
                      hyper_params.label_representation_hidden1),
            torch.nn.Dropout(hyper_params.label_representation_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.label_representation_hidden1, hyper_params.label_representation_hidden2),
            torch.nn.Dropout(hyper_params.label_representation_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.label_representation_hidden2, hyper_params.label_representation_output_dim),
        )
    def forward(self, x, y_m):
        x_feature_kd = self.W_m(x)
        y_new_mapping = self.label_mapping(y_m.sigmoid())
        y_new_prediction = self.representation(torch.cat((x_feature_kd, y_new_mapping), 0))
        return y_new_prediction

class IntegratedModel_mapping(nn.Module):
    def __init__(self, hyper_params):
        super(IntegratedModel_mapping, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(hyper_params.classifier_input_dim, hyper_params.classifier_hidden1),
            torch.nn.Dropout(hyper_params.classifier_dropout),
            nn.ReLU(),
        )
        self.representation = nn.Sequential(
            nn.Linear((hyper_params.classifier_hidden1 + hyper_params.label_mapping_output_dim),
                      hyper_params.label_representation_hidden1),
            torch.nn.Dropout(hyper_params.label_representation_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.label_representation_hidden1, hyper_params.label_representation_hidden2),
            torch.nn.Dropout(hyper_params.label_representation_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.label_representation_hidden2, hyper_params.label_representation_output_dim),
        )
    def forward(self, x, y_m, mapping_model):
        x_feature_kd = self.main(x)
        y_new_mapping = mapping_model(y_m)
        y_new_mapping = y_new_mapping.sigmoid()
        y_new_prediction = self.representation(torch.cat((x_feature_kd, y_new_mapping), 1))
        return y_new_prediction



class IntegratedModel(nn.Module):
    def __init__(self, hyper_params):
        super(IntegratedModel, self).__init__()
        self.W_m = nn.Sequential(
            nn.Linear(hyper_params.classifier_input_dim, hyper_params.classifier_hidden1),
            nn.Dropout(hyper_params.classifier_dropout),
            nn.ReLU(),
        )
        self.representation = nn.Sequential(
            nn.Linear((hyper_params.classifier_hidden1 + hyper_params.label_mapping_output_dim * 4),
                      hyper_params.label_representation_hidden1),
            torch.nn.Dropout(hyper_params.label_representation_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.label_representation_hidden1, hyper_params.label_representation_output_dim),
        )
        self.mapping_W = nn.Sequential(
            nn.Linear(hyper_params.label_mapping_output_dim, hyper_params.label_mapping_output_dim * 4),
            nn.ReLU(),
        )
    def forward(self, x, soft_y_new):
        x_feature_kd = self.W_m(x)
        soft_y_new = self.mapping_W(soft_y_new)
        y_new_prediction = self.representation(torch.cat((x_feature_kd, soft_y_new), 1))
        return y_new_prediction




class IntegratedDSLL(nn.Module):
    def __init__(self, hyper_params):
        super(IntegratedDSLL, self).__init__()
        self.W_m = nn.Sequential(
            nn.Linear(hyper_params.classifier_input_dim, hyper_params.classifier_hidden1),
            torch.nn.Dropout(hyper_params.classifier_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.classifier_hidden1, hyper_params.classifier_hidden2),
        )
        self.transformation = nn.Sequential(
            nn.Linear(hyper_params.label_mapping_output_dim, hyper_params.label_mapping_output_dim * 4),
        )
        self.seniorStudent = nn.Sequential(
            nn.Linear((hyper_params.classifier_hidden2 + hyper_params.label_mapping_output_dim * 4),
                      hyper_params.label_representation_hidden1),
            nn.ReLU(),
            nn.Linear(hyper_params.label_representation_hidden1, hyper_params.label_representation_output_dim),
        )
    def forward(self, x, y_mapping):
        x_feature_kd = self.W_m(x)
        y_transformation = self.transformation(y_mapping)
        y_new_prediction = self.seniorStudent(torch.cat((x_feature_kd, y_transformation), 1))
        return y_new_prediction


import math
class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class _classifier(nn.Module):
    def __init__(self, hyper_params):
        super(_classifier, self).__init__()
        self.W_m = nn.Sequential(
            nn.Linear(hyper_params.classifier_input_dim, hyper_params.classifier_hidden1),
            torch.nn.Dropout(hyper_params.classifier_dropout),
            nn.ReLU(),
            # GELU(),
            # nn.LeakyReLU(),
            nn.Linear(hyper_params.classifier_hidden1, hyper_params.classifier_output_dim),
        )

    def forward(self, input):
        return self.W_m(input)


class _classifierBatchNorm(nn.Module):
    def __init__(self, hyper_params):
        super(_classifierBatchNorm, self).__init__()
        self.W_m = nn.Sequential(
            nn.Linear(hyper_params.classifier_input_dim, hyper_params.classifier_hidden1),
            nn.BatchNorm1d(hyper_params.classifier_hidden1),
            nn.Dropout(hyper_params.classifier_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.classifier_hidden1, hyper_params.classifier_output_dim),
        )

    def forward(self, input):
        return self.W_m(input)


class _classifier2(nn.Module):
    def __init__(self, hyper_params):
        super(_classifier2, self).__init__()
        self.W_m = nn.Sequential(
            nn.Linear(hyper_params.classifier_input_dim, hyper_params.classifier_hidden1),
            torch.nn.Dropout(hyper_params.classifier_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.classifier_hidden1, hyper_params.classifier_hidden2),
            torch.nn.Dropout(hyper_params.classifier_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.classifier_hidden2, hyper_params.classifier_output_dim),
        )

    def forward(self, input):
        return self.W_m(input)

class _S_label_mapping(nn.Module):
    def __init__(self, hyper_params):
        super(_S_label_mapping,self).__init__()
        self.label_mapping = nn.Sequential(
            nn.Linear(hyper_params.label_mapping_input_dim, hyper_params.label_mapping_hidden1),
            torch.nn.Dropout(hyper_params.label_mapping_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.label_mapping_hidden1, hyper_params.label_mapping_output_dim),
        )

    def forward(self, input):
        return self.label_mapping(input)


class _S_label_mapping2(nn.Module):
    def __init__(self, hyper_params):
        super(_S_label_mapping2, self).__init__()
        self.label_mapping = nn.Sequential(
            nn.Linear(hyper_params.label_mapping_input_dim, hyper_params.label_mapping_hidden1),
            torch.nn.Dropout(hyper_params.label_mapping_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.label_mapping_hidden1, hyper_params.label_mapping_hidden2),
            torch.nn.Dropout(hyper_params.label_mapping_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.label_mapping_hidden2, hyper_params.label_mapping_output_dim),
        )

    def forward(self, input):
        return self.label_mapping(input)


class _S_label_linear_mapping(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_S_label_mapping,self).__init__()
        self.linear_mapping = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, input):
        return self.linear_mapping(input)


class _label_representation(nn.Module):
    def __init__(self, hyper_params):
        super(_label_representation, self).__init__()
        self.label_representation = nn.Sequential(
            nn.Linear(hyper_params.label_representation_input_dim, hyper_params.label_representation_hidden1),
            torch.nn.Dropout(hyper_params.label_representation_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.label_representation_hidden1, hyper_params.label_representation_output_dim),
        )

    def forward(self, input):
        return self.label_representation(input)


class _label_representation2(nn.Module):
    def __init__(self, hyper_params):
        super(_label_representation2, self).__init__()
        self.label_representation = nn.Sequential(
            nn.Linear(hyper_params.label_representation_input_dim, hyper_params.label_representation_hidden1),
            torch.nn.Dropout(hyper_params.label_representation_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.label_representation_hidden1, hyper_params.label_representation_hidden2),
            torch.nn.Dropout(hyper_params.label_representation_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.label_representation_hidden2, hyper_params.label_representation_output_dim),
        )
    def forward(self, input):
        return self.label_representation(input)


class _BP_ML(nn.Module):
    def __init__(self, hyper_params):
        super(_BP_ML, self).__init__()
        self.W_m = nn.Sequential(
            nn.Linear(hyper_params.classifier_input_dim, hyper_params.classifier_hidden1),
            nn.Linear(hyper_params.classifier_hidden1, hyper_params.classifier_output_dim),
        )

    def forward(self, input):
        return self.W_m(input)


class _DNN(nn.Module):
    def __init__(self, hyper_params):
        super(_DNN, self).__init__()
        self.W_m = nn.Sequential(
            nn.Linear(hyper_params.classifier_input_dim, hyper_params.classifier_hidden1),
            torch.nn.Dropout(hyper_params.classifier_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.classifier_hidden1, hyper_params.classifier_output_dim),
        )

    def forward(self, input):
        return self.W_m(input)