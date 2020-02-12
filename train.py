# -*- coding:utf-8 -*-

# Deep Streaming Label Learning

from model import _label_representation, _S_label_mapping, _classifier, _classifier2, IntegratedModel,\
    KnowledgeDistillation,  _classifierBatchNorm, _S_label_mapping2, _DNN, _BP_ML, IntegratedDSLL

from helpers import predictor_accuracy, precision_at_ks, predict, predict_integrated, \
    print_predict, LayerActivations, modify_state_dict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    # Returns the function that will be called inside the train loop
    return train_step


def observe_train_DSLL(hyper_params, classifier, training_losses, train_X, mapping_train_Y_new, train_Y_new, test_X,
                        mapping_test_Y_new, test_Y_new):
    print('[%d/%d]Loss: %.3f' % (
        hyper_params.currentEpoch + 1, hyper_params.classifier_epoch, np.mean(training_losses)))
    if ((((hyper_params.currentEpoch + 1) % 10) == 0) | ((hyper_params.currentEpoch + 1)
                                                         == hyper_params.classifier_epoch)):
        print('train performance')
        pred_Y_train = predict_integrated(classifier, train_X, mapping_train_Y_new)
        print_predict(train_Y_new, pred_Y_train, hyper_params)

    if (((hyper_params.currentEpoch + 1) % 5 == 0) | (hyper_params.currentEpoch < 10)):
        print('test performance')
        pred_Y = predict_integrated(classifier, test_X, mapping_test_Y_new)
        print_predict(test_Y_new, pred_Y, hyper_params)


def observe_train(hyper_params, classifier, training_losses, train_X, train_Y, test_X, test_Y):
    print('[%d/%d]Loss: %.3f' % (
        hyper_params.currentEpoch + 1, hyper_params.classifier_epoch, np.mean(training_losses)))
    if ((((hyper_params.currentEpoch + 1) % 10) == 0) | ((hyper_params.currentEpoch + 1)
                                                         == hyper_params.classifier_epoch)):
        print('train performance')
        pred_Y_train = predict(classifier, train_X)
        print_predict(train_Y, pred_Y_train, hyper_params)

    if (((hyper_params.currentEpoch + 1) % 5 == 0) | (hyper_params.currentEpoch < 10)):
        print('test performance')
        pred_Y = predict(classifier, test_X)
        print_predict(test_Y, pred_Y, hyper_params)

def train_KD(hyper_params, train_X, train_Y, test_X, test_Y):
    hyper_params.KD_input_dim = train_X.shape[1]
    hyper_params.KD_output_dim = train_Y.shape[1]

    classifier = KnowledgeDistillation(hyper_params)
    if torch.cuda.is_available():
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), weight_decay=hyper_params.classifier_L2)
    criterion = nn.MSELoss()

    for epoch in range(hyper_params.KD_epoch):
        losses = []
        for i, sample in enumerate(train_X):
            if torch.cuda.is_available():
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1).cuda()
                labelsv = Variable(torch.FloatTensor(train_Y[i])).view(1, -1).cuda()
            else:
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
                labelsv = Variable(torch.FloatTensor(train_Y[i])).view(1, -1)

            classifier.train()
            output = classifier(inputv)
            loss = criterion(output, labelsv)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean().item())
        print('[%d/%d]Distillation Loss: %.3f' % (epoch + 1, hyper_params.KD_epoch, np.mean(losses)))
    print('complete the training')
    return classifier


def train_integrated_model(hyper_params, KD_model, train_X, train_Y, mapping_train_Y_new, train_Y_new,
                                      test_X,  soft_test_Y, mapping_test_Y_new, test_Y_new):
    hyper_params.classifier_input_dim = train_X.shape[1]
    hyper_params.classifier_output_dim = train_Y_new.shape[1]
    hyper_params.classifier_hidden1 = KD_model.state_dict()['W_m.0.weight'].shape[0]
    hyper_params.KD_input_dim = train_X.shape[1]
    hyper_params.kD_output_dim = hyper_params.classifier_hidden1

    classifier_W_m = KD_model
    classifier_W_m_dict = classifier_W_m.state_dict()
    if torch.cuda.is_available():
        integrated_model = IntegratedModel(hyper_params).cuda()
    else:
        integrated_model = IntegratedModel(hyper_params)

    integrated_model_dict = integrated_model.state_dict()

    classifier_W_m_dict = {k: v for k, v in classifier_W_m_dict.items() if k in integrated_model_dict}
    integrated_model_dict.update(classifier_W_m_dict)
    integrated_model.load_state_dict(integrated_model_dict, strict=False)

    # for param in integrated_model.parameters():
    #     param.requires_grad = False
    # mapping_model = torch.load('model/bestModel/10.31experiment/mapping_epoch6_64-00.5soft_0.5hard')   ,'lr': 0.0001

    # optimizer = torch.optim.Adam([
    #     {'params':integrated_model.W_m.parameters(), 'lr': 0.001},
    #     {'params':integrated_model.representation.parameters()},
    #     {'params':integrated_model.mapping_W.parameters()},
    # ], weight_decay=hyper_params.classifier_L2)

    optimizer = optim.Adam(integrated_model.parameters(), weight_decay=hyper_params.classifier_L2)

    criterion = nn.MultiLabelSoftMarginLoss()

    for epoch in range(hyper_params.classifier_epoch):
        hyper_params.currentEpoch = epoch
        losses = []
        for i, sample in enumerate(train_X):
            if torch.cuda.is_available():
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1).cuda()
                labelsv = Variable(torch.FloatTensor(train_Y_new[i])).view(1, -1).cuda()
                mapping_y_new = Variable(torch.FloatTensor(mapping_train_Y_new[i])).view(1, -1).cuda()
            else:
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
                labelsv = Variable(torch.FloatTensor(train_Y_new[i])).view(1, -1)
                mapping_y_new = Variable(torch.FloatTensor(mapping_train_Y_new[i])).view(1, -1)

            integrated_model.train()
            output = integrated_model(inputv, mapping_y_new)
            loss = criterion(output, labelsv) + label_correlation_loss2(output, labelsv)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean().item())
    print('complete the training')
    return integrated_model


def train_classifier(hyper_params, train_X, train_Y, test_X, test_Y):
    hyper_params.classifier_input_dim = train_X.shape[1]
    hyper_params.classifier_output_dim = train_Y.shape[1]

    if hyper_params.classifier_hidden2 == 0:
        classifier = _classifier(hyper_params)
    else:
        classifier = _classifier2(hyper_params)

    if torch.cuda.is_available():
        classifier = classifier.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        classifier = nn.DataParallel(classifier, device_ids=[0, 1])

    optimizer = optim.Adam(classifier.parameters(), weight_decay=hyper_params.classifier_L2)
    criterion = nn.MultiLabelSoftMarginLoss()

    for epoch in range(hyper_params.classifier_epoch):
        losses = []
        classifier.train()

        for i, sample in enumerate(train_X):

            if torch.cuda.is_available():
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1).cuda()
                labelsv = Variable(torch.FloatTensor(train_Y[i])).view(1, -1).cuda()
            else:
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
                labelsv = Variable(torch.FloatTensor(train_Y[i])).view(1, -1)

            output = classifier(inputv)
            loss = criterion(output, labelsv)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean().item())

    print('complete the training')
    return classifier

def lossAdd(x, y):
    loss1 = nn.MultiLabelSoftMarginLoss()
    loss = loss1(x, y) + 0.5 * label_correlation_DIYloss(x, y)
    return loss

def lossAddcorrelation(x, y):
    loss1 = nn.MultiLabelSoftMarginLoss()
    loss = loss1(x, y) + label_correlation_loss2(x, y)
    return loss

def train_classifier_batch(hyper_params, train_X, train_Y, test_X, test_Y, train_loader):
    hyper_params.classifier_input_dim = train_X.shape[1]
    hyper_params.classifier_output_dim = train_Y.shape[1]
    # hyper_params.model_name = 'classifier'
    if hyper_params.batchNorm:
        hyper_params.model_name = 'classifier-BatchNorm'

    if hyper_params.classifier_hidden2 == 0:
        classifier = _classifier(hyper_params)
    else:
        classifier = _classifier2(hyper_params)

    if torch.cuda.is_available():
        classifier = classifier.cuda()
    optimizer = optim.Adam(classifier.parameters(), weight_decay=hyper_params.classifier_L2)
    # optimizer_2 = optim.SGD([{'params': w1, 'lr': 0.1},
    #                          {'params': w2, 'lr': 0.001}])

    if hyper_params.loss == 'entropy':
        criterion = nn.MultiLabelSoftMarginLoss()
    elif hyper_params.loss == 'correlation':
        criterion = label_correlation_loss2
    elif hyper_params.loss == 'correlation_entropy':
        criterion = lossAddcorrelation
    elif hyper_params.loss == 'DIY':
        criterion = DIYloss()
    elif hyper_params.loss == 'DIY_entropy':
        criterion = lossAdd
    else:
        print('please choose loss function (CrossEntropy is default)')
        criterion = nn.MultiLabelSoftMarginLoss()

    train_step = make_train_step(classifier, criterion, optimizer)

    training_losses = []
    # for each epoch
    for epoch in range(hyper_params.classifier_epoch):
        batch_losses = []
        hyper_params.currentEpoch = epoch

        if ((epoch+1) % 20 == 0) & hyper_params.changeloss:
            losses = []
            classifier.train()

            for i, sample in enumerate(train_X):
                if (i+1) % 10 == 0:

                    if torch.cuda.is_available():
                        inputv = Variable(torch.FloatTensor(sample)).view(1, -1).cuda()
                        labelsv = Variable(torch.FloatTensor(train_Y[i])).view(1, -1).cuda()
                    else:
                        inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
                        labelsv = Variable(torch.FloatTensor(train_Y[i])).view(1, -1)

                    output = classifier(inputv)
                    loss = criterion(output, labelsv) + label_correlation_loss2(output, labelsv)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.data.mean().item())

            observe_train(hyper_params, classifier, losses, train_X, train_Y, test_X, test_Y)
            print('\nchange loss:', np.mean(losses))

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(hyper_params.device)
            y_batch = y_batch.to(hyper_params.device)

            loss = train_step(x_batch, y_batch)
            batch_losses.append(loss)
        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)

        observe_train(hyper_params, classifier, training_losses, train_X, train_Y, test_X, test_Y)

    print('complete the training')
    return classifier


def make_train_DSLL(model, loss_fn, optimizer):
    def train_step_DSLL(x, y_mapping, y):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x,y_mapping)

        loss = loss_fn(yhat, y)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step_DSLL


def train_DSLL_model(hyper_params, featureKD_model, train_X, train_Y, mapping_train_Y_new, train_Y_new, test_X, mapping_test_Y_new, test_Y_new, train_loader):
    hyper_params.classifier_input_dim = train_X.shape[1]
    hyper_params.classifier_output_dim = train_Y.shape[1]
    hyper_params.model_name = 'DSLL'
    classifier = IntegratedDSLL(hyper_params)

    classifier_W_m = featureKD_model
    classifier_W_m_dict = classifier_W_m.state_dict()
    classifier_dict = classifier.state_dict()
    classifier_W_m_dict = {k: v for k, v in classifier_W_m_dict.items() if k in classifier_dict}
    classifier_dict.update(classifier_W_m_dict)
    classifier.load_state_dict(classifier_dict, strict=False)

    if torch.cuda.is_available():
        classifier = classifier.cuda()
    # optimizer = optim.Adam(classifier.parameters(), weight_decay=hyper_params.classifier_L2)
    optimizer = torch.optim.Adam([
        {'params':classifier.W_m.parameters()},   # , 'lr': 0.0001},
        {'params':classifier.seniorStudent.parameters()},
        {'params':classifier.transformation.parameters()},
    ], weight_decay=hyper_params.classifier_L2)

    if hyper_params.loss == 'entropy':
        criterion = nn.MultiLabelSoftMarginLoss()
    elif hyper_params.loss == 'correlation':
        criterion = label_correlation_loss2
    elif hyper_params.loss == 'correlation_entropy':
        criterion = lossAddcorrelation
    elif hyper_params.loss == 'DIY':
        criterion = DIYloss()
    elif hyper_params.loss == 'DIY_entropy':
        criterion = lossAdd
    else:
        print('please choose loss function (CrossEntropy is default)')
        criterion = nn.MultiLabelSoftMarginLoss()

    train_step = make_train_DSLL(classifier, criterion, optimizer)

    training_losses = []
    # for each epoch
    for epoch in range(hyper_params.classifier_epoch):
        batch_losses = []
        hyper_params.currentEpoch = epoch

        for x_batch, y_mapping, y_batch in train_loader:
            x_batch = x_batch.to(hyper_params.device)
            y_mapping = y_mapping.to(hyper_params.device)
            y_batch = y_batch.to(hyper_params.device)

            loss = train_step(x_batch, y_mapping, y_batch)
            batch_losses.append(loss)
        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)

        observe_train_DSLL(hyper_params, classifier, training_losses, train_X, mapping_train_Y_new, train_Y_new, test_X,
                           mapping_test_Y_new, test_Y_new)

    print('complete the training')
    return classifier


def train_S_label_mapping(hyper_params, train_Y, train_Y_new, test_Y, test_Y_new):
    hyper_params.label_mapping_input_dim = train_Y.shape[1]
    hyper_params.label_mapping_output_dim = train_Y_new.shape[1]
    title1 = ['train_S_label_mapping', 'input_dim={}, '.format(hyper_params.label_mapping_input_dim),
              'output_dim={}, '.format(hyper_params.label_mapping_output_dim),
              'dropout rate={}, '.format(hyper_params.label_mapping_dropout),
              'hidden1={}, '.format(hyper_params.label_mapping_hidden1),
              'hidden2={}, '.format(hyper_params.label_mapping_hidden2),
              'epoch={}'.format(hyper_params.label_mapping_epoch),  'L2={}'.format(hyper_params.label_mapping_L2)
              ]
    print(title1)
    if hyper_params.label_mapping_hidden2 == 0:
        S_label_mapping = _S_label_mapping(hyper_params)
    else:
        S_label_mapping = _S_label_mapping2(hyper_params)
    if torch.cuda.is_available():
        S_label_mapping = S_label_mapping.cuda()
    optimizer_S = optim.Adam(S_label_mapping.parameters(), weight_decay=hyper_params.label_mapping_L2)
    criterion_S = nn.MultiLabelSoftMarginLoss()
    for epoch in range(hyper_params.label_mapping_epoch):
        losses = []
        for i, sample in enumerate(train_Y):
            if torch.cuda.is_available():
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1).cuda()
                labelsv = Variable(torch.FloatTensor(train_Y_new[i])).view(1, -1).cuda()
            else:
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
                labelsv = Variable(torch.FloatTensor(train_Y_new[i])).view(1, -1)

            output = S_label_mapping(inputv)
            # output = output.sigmoid().round()
            if hyper_params.loss == 'correlation_aware':
                loss = criterion_S(output, labelsv) + label_correlation_loss2(output, labelsv)
            else:
                loss = criterion_S(output, labelsv)

            optimizer_S.zero_grad()
            loss.backward()
            optimizer_S.step()
            losses.append(loss.data.mean().item())
        print('S (label mapping) [%d/%d] Loss: %.3f' % (epoch + 1, hyper_params.label_mapping_epoch, np.mean(losses)))
    print('complete the label mapping')
    return S_label_mapping


def train_label_representation(hyper_params, train_X, mapping_soft_train_Y_new, train_Y_new, test_X, mapping_soft_test_Y_new, test_Y_new):
    hyper_params.label_representation_input_dim = train_X.shape[1] + mapping_soft_train_Y_new.shape[1]
    hyper_params.label_representation_output_dim = train_Y_new.shape[1]
    label_representation = _label_representation(hyper_params)
    if torch.cuda.is_available():
        label_representation = label_representation.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        label_representation = nn.DataParallel(label_representation, device_ids=[0, 1])


    optimizer_label_repre = optim.Adam(label_representation.parameters(),
                                       weight_decay=hyper_params.label_representation_L2)
    criterion_label_repre = nn.MultiLabelSoftMarginLoss()  # nn.MultiLabelSoftMarginLoss()
    train_input = np.hstack((train_X, mapping_soft_train_Y_new))
    train_output_true = train_Y_new
    test_input = np.hstack((test_X, mapping_soft_test_Y_new))
    test_output_true = test_Y_new
    for epoch in range(hyper_params.label_representation_epoch):
        losses = []
        for i, sample in enumerate(train_input):
            if torch.cuda.is_available():
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1).cuda()
                labelsv = Variable(torch.FloatTensor(train_output_true[i])).view(1, -1).cuda()
            else:
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
                labelsv = Variable(torch.FloatTensor(train_output_true[i])).view(1, -1)

            output = label_representation(inputv)
            loss = criterion_label_repre(output, labelsv)

            optimizer_label_repre.zero_grad()
            loss.backward()
            optimizer_label_repre.step()
            losses.append(loss.data.mean().item())
    print('complete the label representation')
    return label_representation


class Label_Correlation_Loss(nn.Module):
    def __init__(self):
        super(Label_Correlation_Loss, self).__init__()

    def forward(self,pred_Y, true_Y):
        return label_correlation_loss(pred_Y, true_Y)

# Label correlation aware loss function
class DIYloss(nn.Module):
    def __init__(self):
        super(DIYloss, self).__init__()
        return
    def forward(self, pred_Y, true_Y):
        mseLoss = nn.MSELoss()
        pred_Y = torch.sigmoid(pred_Y)
        n_one_true = int(torch.sum(true_Y))
        n_zero_true = true_Y.shape[1] - n_one_true
        nonzero_index = torch.nonzero(true_Y[0] > 0).reshape(-1)
        zero_index = torch.nonzero(true_Y[0] == 0).reshape(
            -1)
        Ei = 0
        if n_one_true == 0:
            Ei = (pred_Y[0] ** 2).mean()
        else:
            for k in range(n_one_true):
                for l in range(n_zero_true):
                    Ei = mseLoss((1 + pred_Y[0][zero_index[l]]),
                                 pred_Y[0][nonzero_index[k]]) + Ei
            Ei = 1 / (n_one_true * n_zero_true) * Ei
        return Ei

        return loss


def label_correlation_loss(pred_Y, true_Y):
    pred_Y = torch.sigmoid(pred_Y)
    n_one_true = int(torch.sum(true_Y))
    n_zero_true = true_Y.shape[1] - n_one_true
    nonzero_index = torch.nonzero(true_Y[0] > 0).reshape(-1)
    zero_index = torch.nonzero(true_Y[0] == 0).reshape(-1)
    Ei = 0
    if n_one_true == 0:
        for l in range(n_zero_true):
            Ei = torch.exp( pred_Y[0][zero_index[0][l]]) + Ei
        Ei = 1 / (n_zero_true) * Ei
    else:
        for k in range(n_one_true):
            for l in range(n_zero_true):
                Ei = torch.exp(-(pred_Y[0][nonzero_index[0][k]] - pred_Y[0][zero_index[0][l]])) + Ei
        Ei = 1 / (n_one_true * n_zero_true) * Ei
    return Ei

def label_correlation_loss2(pred_Y, true_Y):
    pred_Y = torch.sigmoid(pred_Y)
    n_one_true = int(torch.sum(true_Y))
    n_zero_true = true_Y.shape[1] - n_one_true
    nonzero_index = torch.nonzero(true_Y[0] > 0).reshape(-1)
    zero_index = torch.nonzero(true_Y[0] == 0).reshape(-1)
    Ei = 0
    if n_one_true == 0:
        for l in range(n_zero_true):
            Ei = torch.exp(pred_Y[0][zero_index[l]] - 1) + Ei
        Ei = 1 / (n_zero_true) * Ei
    elif n_zero_true == 0:
        for l in range(n_one_true):
            Ei = torch.exp(-pred_Y[0][nonzero_index[l]]) + Ei
        Ei = 1 / (n_one_true) * Ei
    else:
        for k in range(n_one_true):
            for l in range(n_zero_true):
                Ei = torch.exp(-(pred_Y[0][nonzero_index[k]] - pred_Y[0][zero_index[l]])) + Ei
        Ei = 1 / (n_one_true * n_zero_true) * Ei
    return Ei

def label_correlation_loss2_old(pred_Y, true_Y):
    pred_Y = torch.sigmoid(pred_Y)
    n_one_true = int(torch.sum(true_Y))
    n_zero_true = true_Y.shape[1] - n_one_true
    nonzero_index = torch.where(true_Y > 0)# nonzero_index = torch.nonzero(true_Y)
    zero_index = torch.where(true_Y == 0)
    Ei = 0
    if n_one_true == 0:
        for l in range(n_zero_true):
            Ei = torch.exp(pred_Y[0][zero_index[1][l]] - 1) + Ei
        Ei = 1 / (n_zero_true) * Ei
    else:
        for k in range(n_one_true):
            for l in range(n_zero_true):
                Ei = torch.exp(-(pred_Y[0][nonzero_index[1][k]] - pred_Y[0][zero_index[1][l]])) + Ei
        Ei = 1 / (n_one_true * n_zero_true) * Ei
    return Ei


# Label correlation aware loss function
def label_correlation_DIYloss(pred_Y, true_Y):
    mseLoss = nn.MSELoss()
    pred_Y = torch.sigmoid(pred_Y)
    n_one_true = int(torch.sum(true_Y))
    n_zero_true = true_Y.shape[1] - n_one_true
    nonzero_index = torch.nonzero(true_Y[0] > 0).reshape(-1)
    zero_index = torch.nonzero(true_Y[0] == 0).reshape(-1)
    Ei = 0
    if n_one_true == 0:
        Ei = (pred_Y[0] ** 2).mean()
    elif n_zero_true == 0:
        Ei = ((pred_Y[0]-1) ** 2).mean()
    else:
        for k in range(n_one_true):
            for l in range(n_zero_true):
                Ei = mseLoss((1 + pred_Y[0][zero_index[l]]), pred_Y[0][nonzero_index[k]]) + Ei
        Ei = 1 / (n_one_true * n_zero_true) * Ei
    return Ei


def label_correlation_DIYloss_old(pred_Y, true_Y):
    mseLoss = nn.MSELoss(reduction='sum')
    pred_Y = torch.sigmoid(pred_Y)
    n_one_true = int(torch.sum(true_Y))
    n_zero_true = true_Y.shape[1] - n_one_true
    nonzero_index = torch.where(true_Y > 0) # nonzero_index = torch.nonzero(true_Y)
    zero_index = torch.where(true_Y == 0)
    Ei = 0
    print(nonzero_index)
    if n_one_true == 0:
        for l in range(n_zero_true):
            Ei = torch.pow(pred_Y[0][zero_index[1][l]], 2) + Ei
        Ei = 1 / (n_zero_true) * Ei
    else:
        for k in range(n_one_true):
            for l in range(n_zero_true):
                Ei = mseLoss((1 + pred_Y[0][zero_index[1][l]]), pred_Y[0][nonzero_index[1][k]]) + Ei
        Ei = 1 / (n_one_true * n_zero_true) * Ei
    # print(n_one_true * n_zero_true)
    return Ei


def label_correlation_loss_batch(pred_Y, true_Y):
    Ei = 0
    for i in range(true_Y.shape[0]):
        n_one_true = int(torch.sum(true_Y[i]))
        n_zero_true = true_Y.shape[1] - n_one_true
        nonzero_index = torch.nonzero(true_Y[i] > 0)
        zero_index = torch.nonzero(true_Y[i] == 0)
        temp = 0
        for k in range(n_one_true):
            for l in range(n_zero_true):
                temp = torch.exp(-(pred_Y[i][nonzero_index[0][k]] - pred_Y[i][zero_index[0][l]])) +temp
        Ei = 1/(n_one_true * n_zero_true) * temp + Ei
    return Ei
