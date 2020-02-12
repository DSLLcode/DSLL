# -*- coding:utf-8 -*-

# Deep Streaming Label Learning

from torch.autograd import Variable
import torch
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import recall_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import coverage_error
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss
import sklearn.metrics as metrics


def split_train(data,test_ratio):
    np.random.seed(22)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices], data[test_indices]

def split_label(data, past_label_num_ratio):   # full-label Y data, past_label_ratio
    np.random.seed(95)
    shuffled_indices = np.random.permutation(data.shape[1])
    past_label_indices = shuffled_indices[:int(data.shape[1]*past_label_num_ratio)]
    new_label_indices = shuffled_indices[int(data.shape[1]*past_label_num_ratio)+1:]
    return data[:, past_label_indices], data[:, new_label_indices]

def predictor_accuracy(classifier, x_test, y_test):
    temp0 = 0
    temp1 = 0
    labeltruth0 = 0
    labeltruth1 = 0
    for i, sample in enumerate(x_test):
        inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
        output = classifier(inputv)
        if i == 0:
            pred_y = output
        else:
            pred_y = torch.cat([pred_y,output], 0)
        for j in range(len(output[0])):
            if y_test[i][j] == 0:
                labeltruth0 = labeltruth0 + 1
                if output[0][j] < 0:
                    temp0 = temp0 + 1
            else:
                labeltruth1 = labeltruth1 + 1
                if output[0][j] > 0:
                    temp1 = temp1 + 1
    print('test label 0： ', labeltruth0, '  test label 1： ',labeltruth1)
    print(temp0,'+', temp1, '/', (len(y_test[0]) * len(x_test)))
    return (temp0+temp1) / (len(y_test[0]) * len(x_test)), pred_y.detach().numpy()

def precision_at_ks(true_Y, pred_Y):
    ks = [1, 3, 5]
    result = {}
    true_labels = [set(true_Y[i, :].nonzero()[0]) for i in range(true_Y.shape[0])]
    # true_Y.toarray()[1, :].nonzero()[0]
    label_ranks = np.fliplr(np.argsort(pred_Y, axis=1))
    for k in ks:
        pred_labels = label_ranks[:, :k]
        precs = [len(t.intersection(set(p))) / k
                 for t, p in zip(true_labels, pred_labels)]
        result[k] = round(np.mean(precs), 4)
    return result

def one_error(ground_truth, prediction):
    true_labels = [set(ground_truth[i, :].nonzero()[0]) for i in range(ground_truth.shape[0])]
    label_ranks = np.fliplr(np.argsort(prediction, axis=1))
    pred_labels = label_ranks[:, :1]
    precs = [(1 - len(t.intersection(set(p))))
             for t, p in zip(true_labels, pred_labels)]
    result = np.mean(precs)
    return result

def predict(model, input_X):
    with torch.no_grad():
        model.eval()
        model.cpu()
        if torch.cuda.is_available():
            output_Y = model(Variable(torch.FloatTensor(input_X)))
            # output_Y = model(Variable(torch.FloatTensor(input_X)).cuda())
            model.cuda()
        else:
            output_Y = model(Variable(torch.FloatTensor(input_X)))
        # prediction = np.round(torch.sigmoid(output_Y).cpu().detach().numpy())
        prediction = torch.sigmoid(output_Y).cpu().detach().numpy()
        model.train()

    return prediction

def print_predict(ground_truth, prediction, hyper_params):
    rounded = 4
    AUC_macro = round(roc_auc_score(ground_truth, prediction, average='macro'), rounded)
    AUC_micro = round(roc_auc_score(ground_truth, prediction, average='micro'), rounded)
    Coverage_error = round((coverage_error(ground_truth, prediction)) / ground_truth.shape[1], rounded)
    rankloss = round(label_ranking_loss(ground_truth, prediction), rounded)
    One_error = round(one_error(ground_truth, prediction), rounded)
    Precision_at_ks = precision_at_ks(ground_truth, prediction)
    Log_loss = round(log_loss(ground_truth, prediction), rounded)
    Average_precision_score = round(average_precision_score(ground_truth, prediction), rounded)

    prediction = np.round(prediction)

    F1_Micro = round(f1_score(ground_truth, prediction, average='micro'), rounded)
    Hamming_loss = round(hamming_loss(ground_truth, prediction), rounded)
    Accuracy = round(accuracy_score(ground_truth, prediction), rounded)
    Recall_score_macro = round(recall_score(ground_truth, prediction, average='macro'), rounded)
    Recall_score_micro = round(recall_score(ground_truth, prediction, average='micro'), rounded)
    Precision_score_macro = round(precision_score(ground_truth, prediction, average='macro'), rounded)
    Precision_score_micro = round(precision_score(ground_truth, prediction, average='micro'), rounded)
    Jaccard_score_macro = round(jaccard_score(ground_truth, prediction, average='macro'), rounded)
    Jaccard_score_micro = round(jaccard_score(ground_truth, prediction, average='micro'), rounded)

    print('Recall_score_macro:   ', Recall_score_macro)
    print('Recall_score_micro:   ', Recall_score_micro)
    print('Precision_score_macro:   ', Precision_score_macro)
    print('Precision_score_micro:   ', Precision_score_micro)
    print('Jaccard_score_macro:   ', Jaccard_score_macro)
    print('Jaccard_score_micro:   ', Jaccard_score_micro)
    print("Accuracy = ", Accuracy)
    print('precision_at_ks: ', Precision_at_ks)
    print('Hamming_loss: ', Hamming_loss)
    print('Log_loss:  ', Log_loss)
    print('Average_precision_score: ', Average_precision_score)
    print('F1_Micro ', F1_Micro)
    print('One_error: ', One_error)
    print('Ranking loss: ', rankloss)
    print('coverage: ', Coverage_error)
    print('AUC-micro:   ', AUC_micro)
    print('AUC-macro:   ', AUC_macro)

    print('\n')

def predict_integrated(model, input_X, input_mapping_Y_new):
    model.eval()
    if torch.cuda.is_available():
        output_Y = model(Variable(torch.FloatTensor(input_X)).cuda(), Variable(torch.FloatTensor(input_mapping_Y_new)).cuda())
    else:
        output_Y = model(Variable(torch.FloatTensor(input_X)), Variable(torch.FloatTensor(input_mapping_Y_new)))
    prediction = torch.sigmoid(output_Y).cpu().detach().numpy()
    model.train()
    return prediction


# hook
class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):

        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


def string_rename(old_string, new_string, start, end):
    new_string = old_string[:start] + new_string + old_string[end:]
    return new_string


def modify_model(pretrained_file, model, old_prefix, new_prefix):
    '''
    :param pretrained_file:
    :param model:
    :param old_prefix:
    :param new_prefix:
    :return:
    '''
    pretrained_dict = torch.load(pretrained_file)
    model_dict = model.state_dict()
    state_dict = modify_state_dict(pretrained_dict, model_dict, old_prefix, new_prefix)
    model.load_state_dict(state_dict)
    return model


def modify_state_dict(pretrained_dict, model_dict, old_prefix, new_prefix):
    '''
    model dict
    :param pretrained_dict:
    :param model_dict:
    :param old_prefix:
    :param new_prefix:
    :return:
    '''
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            # state_dict.setdefault(k, v)
            state_dict[k] = v
        else:
            for o, n in zip(old_prefix, new_prefix):
                prefix = k[:len(o)]
                if prefix == o:
                    kk = string_rename(old_string=k, new_string=n, start=0, end=len(o))
                    print("rename layer modules:{}-->{}".format(k, kk))
                    state_dict[kk] = v
    return state_dict


from collections import OrderedDict


def rename_state_dict_keys(source, key_transformation, target=None):
    """
    source             -> Path to the saved state dict.
    key_transformation -> Function that accepts the old key names of the state
                          dict as the only argument and returns the new key name.
    target (optional)  -> Path at which the new state dict should be saved
                          (defaults to `source`)
    Example:
    Rename the key `layer.0.weight` `layer.1.weight` and keep the names of all
    other keys.
    ```py
    def key_transformation(old_key):
        if old_key == "layer.0.weight":
            return "layer.1.weight"
        return old_key
    rename_state_dict_keys(state_dict_path, key_transformation)
    ```
    """
    if target is None:
        target = source

    state_dict = torch.load(source)
    # state_dict = state_dict.state_dict()    
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value

    torch.save(new_state_dict, target)




