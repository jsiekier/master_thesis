import os
import tensorflow as tf
import utils as utils
import math
import operator

def read_lex(data_dir,vocab_name):
    lexicon = []
    vocab_path = os.path.join(data_dir,vocab_name)
    with open(vocab_path, 'r') as rf:
        for line in rf:
            word = line.split()[0]
            lexicon.append(word)
    return lexicon

def myrelu(features):
    return tf.maximum(features, 0.0)

def read_input_data_tweets(data_dir,path_x,path_y):
    data_url = os.path.join(data_dir, path_x)
    data_url_y = os.path.join(data_dir,path_y)
    data_set_y = utils.data_set_y(data_url_y)
    data_set, data_count = utils.data_set(data_url)
    return data_set_y,data_set,data_count

def get_indices(data_list, index):
    return [data_list[i] for i in index]

def calc_mean(res_folds):
    mean=[0.0] * len(res_folds[0])
    for i in range(len(res_folds)):
        for j in range(len(res_folds[i])):
            mean[j]+=res_folds[i][j]
    for j in range(len(mean)):
        mean[j]/=float(len(res_folds))
    return mean
def calc_y_prob(train_set, n_class):
    counter = [0] * n_class
    for i in train_set:
        counter[i]+=1
    res=0
    for i in range(n_class):
        if counter[i]!=0:
            res+=math.log(counter[i]/n_class)
    return res

def sort_desc(unsorted_list):
    return list(reversed(sorted(unsorted_list, key=operator.itemgetter(0))))



def split_train_set(train_set, train_count, train_set_y, n_labs):
    train_set_with_lab = train_set[:n_labs]
    train_set_y = train_set_y[:n_labs]
    train_count_with_lab = train_count[:n_labs]
    train_count_without_lab = train_count[n_labs:]
    train_set_without_lab = train_set[n_labs:]
    train_size_with_lab = len(train_set_with_lab)
    train_size_without_lab = len(train_set_without_lab)

    validation_size_with_lab = int(train_size_with_lab * 0.1)
    validation_size_without_lab = int(train_size_without_lab * 0.1)

    dev_set_with_lab = train_set_with_lab[:validation_size_with_lab]
    dev_set_without_lab = train_set_without_lab[:validation_size_without_lab]

    dev_count_with_lab = train_count_with_lab[:validation_size_with_lab]
    dev_count_without_lab = train_count_without_lab[:validation_size_without_lab]

    dev_set_y = train_set_y[:validation_size_with_lab]
    train_set_with_lab = train_set_with_lab[validation_size_with_lab:]
    train_set_without_lab = train_set_without_lab[validation_size_without_lab:]

    train_count_with_lab = train_count_with_lab[validation_size_with_lab:]
    train_count_without_lab = train_count_without_lab[validation_size_without_lab:]
    train_set_y = train_set_y[validation_size_with_lab:]

    return dev_set_with_lab, dev_set_without_lab, dev_count_with_lab, dev_count_without_lab, dev_set_y, \
           train_set_with_lab, train_set_without_lab, train_count_with_lab, train_count_without_lab, train_set_y
