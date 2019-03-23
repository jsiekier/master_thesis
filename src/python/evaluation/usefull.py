import os
import tensorflow as tf
import utils as utils
import math

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

def merge_sort(unsorted_list):
    if len(unsorted_list) <= 1:
        return unsorted_list
# Find the middle point and devide it
    middle = len(unsorted_list) // 2
    left_list = unsorted_list[:middle]
    right_list = unsorted_list[middle:]

    left_list = merge_sort(left_list)
    right_list = merge_sort(right_list)
    return list(merge(left_list, right_list))

# Merge the sorted halves

def merge(left_half,right_half):

    res = []
    while len(left_half) != 0 and len(right_half) != 0:
        if left_half[0][0] >= right_half[0][0]:
            res.append(left_half[0])
            left_half.remove(left_half[0])
        else:
            res.append(right_half[0])
            right_half.remove(right_half[0])
    if len(left_half) == 0:
        res = res + right_half
    else:
        res = res + left_half
    return res

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
