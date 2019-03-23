import tensorflow as tf
import random
import numpy as np
import math


def data_set_y(data_url):
    data = []
    fin = open(data_url)
    while True:
        line = fin.readline()
        if not line:
            break
        data.append(int(line))
    fin.close()
    return data


def data_set(data_url):
    """process data input."""
    data = []
    word_count = []
    fin = open(data_url)
    while True:
        line = fin.readline()
        if not line:
            break
        id_freqs = line.split()
        doc = {}
        count = 0
        for id_freq in id_freqs[1:]:
            items = id_freq.split(':')
            # python starts from 0
            if int(items[0]) - 1 < 0:
                print('WARNING INDICES!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            doc[int(items[0]) - 1] = int(items[1])
            count += int(items[1])
        if count > 0:
            data.append(doc)
            word_count.append(count)
    fin.close()
    return data, word_count

def create_batches_new(len_labeled, len_unlabeled, batch_size, shuffle):
    batches_labeled = []
    batches_unlabeled=[]
    ids_labeled = list(range(len_labeled))
    ids_unlabeled=list(range(len_unlabeled))

    if len_labeled>len_unlabeled:
        rest=len_labeled%len_unlabeled
        mul=len_labeled//len_unlabeled
        ids_unlabeled=ids_unlabeled*mul+ids_unlabeled[:rest]
    elif len_labeled<len_unlabeled:
        rest=len_unlabeled%len_labeled
        mul=len_unlabeled//len_labeled
        ids_labeled=ids_labeled*mul+ids_labeled[:rest]

    if shuffle:
        random.shuffle(ids_labeled)
        random.shuffle(ids_unlabeled)
    max_len=np.maximum(len_labeled,len_unlabeled)
    for i in range(int(max_len / batch_size)):
        start = i * batch_size
        end = (i + 1) * batch_size
        batches_labeled.append(ids_labeled[start:end])
        batches_unlabeled.append(ids_unlabeled[start:end])

    # the batch of which the length is less than batch_size
    rest = max_len % batch_size
    if rest > 0:
        batches_labeled.append(list(ids_labeled[-rest:]) + [-1] * (batch_size - rest))  # -1 as padding
        batches_unlabeled.append(list(ids_unlabeled[-rest:]) + [-1] * (batch_size - rest))  # -1 as padding
    return batches_labeled,batches_unlabeled

def create_batches(data_size, batch_size, shuffle=True):
    """create index by batches."""
    batches = []
    ids = list(range(data_size))
    if shuffle:
        random.shuffle(ids)
    for i in range(int(data_size / batch_size)):
        start = i * batch_size
        end = (i + 1) * batch_size
        batches.append(ids[start:end])
    # the batch of which the length is less than batch_size
    rest = data_size % batch_size
    if rest > 0:
        batches.append(list(ids[-rest:]) + [-1] * (batch_size - rest))  # -1 as padding
    return batches


def create_batches_without_lab(data_size, batch_size,n_clss, shuffle=True):
    """create index by batches."""

    batches = []
    ids = list(range(data_size))
    div=math.floor(batch_size / n_clss)
    if shuffle:
        random.shuffle(ids)
    for i in range(int(data_size / div)):
        start = i * div
        end = (i + 1) * div
        batches.append(ids[start:end])
    # the batch of which the length is less than batch_size
    rest = data_size % div
    if rest > 0:
        batches.append(list(ids[-rest:]) + [-1] * (div - rest))  # -1 as padding
    return batches


def fetch_data_y(data,  idx_batch,class_size):
    """fetch input data by batch."""
    batch_size = len(idx_batch)
    data_batch = np.zeros((batch_size, class_size))
    for i, doc_id in enumerate(idx_batch):#for all documents
        if doc_id != -1:
            data_batch[i, data[doc_id]] = 1.0

    return data_batch#,count_batch, #mask
def fetch_data_y_new(data,  idx_batch,class_size):
    batch_size = len(idx_batch)
    data_batch = np.zeros((batch_size, class_size))
    data_batch_y_neg, data_batch_y_pos=np.zeros((batch_size, class_size)),np.zeros((batch_size, class_size))
    for i, doc_id in enumerate(idx_batch):  # for all documents
        if doc_id != -1:
            data_batch[i, data[doc_id]] = 1.0
            data_batch_y_neg[i,0]=1.0
            data_batch_y_pos[i, 1] = 1.0
    return data_batch,data_batch_y_neg,data_batch_y_pos

def fetch_data_y_dummy_new(data, class_size):
    batch_size = len(data)
    data_batch = np.zeros((batch_size, class_size))
    data_batch_y_neg, data_batch_y_pos = np.zeros((batch_size, class_size)), np.zeros((batch_size, class_size))
    return data_batch, data_batch_y_neg, data_batch_y_pos



def fetch_data_y_dummy( idx_batch, n_clss,clss_idx):
    y_batch=np.zeros((len(idx_batch), n_clss))
    for i in range(len(idx_batch)):
        y_batch[i,clss_idx]=1.0
    return y_batch


def fetch_data_new(data, idx_batch, vocab_size):
    batch_size = len(idx_batch)
    data_batch = np.zeros((batch_size, vocab_size))
    mask = np.zeros(batch_size)
    for i, doc_id in enumerate(idx_batch):
        if doc_id != -1:
            for word_id, freq in data[doc_id].items():
                data_batch[i, word_id] = freq
            mask[i] = 1.0
    return data_batch, mask

def fetch_data(data, count, idx_batch, vocab_size):
    """fetch input data by batch."""
    batch_size = len(idx_batch)
    data_batch = np.zeros((batch_size, vocab_size))
    count_batch = []
    mask = np.zeros(batch_size)
    indices = []
    values = []
    for i, doc_id in enumerate(idx_batch):
        if doc_id != -1:
            for word_id, freq in data[doc_id].items():
                data_batch[i, word_id] = freq
            count_batch.append(count[doc_id])
            mask[i] = 1.0
        else:
            count_batch.append(0)
    return data_batch, count_batch, mask


def fetch_data_without_lab(data, count, idx_batch, vocab_size,n_clss):
    """fetch input data by batch."""
    batch_size = len(idx_batch*n_clss)
    data_batch_y = np.zeros((batch_size, n_clss))
    data_batch = np.zeros((batch_size, vocab_size))
    count_batch = []
    mask = np.zeros(batch_size)
    indices = []
    values = []
    for i, doc_id in enumerate(idx_batch):
        if doc_id != -1:
            for j in range(n_clss):
                for word_id, freq in data[doc_id].items():
                    data_batch[i*n_clss+j, word_id] = freq
                count_batch.append(count[doc_id])
                data_batch_y[i*n_clss+j,j]=1.0
                mask[i] = 1.0

        else:
            count_batch.append(0)
    return data_batch, count_batch, mask,data_batch_y


def variable_parser(var_list, prefix):
    """return a subset of the all_variables by prefix."""
    ret_list = []
    for var in var_list:
        varname = var.name
        varprefix = varname.split('/')[0]
        if varprefix == prefix:
            ret_list.append(var)
        elif prefix in varname:
            ret_list.append(var)
    return ret_list


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32),high


def linear_LDA(inputs,
               output_size,
               no_bias=False,
               bias_start_zero=False,
               matrix_start_zero=False,
               scope=None):
    """Define a linear connection."""
    with tf.variable_scope(scope or 'Linear'):
        if matrix_start_zero:
            matrix_initializer = tf.constant_initializer(
                0)  # (tf.truncated_normal_initializer()+1.0)/100.#tf.constant_initializer(0)
        else:
            matrix_initializer = None  # tf.truncated_normal_initializer(mean = 0.0, stddev=0.01)#None
        if bias_start_zero:
            bias_initializer = tf.constant_initializer(0)
        else:
            bias_initializer = None  # tf.constant_initializer(0.1)#tf.truncated_normal_initializer(mean = 0.0 , stddev=0.01)#None
        input_size = inputs.get_shape()[1].value
        # tf.Variable(xavier_init(input_size, output_size))
        matrix = tf.nn.softmax(tf.contrib.layers.batch_norm(tf.Variable(xavier_init(input_size, output_size))))
        # tf.get_variable('Matrix', [input_size, output_size],initializer=matrix_initializer)
        # matrix = tf.Print(matrix,[matrix[0]],summarize=50,message='phi')

        output = tf.matmul(inputs, matrix)  # no softmax on input, it should already be normalized
        if not no_bias:
            bias_term = tf.get_variable('Bias', [output_size],
                                        initializer=bias_initializer)
            output = output + bias_term
    return output


def linear(inputs,
           output_size,
           no_bias=False,
           bias_start_zero=False,
           matrix_start_zero=False,
           scope=None):
    """Define a linear connection."""
    with tf.variable_scope(scope or 'Linear'):
        if matrix_start_zero:
            matrix_initializer = tf.constant_initializer(
                0)  # (tf.truncated_normal_initializer()+1.0)/100.#tf.constant_initializer(0)
        else:
            matrix_initializer = None  # tf.truncated_normal_initializer(mean = 0.0, stddev=0.01)#None
        if bias_start_zero:
            bias_initializer = tf.constant_initializer(0)
        else:
            bias_initializer = None  # tf.constant_initializer(0.1)#tf.truncated_normal_initializer(mean = 0.0 , stddev=0.01)#None
        input_size = inputs.get_shape()[1].value
        matrix = tf.get_variable('Matrix', [input_size, output_size], initializer=matrix_initializer)
        #matrix_init,s=xavier_init(input_size, output_size)
        #matrix = tf.Variable(matrix_init,name='Matrix',dtype=tf.float32)
        # matrix = tf.nn.softmax(tf.contrib.layers.batch_norm(tf.Variable(xavier_init(input_size, output_size))))
        #matrix=tf.get_variable('Matrix', [input_size, output_size], initializer=matrix_init)
        output = tf.matmul(inputs, matrix)
        if not no_bias:
            bias_term = tf.get_variable('Bias', [output_size],initializer=bias_initializer)
            output = output + bias_term
    return output


def mlp(inputs,
        mlp_hidden=[],
        mlp_nonlinearity=tf.nn.tanh,
        scope=None,is_training=False):
    """Define an MLP."""
    with tf.variable_scope(scope or 'Linear'):
        mlp_layer = len(mlp_hidden)
        res = inputs
        for l in range(mlp_layer):
            res = tf.contrib.layers.batch_norm( mlp_nonlinearity(linear(res, mlp_hidden[l], scope='l' + str(l))), is_training=is_training)
        return res


def fetch_data_w_c(data, idx_batch):
    """fetch input data by batch."""
    batch_size = len(idx_batch)
    data_batch = []
    mask = np.zeros(batch_size)
    shape=data[0].shape
    for i, doc_id in enumerate(idx_batch):

        if doc_id != -1:
            data_batch.append(data[doc_id])
            mask[i] = 1.0
        else:
            data_batch.append(np.zeros(shape))

    return np.asarray(data_batch), mask
def fetch_data_without_idx_new(to_label, vocab_size):
    batch_size = len(to_label)
    data_batch = np.zeros((batch_size, vocab_size))
    mask = np.zeros(batch_size)
    for i in range(batch_size):
        for word_id, freq in to_label[i].items():
            data_batch[i, word_id] = freq
        mask[i] = 1.0

    return data_batch,mask


def fetch_data_without_idx(train_set_without_lab, count, vocab_size):
    """fetch input data by batch."""
    batch_size = len(train_set_without_lab)
    data_batch = np.zeros((batch_size, vocab_size))
    count_batch = []
    mask = np.zeros(batch_size)
    for i in range(batch_size):

        for word_id, freq in train_set_without_lab[i].items():
            data_batch[i, word_id] = freq
        count_batch.append(count[i])
        mask[i] = 1.0

    return data_batch, count_batch, mask



