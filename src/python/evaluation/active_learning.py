import random
import os
from sklearn.model_selection import KFold as fold
import argparse
import tensorflow as tf
import utils as utils
import sys
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import  usefull as use
import numpy as np
import pandas as pd
import pickle
import v1
import v2
import v3


def parseArgs():
    argparser = argparse.ArgumentParser()
    # define arguments
    argparser.add_argument('--docs_to_add', default=100, type=int)
    argparser.add_argument('--adam_beta1', default=0.9, type=float)
    argparser.add_argument('--adam_beta2', default=0.999, type=float)
    argparser.add_argument('--dir_prior', default=0.1, type=float)
    argparser.add_argument('--n_topic', default=75, type=int)
    argparser.add_argument('--learning_rate', default=1e-3, type=float)
    argparser.add_argument('--data_dir', default="data",type=str)
    argparser.add_argument('--batch_size', default=50, type=int)
    argparser.add_argument('--n_hidden', default="512,256", type=str)
    argparser.add_argument('--warm_up_period', default=100, type=int)
    argparser.add_argument('--non_linearity', default='relu', type=str)  # 'tanh',"lrelu"
    argparser.add_argument('--uncertainty_strategy', default='entropy', type=str)  # variation, bald, entropy
    argparser.add_argument('--n_dropout_rounds', default=1, type=int)
    argparser.add_argument('--use_density_mul', default=0, type=int)
    argparser.add_argument('--random_init_docs', default=1, type=int)
    argparser.add_argument('--density_file', default='density_fold_old_', type=str)
    argparser.add_argument('--max_learning_it', default=300, type=int)
    argparser.add_argument('--no_improvement_it', default=1, type=int)
    argparser.add_argument('--random_selection', default=0, type=int)  # choose active learning or random
    argparser.add_argument('--semi_supervised', default=1, type=int)
    argparser.add_argument('--job_data', default="00", type=str)# first num: fold_num second: seed
    argparser.add_argument('--VAE_version', default="V1", type=str)#'V1','V2','V3'
    argparser.add_argument('--representativeness', default="random", type=str)#"density","submod","PU","random"
    argparser.add_argument('--sub_set_size', default=100, type=int)
    argparser.add_argument('--out_file', default="random.pkl", type=str)


    return argparser.parse_args()  # argstring.split()


def split_train_set(train_set_d, train_set_y_d, label_keys, label_keys_new):
    label_keys = list(set(label_keys).union(label_keys_new))
    unlabeled_keys = list(set(range(len(train_set_d))).difference(set(label_keys)))
    unlabeled_keys.sort()
    label_keys.sort()

    train_set_with_lab = use.get_indices(train_set_d, label_keys)
    train_set_y = use.get_indices(train_set_y_d, label_keys)
    train_set_without_labx = use.get_indices(train_set_d, unlabeled_keys)

    train_size_with_lab = len(train_set_with_lab)
    train_size_without_lab = len(train_set_without_labx)

    validation_size_with_lab = int(train_size_with_lab * 0.1)
    validation_size_without_lab = int(train_size_without_lab * 0.1)

    dev_set_with_lab = train_set_with_lab[:validation_size_with_lab]
    dev_set_without_lab = train_set_without_labx[:validation_size_without_lab]

    dev_set_y = train_set_y[:validation_size_with_lab]
    train_set_with_lab = train_set_with_lab[validation_size_with_lab:]
    train_set_without_lab = train_set_without_labx[validation_size_without_lab:]

    train_set_y = train_set_y[validation_size_with_lab:]

    return dev_set_with_lab, dev_set_without_lab, dev_set_y, \
           train_set_with_lab, train_set_without_lab,  \
           train_set_y, label_keys, unlabeled_keys, train_set_without_labx


def calc_avg(prediction_unlabeled_data):
    avg = []
    T = len(prediction_unlabeled_data)
    n_samples = len(prediction_unlabeled_data[0][0])
    n_class = len(prediction_unlabeled_data[0][0][0])

    for i in range(n_samples):
        prediction = []
        for j in range(n_class):
            summary = 0
            for k in range(T):
                summary += prediction_unlabeled_data[k][0][i][j]
            prediction.append(summary / float(T))
        avg.append(prediction)
    return avg


def calc_entropy(prediction_unlabeled_data):
    avg_pred = calc_avg(prediction_unlabeled_data)
    entropy_arr = []
    for i in range(len(avg_pred)):
        prediction = avg_pred[i]
        entropy = 0.0
        for prop_y in prediction:
            entropy += prop_y * np.log(prop_y)
        entropy *= -1.0
        entropy_arr.append(entropy)
    return entropy_arr


def calc_variation(prediction_unlabeled_data):
    variation_arr = []
    avg_pred = calc_avg(prediction_unlabeled_data)
    for i in range(len(avg_pred)):
        prediction = avg_pred[i]
        var = 1.0 - max(prediction)
        variation_arr.append(var)
    return variation_arr


def calc_bald(prediction_unlabeled_data):
    blast_arr = []
    T = len(prediction_unlabeled_data)
    clss_len = len(prediction_unlabeled_data[0][0][0])
    for j in range(len(prediction_unlabeled_data[0][0])):
        first_part = 0
        second_part = 0
        for clss in range(clss_len):
            p_c = 0
            for pred in range(T):
                prediction = prediction_unlabeled_data[pred][0][j][clss]
                if prediction != 0:
                    second_part += (prediction * np.log(prediction))
                    p_c += prediction
            p_c /= float(T)
            first_part += (p_c * np.log(p_c))
        blast = -first_part + second_part / float(T)
        blast_arr.append(blast)
    return blast_arr


def cals_docs_to_add(prediction_unlabeled_data, doc_density, docs_to_add, unlabeled_keys, uncertainty_strategy,sub_set_size=700):
    disagree = []
    new_train = []
    unlabeled_keys.sort()

    uncertainty_arr = uncertainty_strategy(prediction_unlabeled_data)
    assert len(uncertainty_arr) == len(unlabeled_keys)
    for i in range(len(unlabeled_keys)):
        doc_id = unlabeled_keys[i]

        disagree.append((uncertainty_arr[i], doc_id))
    # sort
    sorted_list = use.sort_desc(disagree)
    # select best docs:
    for i in range(docs_to_add):
        new_train.append(sorted_list[i][1])
    new_train.sort()
    return new_train


def cals_docs_to_add_density(prediction_unlabeled_data, doc_density, docs_to_add, unlabeled_keys, uncertainty_strategy
                             ,sub_set_size=700):
    disagree = []
    new_train = []
    unlabeled_keys.sort()

    uncertainty_arr = uncertainty_strategy(prediction_unlabeled_data)
    for i in range(len(unlabeled_keys)):
        doc_id = unlabeled_keys[i]

        disagree.append((uncertainty_arr[i], doc_id))
    # sort by disagreement
    sorted_list = use.sort_desc(disagree)

    disagree2 = []
    if len(unlabeled_keys) > sub_set_size:
        for i in range(sub_set_size):
            doc_id = sorted_list[i][1]
            disagree2.append((-doc_density[doc_id], doc_id))

        # sort by density
        sorted_list2 = use.sort_desc(disagree2)

        # select best docs:
        for i in range(docs_to_add):
            new_train.append(sorted_list2[i][1])
        return new_train
    else:
        # select best docs:
        for i in range(docs_to_add):
            new_train.append(sorted_list[i][1])
        return new_train


def cals_docs_to_add_density_mul(prediction_unlabeled_data, doc_density, docs_to_add, unlabeled_keys,
                                 uncertainty_strategy,sub_set_size=700):
    disagree = []
    new_train = []
    unlabeled_keys.sort()

    uncertainty_arr = uncertainty_strategy(prediction_unlabeled_data)
    for i in range(len(unlabeled_keys)):
        doc_id = unlabeled_keys[i]

        disagree.append((uncertainty_arr[i] * doc_density[doc_id], doc_id))
    # sort by disagreement * density
    sorted_list = use.sort_desc(disagree)
    # select best docs:
    for i in range(docs_to_add):
        new_train.append(sorted_list[i][1])
    return new_train


def calc_best_density(key_arr, init_add_num, doc_density):
    dens_arr = []
    output = []
    for i in key_arr:
        dens_arr.append((doc_density[i], i))
    sorted_list = use.sort_desc(dens_arr)
    for i in range(init_add_num):
        output.append(sorted_list[i][1])
    return output


def get_y_labs(prediction_unlabeled_data, label_keys_sub, unlabeled_keys):
    unlabeled_keys.sort()
    rel_idx = []
    labs = set(label_keys_sub)
    for c, key in enumerate(unlabeled_keys):
        if key in labs:
            rel_idx.append(c)
    y_lab = []
    for i in rel_idx:
        y_lab.append(np.argmax(prediction_unlabeled_data[0][0][i]))

    return y_lab


def get_indices(train_count_without_labx, label_keys_sub, unlabeled_keys):
    unlabeled_keys.sort()
    rel_idx = []
    labs = set(label_keys_sub)
    for c, key in enumerate(unlabeled_keys):
        if key in labs:
            rel_idx.append(c)
    return [train_count_without_labx[i] for i in rel_idx]


def active_learning(train_set_d, train_set_y_d,
                    test_set, test_set_y, docs_to_add,
                    doc_density, num_iterations, n_class, args, job_seed, vocab_size, non_linearity, lexicon,
                    init_add=50, doc_keys=None):
    if args.uncertainty_strategy == "entropy":
        uncertainty_strategy = calc_entropy
    elif args.uncertainty_strategy == "variation":
        uncertainty_strategy = calc_variation
    else:
        uncertainty_strategy = calc_bald
    if args.representativeness=="density":
        if args.use_density_mul:
                select_docs = cals_docs_to_add_density_mul
        else:
            select_docs = cals_docs_to_add_density
    else:
        select_docs = cals_docs_to_add
    if args.VAE_version=="V1":
        nvdm_dirichlet=v1
    elif args.VAE_version == "V2":
        nvdm_dirichlet=v2
    else:
        nvdm_dirichlet=v3

    if not doc_keys:
        label_keys = random.sample(range(0, len(train_set_y_d)), init_add)
    else:
        label_keys = doc_keys

    result_dict = {}
    result_f1 = []
    num_docs = []
    test_predictions, num_pos, num_neg = [], [], []
    label_keys_new = []
    for i in range(num_iterations):

        dev_set_with_lab, dev_set_without_lab,dev_set_y, \
        train_set_with_lab, train_set_without_lab,  \
        train_set_y, label_keys, unlabeled_keys, \
        train_set_without_labx= split_train_set(train_set_d,  train_set_y_d,
                                                                           label_keys, label_keys_new)
        np.random.seed(job_seed)
        tf.set_random_seed(job_seed)
        random.seed(job_seed)
        number_pos = float(sum(train_set_y) + sum(dev_set_y)) / float(len(train_set_y) + len(dev_set_y))
        num_pos.append(number_pos)
        num_neg.append(1.0 - number_pos)
        mlp_arr = [int(c) for c in args.n_hidden.split(",")]
        nvdm = nvdm_dirichlet.NVDM(vocab_size=vocab_size,
                                         mlp_arr=mlp_arr,
                                         n_topic=args.n_topic,
                                         learning_rate=args.learning_rate,
                                         batch_size=args.batch_size,
                                         non_linearity=non_linearity,
                                         adam_beta1=args.adam_beta1,
                                         adam_beta2=args.adam_beta2,
                                         dir_prior=args.dir_prior,
                                         n_class=n_class,
                                         N=len(train_set_without_lab) + len(train_set_with_lab),
                                         seed=job_seed)

        f1_mes, prediction_unlabeled_data,test_pred = nvdm.train_x(dev_set_with_lab,
                                                         dev_set_without_lab,
                                                         dev_set_y,
                                                         train_set_with_lab,
                                                         train_set_without_lab,
                                                         train_set_y,
                                                         test_set,
                                                         test_set_y,
                                                         train_set_without_labx,
                                                         model_name="model" + args.job_data,
                                                         n_dropout_rounds=args.n_dropout_rounds,
                                                         max_learning_iterations=args.max_learning_it,
                                                         no_improvement_iterations=args.no_improvement_it,
                                                         semi_supervised=args.semi_supervised,
                                                         debug=False,it=i)

        result_f1.append(f1_mes)
        test_predictions.append(test_pred)
        num_docs.append(len(label_keys))

        if len(unlabeled_keys) >= docs_to_add:
            if args.random_selection:
                label_keys_new = random.sample(unlabeled_keys, docs_to_add)
            else:
                if len(unlabeled_keys) > args.sub_set_size:
                    label_keys_sub = select_docs(prediction_unlabeled_data, doc_density, args.sub_set_size,
                                                 unlabeled_keys, uncertainty_strategy,args.sub_set_size)
                else:
                    label_keys_sub = select_docs(prediction_unlabeled_data, doc_density, len(unlabeled_keys),
                                                 unlabeled_keys, uncertainty_strategy,args.sub_set_size)
                all_data_x = train_set_with_lab + dev_set_with_lab + train_set_without_labx
                if args.sub_set_size==docs_to_add:
                    label_keys_new = label_keys_sub
                elif args.representativeness=="submod":
                    y_labels = get_y_labs(prediction_unlabeled_data, label_keys_sub, unlabeled_keys)


                    all_data_y=train_set_y+dev_set_y+get_y_labs(prediction_unlabeled_data,unlabeled_keys,unlabeled_keys)
                    pseudo_keys=range(len(all_data_x))
                    part_res_all_data = calc_sum_dic(zip(pseudo_keys, all_data_x, all_data_y))

                    label_keys_new = submodularity_NB(get_indices(train_set_without_labx, label_keys_sub, unlabeled_keys),
                                                      label_keys_sub, y_labels, vocab_size, n_class, docs_to_add,part_res_all_data)
                elif args.representativeness == "random":
                    label_keys_new = random.sample(label_keys_sub, docs_to_add)
                elif args.representativeness == "density":
                    label_keys_new=label_keys_sub
                else:
                    label_keys_new = calc_representative_set(all_data_x,
                                                             get_indices(train_set_without_labx, label_keys_sub,
                                                                         unlabeled_keys),
                                                             label_keys_sub, lexicon, docs_to_add)

        else:
            break
    result_dict["num_pos"] = num_pos
    result_dict["num_neg"] = num_neg
    result_dict["f1"] = result_f1
    result_dict["prediction"] = test_predictions
    result_dict["test_labs"] = test_set_y
    return result_dict


def create_data_frame(set_x, lex_len):
    result = pd.DataFrame(set_x, columns=range(lex_len))
    result = result.fillna(0)
    return result


def calc_representative_set(total_x,uncertainty_x,uncertainty_keys,lexicon,iterations = 60,t=10,num_models=30):

    rep_keys = pd.DataFrame({"key": uncertainty_keys})
    uncertainty_matrix = create_data_frame(uncertainty_x, len(lexicon))
    rest_matrix = create_data_frame(total_x, len(lexicon))
    rest_matrix["lab"] = 0
    uncertainty_matrix["lab"] = 1
    n = len(uncertainty_x)

    for ite in range(iterations):
        prediction_arr = [[0, 0] for _ in range(len(uncertainty_x))]
        for _ in range(num_models):
            permutation = np.random.permutation(uncertainty_matrix.index)

            rep_keys = rep_keys.reindex(permutation)
            uncertainty_matrix = uncertainty_matrix.reindex(permutation)

            fst = int(n / 2)  # first half
            lst = n - fst  # second half

            df_1 = uncertainty_matrix.head(fst).copy()
            df_2 = uncertainty_matrix.tail(lst).copy()

            train = pd.concat([rest_matrix.sample(n=lst), df_2])

            y = train.lab
            X = train.drop(columns=['lab'], axis=1).copy()
            clf=LinearSVC(C=0.0001,max_iter=4000) #SGDClassifier(loss='log', penalty='l1',alpha=1e-3, random_state=42,max_iter=10, tol=None)
                                                    #MultinomialNB(fit_prior=False)
            clf.fit(X, y)

            rem = df_1.drop(columns=['lab'], axis=1)
            result = [i for i in clf.predict(rem)]
            #####################################
            for entry, permut_idx in zip(result, permutation):
                prediction_arr[permut_idx][0] += entry
                prediction_arr[permut_idx][1] += 1
            #####################################

        pred_res = [float(x[0]) / float(x[1]) if x[1] != 0 else 0.0 for x in prediction_arr]

        for _ in range(t):
            drop_idx=np.argmax(pred_res)
            rep_keys = rep_keys.drop(drop_idx).copy()
            uncertainty_matrix = uncertainty_matrix.drop(drop_idx).copy()
            pred_res[drop_idx] = 0.0
            n = n - 1

    res = rep_keys.values
    res = [row[0] for row in res]
    res.sort()
    print("len docs to add:", len(res))

    return res


def calc_sum_dic(all_data):
    part_res = {}
    for entry in all_data:
        for k, v in entry[1].items():
            v = min(v, 3)
            if (k, v, entry[2]) in part_res:
                part_res[(k, v, entry[2])] += 1.0
            else:
                part_res[(k, v, entry[2])] = 1.0
    return part_res


def eval_f(part_res_all_data, permut_dict, vocab_size, n_class):
    f_nb = 0
    for key, value in permut_dict.items():
        f_nb += part_res_all_data[key] * np.log(value)
    return f_nb


def submodularity_NB(sub_x, sub_x_keys, sub_y, voc_len, clss_num, docs_to_add, part_res_all_data):
    # calculate all permutations of subset:
    assert len(sub_x) == len(sub_x_keys) == len(sub_y)
    Ai = []
    Ai_keys = set()
    f_ai = 0
    for i in range(docs_to_add):
        f_ai_min = float("inf")
        min_key = None
        min_elem = None
        for entry in zip(sub_x_keys, sub_x, sub_y):
            if not entry[0] in Ai_keys:
                permut_dict_ai_e = calc_sum_dic(Ai + [entry])
                f_a_elem = eval_f(part_res_all_data, permut_dict_ai_e, voc_len, clss_num)
                if f_a_elem - f_ai < f_ai_min:
                    f_ai_min = f_a_elem - f_ai
                    min_key = entry[0]
                    min_elem = entry
        f_ai = f_ai_min
        Ai.append(min_elem)
        Ai_keys.add(min_key)

    return list(Ai_keys)


def calc_clss_list(values, clss_num):
    result = [{} for i in range(clss_num)]
    for entry in values:
        result[entry[2]][entry[0]] = entry[1]
    return result


def calc_distance(entry1, entry2):
    distance = 0
    for k, v in entry1.items():
        if k in entry2:
            distance += ((v - entry2[k]) ** 2)
        else:
            distance += (v ** 2)

    for k, v in entry2.items():
        if k not in entry1:
            distance += (v ** 2)

    return np.sqrt(distance)


def calc_max_distance(V):
    max_distance = 0
    for entry1 in V:
        for entry2 in V:
            distance = calc_distance(entry1[1], entry2[1])
            if distance > max_distance:
                max_distance = distance
    return max_distance

def read_density(density_path):
    density_file = open(density_path, "r")
    return [float(density) for density in density_file]

def write_description(out, args):
    out.write(str(args) + "\n")
    print(str(args))


if __name__ == "__main__":
    args = parseArgs()
    data_dir = args.data_dir
    out_path = str(os.path.join(data_dir, args.job_data + args.out_file))
    summary_file = open(out_path, "w")

    job_data = args.job_data
    fold_id = int(job_data[0])
    my_seed = int(job_data[1])

    args = parseArgs()
    docs_to_add = args.docs_to_add
    data_dir = args.data_dir
    non_linearity_s = args.non_linearity

    if non_linearity_s == 'tanh':
        non_linearity = tf.nn.tanh
    elif non_linearity_s == 'sigmoid':
        non_linearity = tf.nn.sigmoid
    elif non_linearity_s == "lrelu":
        non_linearity = tf.nn.leaky_relu
    else:
        non_linearity = use.myrelu
    density_file = args.density_file


    lexicon = use.read_lex(data_dir, "vocab_path_old.txt")
    vocab_size = len(lexicon)
    data_set_y, data_set, data_count = use.read_input_data_tweets(data_dir, 'out_path_x_old.txt',
                                                                  'out_path_y_old.txt')
    n_class = 2
    n_splits = 5
    n_rounds = 30
    init_add = 100

    # cross_validation:
    sss = fold(n_splits=n_splits, random_state=0)
    f1_fold = []
    num_steps = []
    keys_ = []

    print(str(args))
    curr_fold = -1
    for train_index, test_index in sss.split(data_set_y):

        curr_fold += 1
        if curr_fold == fold_id:

            train_set,train_set_y = use.get_indices(data_set, train_index), use.get_indices(data_set_y,train_index)
            test_set, test_set_y = use.get_indices(data_set, test_index),  use.get_indices(data_set_y,test_index)
            if args.representativeness=="density":
                density_path = str(os.path.join(data_dir, density_file)) + str(curr_fold) + ".txt"
                doc_density = read_density(density_path)
            else:
                doc_density = None
            out_dict = active_learning(train_set, train_set_y,
                                                          test_set,  test_set_y, docs_to_add,
                                                          doc_density, n_rounds, n_class, args, my_seed, vocab_size,
                                                          non_linearity,
                                                          lexicon,
                                                          init_add=init_add, doc_keys=None)
            out_dict["description"] = args
            pickle.dump(out_dict, summary_file)
            summary_file.close()

