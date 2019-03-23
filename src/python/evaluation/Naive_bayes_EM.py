import argparse
import numpy as np
import utils
import os
from scipy.sparse import csr_matrix, vstack
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import KFold as fold
import usefull as use
from active_learning import  read_density
import scipy.stats.distributions as d
import pandas as pd
import random
import math





def parseArgs():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_dir', default="data_julia", type=str)
    argparser.add_argument('--density_file', default='density_fold_old_', type=str)
    argparser.add_argument('--use_EM', default=0, type=int)
    argparser.add_argument('--random', default=0, type=int)
    return argparser.parse_args()  # argstring.split()


def transform(data,voc_len):
    col=[]
    row=[]
    vals=[]
    row_counter=-1
    for doc in data:
        row_counter+=1
        for key, value in doc.items():
            row.append(row_counter)
            col.append(key)
            vals.append(value)
    return csr_matrix( (vals,(row,col)), shape=(len(data),voc_len) )
def has_change(labeled_docs, labeled_docs2):
    for i in range(len(labeled_docs)):
        if labeled_docs[i]!=labeled_docs2[i]:
            return True
    return False
def label_docs_EM(y_pred_class,train_y, train, test):
    change = True
    nb=None
    train_data_x= vstack((train,test))
    while (change):
        myList=y_pred_class.tolist()
        train_data_y = [i for i in train_y]
        train_data_y.extend(myList)
        nb = MultinomialNB()
        nb.fit(train_data_x, train_data_y)
        labeled_docs2 = nb.predict(test)
        change = has_change(y_pred_class, labeled_docs2)
        y_pred_class = labeled_docs2
    return y_pred_class,nb

#   Eq: 5
def label_docs(docs, theta_w_c, theta_c,n_class):
    labled_docs = []
    for i in range(docs.shape[0]):#docs
        best_label = None
        max_prop = 0

        doc=docs[i,:]
        rows, cols = doc.nonzero()

        multi=[0]*n_class
        for label in range(n_class):
            multi[label] = theta_c[label]
            for word in cols:
                multi[label] *= theta_w_c[(word, label)]

        sum1 = sum(multi)
        if sum1 != 0:
            for label in range(n_class):
                if (float(multi[label] )/ float(sum1)) > max_prop:
                    max_prop = float(multi[label])/ float(sum1)
                    best_label = label
        else:
            best_label=random.randint(0,n_class-1)

        labled_docs.append(best_label)
    return labled_docs

def create_committee(samples_x, samples_y,test,n_class):
    D=float(len(samples_y))
    theta_w_c = {}
    theta_c={}
    v_t_j={}

    for j in range(samples_x.shape[1]):  # vocab
        a=samples_x[:,j]
        rows, cols = a.nonzero()
        lab0=[0]*n_class
        for i in rows:
            lab0[samples_y[i]]+=a[i,0]

        for lab in range(n_class):
            v_t_j[(j, lab)] = d.gamma.rvs(1 + lab0[lab])

    lab0 = [0] * n_class
    for entry in samples_y:
        lab0[entry]+=1
    for lab in range(n_class):
        theta_c[lab]=float(lab0[lab])/D


    for label in range(n_class):
        sum1= 0
        for j in range(samples_x.shape[1]):
            sum1+=v_t_j[(j,label)]
        for j in range(samples_x.shape[1]):
            theta_w_c[(j,label)]=float(v_t_j[(j,label)])/float(sum1)

    labeled_docs = label_docs(test, theta_w_c, theta_c,n_class)
    return pd.Series(labeled_docs)

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

def cals_docs_to_add(all_committees, test, doc_density, num_docs_added, doc_indices,n_class):
    disagree = []
    new_train=[]
    k= float(len(all_committees))
    p_avg={}

    for doc in range(test.shape[0]):
        for label in range(n_class):
            avg = 0
            for comittee in all_committees:
                if comittee[doc] == label:
                    avg += 1
            p_avg[(label,doc)] = (float(avg) / k)

        sum1 = 0
        for comittee in all_committees:
            for label in range(n_class):
                if comittee[doc]==label and p_avg[(label,doc)]!= 0:
                    sum1+=math.log(1.0/p_avg[(label, doc)])
        disagree.append((doc_density[doc_indices[doc]]*sum1/k,doc_indices[doc]))#TODO minus added!!!!!!!!!!!!!

    # sort
    sorted_list= merge_sort(disagree)
    #select best docs:
    for i in  range(num_docs_added):
        new_train.append(sorted_list[i][1])
    return new_train
def active_learning(train_set_d, train_set_y_d,test_set, test_set_y, docs_to_add,
                    doc_density,num_iterations,n_class,lexicon,args,init_add=50,doc_keys=None):

    if not doc_keys:
        label_keys = random.sample(range(0, len(train_set_y_d)), init_add)
    else:
        label_keys=doc_keys

    result_f1= []
    num_docs = []
    label_keys_new = []
    test = transform(test_set, len(lexicon))
    for i in range(num_iterations):

        label_keys = list(set(label_keys).union(label_keys_new))
        unlabeled_keys = list(set(range(len(train_set_d))).difference(set(label_keys)))
        unlabeled_keys.sort()
        label_keys.sort()

        train_set_with_lab = use.get_indices(train_set_d, label_keys)
        train_set_y = use.get_indices(train_set_y_d, label_keys)

        train_set_without_labx = use.get_indices(train_set_d, unlabeled_keys)


        ####f1
        train = transform(train_set_with_lab, len(lexicon))


        unlabeled_train = transform(train_set_without_labx, len(lexicon))

        nb = MultinomialNB()
        nb.fit(train, train_set_y)
        if args.use_EM:
            if unlabeled_train.shape[0] > 0:
                pred_unlabeled = nb.predict(unlabeled_train)

                _, nb = label_docs_EM(pred_unlabeled, train_set_y, train, unlabeled_train)

        pred = nb.predict(test)

        if n_class == 2:
            f1_mes = metrics.f1_score(test_set_y, pred)
        else:
            f1_mes = metrics.f1_score(test_set_y, pred, average='weighted')

        result_f1.append(f1_mes)
        num_docs.append(len(label_keys))

        all_committees=[]
        for j in range(3):
            # create a committee member:
            y_pred_class = create_committee(train, train_set_y,test,n_class)

            # apply EM with the unlabeled Data. Loop while Parameters change:
            y_pred_class,nb=label_docs_EM(y_pred_class,train_set_y, train, test)

            all_committees.append(y_pred_class)
        # Calculate the disagreement for each unlabeled Document. multiply by its density
        # and request class label for the ones with the highes score:

        if len(unlabeled_keys)>=docs_to_add:
            if args.random:
                label_keys_new = random.sample(unlabeled_keys, docs_to_add)
            else:
                label_keys_new = cals_docs_to_add(all_committees, test, doc_density, docs_to_add,unlabeled_keys,n_class)
        else:
            break
    return result_f1,num_docs,label_keys


def main_cls(seed):
    np.random.seed(seed)
    args = parseArgs()
    data_dir = args.data_dir

    lexicon = use.read_lex(data_dir, "vocab_path_old.txt")
    data_set_y, data_set, data_count = use.read_input_data_tweets(data_dir,'out_path_x_old.txt', 'out_path_y_old.txt')
    n_class = 2
    n_splits=5
    n_rounds=30
    init_add=100


    sss = fold(n_splits=n_splits, random_state=0)
    res_folds = []
    indx = -1
    density_file = args.density_file
    my_keys=[]
    for train_index, test_index in sss.split(data_set_y):
        indx += 1
        train_set, train_count, train_set_yy = use.get_indices(data_set, train_index), \
                                               use.get_indices(data_count, train_index), use.get_indices(data_set_y,
                                                                                                         train_index)
        test_set, test_count, test_set_y = use.get_indices(data_set, test_index), \
                                           use.get_indices(data_count, test_index), use.get_indices(data_set_y,
                                                                                                    test_index)
        density_path = str(os.path.join(data_dir, density_file)) + str(indx) + ".txt"
        doc_density = read_density(density_path)
        #doc_density = [1]*len(train_set_yy) #read_density(density_path)
        f1_res, num_steps, doc_keys = active_learning(train_set, train_set_yy, test_set, test_set_y, 100,
                                                      doc_density,n_rounds, n_class, lexicon, args,init_add=init_add,
                                                      doc_keys=None)

        res_folds.append(f1_res)
        my_keys.append(doc_keys)


    mean = use.calc_mean(res_folds)
    return mean


if __name__ == '__main__':
    mean_mean=[]
    for i in range(10):
        print(i)
        res=main_cls(i)
        print(res)
        mean_mean.append(res)
    print(use.calc_mean(mean_mean))