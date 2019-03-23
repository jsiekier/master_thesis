"""Document density implementation by Julia Siekiera
Requires:                       format per line
            vocab_path.txt:word
            out_path_x.txt:1 word_id+1:number_of_word_per_sentence ...
            out_path_y.txt:label (integer)
"""
import math
import argparse
from sklearn.model_selection import KFold as fold
import usefull as use
import os
import utils

def parseArgs():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--lamb', default=0.8, type=float)
    argparser.add_argument('--beta', default=3, type=float)
    argparser.add_argument('--data_dir', default="../data", type=str)
    argparser.add_argument('--dataset_name', default='tweets', type=str)
    argparser.add_argument('--density_file', default='density_', type=str)
    return argparser.parse_args()

#Propapility of a word
def calc_voc(data, voc_size):
    prop_voc = [0] * voc_size
    all=0
    for j in range(len(data)):
        for key, value in data[j].items():
            prop_voc[key] += value
            all+=value

    return [float(i) / float(all) for i in prop_voc]

#   Eq: 9
def calculate_doc_density(docs, voc_prob, beta, lam):
    doc_density = {}
    D = float(len(docs))
    i=-1
    for doc_i in docs:
        i+=1
        part_res = 0.0
        for doc_j in docs:
            dis=calc_distance(voc_prob, doc_i, doc_j, beta, lam)
            if dis!=0:
                part_res += math.log(dis)
        doc_density[i] = math.exp(part_res/D)
    return doc_density

#   Eq: 8
def calc_distance(voc_prop, doc_i, doc_j, beta, lam):
    #   KL divergence:
    kl_div = 0
    len_j=float(len(doc_j))
    len_i=float(len(doc_i))
    for word,num in doc_j.items():
        if word not in doc_i and voc_prop[word]!=0:
            kl_div += ((float(num)/ len_j)* math.log((float(num) / len_j)/ (( (1 - lam) * voc_prop[word]))))

        else:
            kl_div += ((float(num) / len_j)* math.log((float(num) / len_j)
                                                      / ((lam * (float(doc_i[word]) / len_i) + (1 - lam) * voc_prop[word]))))
    return math.exp(-beta * kl_div)


def write(doc_density, indx, density_file,data_dir):
    density_output=open(data_dir+"/"+density_file+str(indx)+".txt","w")
    for key,density in doc_density.items():
        density_output.write(str(density)+"\n")
    density_output.close()

def read_input_data(data_dir):
    train_url = os.path.join(data_dir, 'out_x20.txt')
    train_url_y = os.path.join(data_dir, 'out_y20.txt')
    test_url = os.path.join(data_dir, 'out_x_test20.txt')
    test_url_y = os.path.join(data_dir, 'out_y_test20.txt')
    train_set_y = utils.data_set_y(train_url_y)
    test_set_y = utils.data_set_y(test_url_y)
    train_set, train_count = utils.data_set(train_url)
    test_set, test_count = utils.data_set(test_url)

    train_set_y.extend(test_set_y)
    train_set.extend(test_set)
    train_count.extend(test_count)

    return train_set_y,train_set, train_count
if __name__ == "__main__":

    args = parseArgs()
    lambdaa = args.lamb
    beta = args.beta

    data_dir = args.data_dir
    dataset_name = args.dataset_name

    if dataset_name == "tweets":
        lexicon = use.read_lex(data_dir, "vocab_path.txt")
        vocab_size = len(lexicon)
        data_set_y, data_set, data_count = use.read_input_data_tweets(data_dir,'out_path_x.txt', 'out_path_y.txt')
        n_splits=5

    else:
        lexicon = use.read_lex(data_dir, "vocab.new")
        vocab_size = len(lexicon)
        data_set_y, data_set, data_count = read_input_data(data_dir)
        n_splits=3
        # cross_validation:
    sss = fold(n_splits=n_splits, random_state=0)
    indx = 0
    for train_index, test_index in sss.split(data_set_y):
        train_set, train_count, train_set_y = use.get_indices(data_set, train_index), \
                                              use.get_indices(data_count, train_index), use.get_indices(data_set_y,
                                                                                                        train_index)
        test_set, test_count, test_set_y = use.get_indices(data_set, test_index), \
                                           use.get_indices(data_count, test_index), use.get_indices(data_set_y,
                                                                                                    test_index)


        voc = calc_voc(train_set, vocab_size)
        # calculate the density for each tweet:
        doc_density = calculate_doc_density(train_set, voc, beta, lambdaa)
        write(doc_density, indx, args.density_file, data_dir)
        indx += 1
        print(indx)


