import os
import glob
import random
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


def read_dictionary(dicname, clss):
    result = []
    for filename in glob.glob(os.path.join(dicname, '*.txt')):
        text = ""
        txt_file = open(filename, "r", encoding="utf-8")
        for line in txt_file:
            text += line + " "
        result.append((clss, text))
        txt_file.close()

    return result


def create_data_set(data_url, vect, out_file):
    out_test = open(out_file, "w")

    data_pos = read_dictionary(os.path.join(data_url, 'pos\\'), 1)
    data_neg = read_dictionary(os.path.join(data_url, 'neg\\'), 0)
    input_data = data_pos + data_neg
    random.shuffle(input_data)

    for target, text in input_data:
        indices = vect.transform([text])
        rows_j, cols_j = indices.nonzero()
        if len(cols_j) > 0:
            line = str(target)
            for j in cols_j:
                line += " " + str(j + 1) + ":" + str(int(indices[0, j]))
            if line != str(target):
                out_test.write(line + "\n")
    out_test.close()


def create_voc(data_url):
    vocab = []

    ps = PorterStemmer()
    data_pos = read_dictionary(os.path.join(data_url, 'pos\\'), 1)
    data_neg = read_dictionary(os.path.join(data_url, 'neg\\'), 0)
    input_data = data_pos + data_neg
    random.shuffle(input_data)

    for target, text in input_data:
        new_txt=""
        words = word_tokenize(text)
        for word in words:
            new_txt+=" "+ps.stem(word)
        vocab.append(new_txt)
    return list(vocab)



def read_input_data(data_dir):
    stop_word=["a", "an", "and", "are", "as", "at", "be", "by",
                    "for", "if", "in", "into", "is", "it", "of", "on", "or", "such", "that", "the", "their",
                    "then", "there", "these", "they", "this", "to", "was", "will", "with"]
    train_url = os.path.join(data_dir, 'train')
    test_url = os.path.join(data_dir, 'test')
    vocab_out = open("/imdb.vocab", "w", encoding="utf-8")
    vocab = create_voc(train_url)

    vect = CountVectorizer(stop_words=stop_word,min_df=0.001)
    vect.fit(list(vocab))

    vocab_size=len(vect.vocabulary_)
    print(vocab_size)
    voc=vect.get_feature_names()
    for v in voc:
        vocab_out.write(v+"\n")

    vocab_out.close()
    create_data_set(train_url, vect, "train_movie_rev.txt")
    create_data_set(test_url, vect, "test_movie_rev.txt")


read_input_data("aclImdb_v1\\aclImdb")
