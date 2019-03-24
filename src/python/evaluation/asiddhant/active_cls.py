from __future__ import print_function
from collections import OrderedDict
import os
import neural_cls
from neural_cls.util import Trainer, Loader
from neural_cls.models import BiLSTM
from neural_cls.models import CNN
from neural_cls.models import BiLSTM_MC
from neural_cls.models import BiLSTM_BB
from neural_cls.models import CNN_MC
from neural_cls.models import CNN_BB
import matplotlib.pyplot as plt
import torch
from active_learning import Acquisition_CLS
import _pickle as pkl
import numpy as np

import argparse

'''The code was provided by https://github.com/asiddhant/Active-NLP and adapted by Julia Siekiera to our evaluation setting'''
def calc_mean(res_folds):
    mean = [0.0] * len(res_folds[0])
    for i in range(len(res_folds)):
        for j in range(len(res_folds[i])):
            mean[j] += res_folds[i][j]
    for j in range(len(mean)):
        mean[j] /= float(len(res_folds))
    return mean


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', action='store', dest='dataset', default="twitter", type=str,
                    help='Dataset to be Used')  # 'mareview'
parser.add_argument('--result_path', action='store', dest='result_path', default='neural_cls/results/',
                    type=str, help='Path to Save/Load Result')
parser.add_argument('--usemodel', default='CNN', type=str, dest='usemodel',
                    help='Model to Use')  # CNN
parser.add_argument('--worddim', default=300, type=int, dest='worddim',
                    help="Word Embedding Dimension")
parser.add_argument('--pretrnd', default="wordvectors/glove.6B.300d.txt", type=str, dest='pretrnd',
                    help="Location of pretrained embeddings")
parser.add_argument('--reload', default=0, type=int, dest='reload',
                    help="Reload the last saved model")
parser.add_argument('--checkpoint', default=".", type=str, dest='checkpoint',
                    help="Location of trained Model")
parser.add_argument('--initdata', default=100, type=int, dest='initdata',
                    help="Number of Data Points to begin with")
parser.add_argument('--acquiremethod', default='mnlp', type=str, dest='acquiremethod',
                    help="Percentage of Data to Acquire from Rest of Training Set")  # 'random',mnlp
parser.add_argument('--num_docs_to_add', default=100, type=int, dest='num_docs_to_add',
                    help="Number of data points to add in each step")
parser.add_argument('--out_file', default="ac_with_stem.txt", type=str)
parser.add_argument('--job_id', default="100", type=str)

parameters = OrderedDict()

opt = parser.parse_args()

parameters['model'] = opt.usemodel
parameters['wrdim'] = opt.worddim
parameters['ptrnd'] = opt.pretrnd

if opt.usemodel == 'CNN' and opt.dataset == 'trec':
    parameters['dpout'] = 0.5
    parameters['wlchl'] = 100
    parameters['nepch'] = 10

    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 6
    parameters['acqmd'] = 'd'

elif opt.usemodel == 'CNN' and opt.dataset == 'mareview':
    parameters['dpout'] = 0.5
    parameters['wlchl'] = 100
    parameters['nepch'] = 10

    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 2
    parameters['acqmd'] = 'd'

elif opt.usemodel == 'BiLSTM' and opt.dataset == 'trec':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 200
    parameters['nepch'] = 10

    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 6
    parameters['acqmd'] = 'd'

elif opt.usemodel == 'BiLSTM' and opt.dataset == 'mareview':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 200
    parameters['nepch'] = 10

    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 2
    parameters['acqmd'] = 'd'

elif opt.usemodel == 'CNN_MC' and opt.dataset == 'trec':
    parameters['dpout'] = 0.5
    parameters['wlchl'] = 100
    parameters['nepch'] = 10

    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 6
    parameters['acqmd'] = 'm'

elif opt.usemodel == 'CNN_MC' and opt.dataset == 'mareview':
    parameters['dpout'] = 0.5
    parameters['wlchl'] = 100
    parameters['nepch'] = 10

    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 2
    parameters['acqmd'] = 'm'

elif opt.usemodel == 'BiLSTM_MC' and opt.dataset == 'trec':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 200
    parameters['nepch'] = 10

    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 6
    parameters['acqmd'] = 'm'

elif opt.usemodel == 'BiLSTM_MC' and opt.dataset == 'mareview':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 200
    parameters['nepch'] = 10

    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 2
    parameters['acqmd'] = 'm'

elif opt.usemodel == 'CNN_BB' and opt.dataset == 'trec':
    parameters['wlchl'] = 100
    parameters['nepch'] = 25

    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 6
    parameters['sigmp'] = float(np.exp(-3))

    parameters['acqmd'] = 'b'

elif opt.usemodel == 'CNN_BB' and opt.dataset == 'mareview':
    parameters['wlchl'] = 100
    parameters['nepch'] = 25

    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 2
    parameters['sigmp'] = float(np.exp(-3))

    parameters['acqmd'] = 'b'

elif opt.usemodel == 'BiLSTM_BB' and opt.dataset == 'trec':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 200
    parameters['nepch'] = 25

    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 6
    parameters['sigmp'] = float(np.exp(-3))

    parameters['acqmd'] = 'b'

elif opt.usemodel == 'BiLSTM_BB' and opt.dataset == 'mareview':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 200
    parameters['nepch'] = 25

    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 2
    parameters['sigmp'] = float(np.exp(-3))

    parameters['acqmd'] = 'b'
elif opt.usemodel == 'CNN' and opt.dataset == 'twitter':
    parameters['dpout'] = 0.5
    parameters['wlchl'] = 100
    parameters['nepch'] = 10

    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50  # 35?
    parameters['opsiz'] = 2  # was 6
    parameters['acqmd'] = 'd'
elif opt.usemodel == 'CNN_BB' and opt.dataset == 'twitter':
    parameters['wlchl'] = 100
    parameters['nepch'] = 10

    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 2
    parameters['sigmp'] = float(np.exp(-3))
    parameters['acqmd'] = 'b'
else:
    raise NotImplementedError()

use_cuda = False
use_dataset = opt.dataset
dataset_path = os.path.join('datasets', use_dataset)
if opt.dataset == 'twitter':
    dataset_path = 'data/twitter'

if not os.path.exists(opt.job_id):
    os.makedirs(opt.job_id)

result_path = os.path.join(opt.job_id, use_dataset)
model_name = opt.usemodel
model_load = opt.reload
checkpoint = opt.checkpoint
init_num = opt.initdata
acquire_method = opt.acquiremethod
loader = Loader()

print('Model:', model_name)
print('Dataset:', use_dataset)
print('Acquisition:', acquire_method)
out_path = str(os.path.join(dataset_path, opt.job_id+opt.out_file))
out = open(out_path, "w")
out.write(str(opt) + "\n")



if not os.path.exists(result_path):
    os.makedirs(result_path)

if not os.path.exists(os.path.join(result_path, model_name)):
    os.makedirs(os.path.join(result_path, model_name))

if not os.path.exists(os.path.join(result_path, model_name, 'active_checkpoint', acquire_method)):
    os.makedirs(os.path.join(result_path, model_name, 'active_checkpoint', acquire_method))

if opt.dataset == 'trec':
    train_data, test_data, mappings = loader.load_trec(dataset_path, parameters['ptrnd'],
                                                       parameters['wrdim'])
elif opt.dataset == 'mareview':
    train_data, test_data, mappings = loader.load_mareview(dataset_path, parameters['ptrnd'],
                                                           parameters['wrdim'])
elif opt.dataset == 'twitter':

    f_mes_folds = []
    fold_id = int(opt.job_id[1])

    for split_id in range(5):
        if split_id==fold_id:

            train_data, test_data, mappings = loader.load_tweets(dataset_path, parameters['ptrnd'],
                                                                 parameters['wrdim'], n_splits=5, split_id=split_id)

            word_to_id = mappings['word_to_id']
            tag_to_id = mappings['tag_to_id']
            word_embeds = mappings['word_embeds']

            print('Load Complete')

            total_sentences = len(train_data)
            avail_budget = total_sentences

            print('Building Model............................................................................')
            if (model_name == 'BiLSTM'):
                print('BiLSTM')
                word_vocab_size = len(word_to_id)
                word_embedding_dim = parameters['wrdim']
                word_hidden_dim = parameters['wldim']
                output_size = parameters['opsiz']

                model = BiLSTM(word_vocab_size, word_embedding_dim, word_hidden_dim,
                               output_size, pretrained=word_embeds)

            elif (model_name == 'CNN'):
                print('CNN')
                word_vocab_size = len(word_to_id)
                word_embedding_dim = parameters['wrdim']
                word_out_channels = parameters['wlchl']
                output_size = parameters['opsiz']

                model = CNN(word_vocab_size, word_embedding_dim, word_out_channels,
                            output_size, pretrained=word_embeds)

            elif (model_name == 'BiLSTM_MC'):
                print('BiLSTM_MC')
                word_vocab_size = len(word_to_id)
                word_embedding_dim = parameters['wrdim']
                word_hidden_dim = parameters['wldim']
                output_size = parameters['opsiz']

                model = BiLSTM_MC(word_vocab_size, word_embedding_dim, word_hidden_dim,
                                  output_size, pretrained=word_embeds)

            elif (model_name == 'CNN_MC'):
                print('CNN_MC')
                word_vocab_size = len(word_to_id)
                word_embedding_dim = parameters['wrdim']
                word_out_channels = parameters['wlchl']
                output_size = parameters['opsiz']

                model = CNN_MC(word_vocab_size, word_embedding_dim, word_out_channels,
                               output_size, pretrained=word_embeds)

            elif (model_name == 'CNN_BB'):
                print('CNN_BB')
                word_vocab_size = len(word_to_id)
                word_embedding_dim = parameters['wrdim']
                word_out_channels = parameters['wlchl']
                output_size = parameters['opsiz']
                sigma_prior = parameters['sigmp']

                model = CNN_BB(word_vocab_size, word_embedding_dim, word_out_channels,
                               output_size, sigma_prior=sigma_prior, pretrained=word_embeds)

            elif (model_name == 'BiLSTM_BB'):
                print('BiLSTM_BB')
                word_vocab_size = len(word_to_id)
                word_embedding_dim = parameters['wrdim']
                word_hidden_dim = parameters['wldim']
                output_size = parameters['opsiz']
                sigma_prior = parameters['sigmp']

                model = BiLSTM_BB(word_vocab_size, word_embedding_dim, word_hidden_dim,
                                  output_size, sigma_prior=sigma_prior, pretrained=word_embeds)

            if model_load:
                print('Loading Saved Weights....................................................................')
                acquisition_path = os.path.join(result_path, model_name, 'active_checkpoint', acquire_method,
                                                checkpoint, 'acquisition2.p')
                acquisition_function = pkl.load(open(acquisition_path, 'rb'))

            else:

                acquisition_function = Acquisition_CLS(train_data, init_num=init_num, seed=int(opt.job_id[2]),
                                                       acq_mode=parameters['acqmd'], usecuda=use_cuda)
            if use_cuda:
                model.cuda()
            learning_rate = parameters['lrate']
            num_epochs = parameters['nepch']
            print('Initial learning rate is: %s' % (learning_rate))
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            trainer = Trainer(model, optimizer, result_path, model_name, tag_to_id, usedataset=opt.dataset,
                              usecuda=use_cuda)  # TODO cuda

            active_train_data = [train_data[i] for i in acquisition_function.train_index]
            sentences_acquired = len(acquisition_function.train_index)

            num_docs_to_add = opt.num_docs_to_add  # 5

            # num_acquisitions_required = 34#25
            # acquisition_strat_all = [2]*24 + [5]*10 + [0]
            # acquisition_strat = acquisition_strat_all[:num_acquisitions_required]

            training_docs = 0
            fmeasures = [0]
            x_axis = [init_num]
            while training_docs <= 3000:
                training_docs += num_docs_to_add
                #checkpoint_folder = os.path.join('active_checkpoint', acquire_method, str(sentences_acquired).zfill(8))
                #checkpoint_path = os.path.join(result_path, model_name, checkpoint_folder)
                #if not os.path.exists(checkpoint_path):
                #    os.makedirs(checkpoint_path)

                acq_plot_every = max(len(acquisition_function.train_index) / (5 * parameters['batch_size']), 1)
                losses, all_F, trained_model = trainer.train_model(num_epochs, active_train_data, test_data, learning_rate,
                                                    batch_size=parameters['batch_size'],
                                                    plot_every=acq_plot_every)

                #pkl.dump(acquisition_function, open(os.path.join(checkpoint_path, 'acquisition1.p'), 'wb'))

                acquisition_function.obtain_data(trained_model=trained_model,
                                                 model_name=model_name,
                                                 data=train_data, acquire=num_docs_to_add, method=acquire_method)

                #pkl.dump(acquisition_function, open(os.path.join(checkpoint_path, 'acquisition2.p'), 'wb'))

                print('*' * 80)
                saved_epoch = np.argmax(np.array([item[1] for item in all_F]))
                print('Budget Exhausted: %d, Best F on Train %.3f, Best F on Test %.3f' % (sentences_acquired,
                                                                                           all_F[saved_epoch][0],
                                                                                           all_F[saved_epoch][1]))
                print('*' * 80)
                fmeasures.append(all_F[-1][1])#changed from saved_epoch to -1
                active_train_data = [train_data[i] for i in acquisition_function.train_index]
                sentences_acquired = len(acquisition_function.train_index)

                x_axis.append(x_axis[-1] + num_docs_to_add)
                #plt.clf()
                #plt.plot(x_axis, fmeasures)
                #plt.savefig(os.path.join(checkpoint_path, 'fmeasureplot.png'))
                print(x_axis, fmeasures)
                print(training_docs)
            out.write(str(fmeasures) + "\n")
            out.write(str(x_axis) + "\n")
            out.flush()
            f_mes_folds.append(fmeasures)
            print(f_mes_folds)



else:
    raise NotImplementedError()
out.close()
