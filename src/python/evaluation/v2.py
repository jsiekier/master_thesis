from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

import math
#import best_practice.utils as utils
import utils as utils

from sklearn import metrics

class NVDM(object):
    def __init__(self,
                 vocab_size,
                 mlp_arr,
                 n_topic,
                 learning_rate,
                 batch_size,
                 non_linearity,
                 adam_beta1,
                 adam_beta2,
                 dir_prior,
                 n_class,
                 N,
                 seed):
        np.random.seed(seed)
        tf.set_random_seed(seed)

        tf.reset_default_graph()
        self.vocab_size = vocab_size
        self.n_hidden = mlp_arr[0]
        self.n_topic = n_topic
        self.non_linearity = non_linearity
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_clss=n_class

        self.y = tf.placeholder(tf.float32, [None, n_class], name='input_y')
        self.lab = tf.placeholder(tf.float32, [1], name='input_with_lab')
        self.x = tf.placeholder(tf.float32, [None, vocab_size], name='input')
        self.idx=tf.placeholder(tf.int32,[1],name="index")
        self.mask = tf.placeholder(tf.float32, [None], name='mask')  # mask paddings
        self.warm_up = tf.placeholder(tf.float32, (), name='warm_up')  # warm up
        self.training=tf.placeholder(tf.bool,(),name="training")
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.min_alpha = tf.placeholder(tf.float32, (), name='min_alpha')
        back_mlp=False
        cat_distribution=False

        # encoder
        with tf.variable_scope('encoder'):
            #self.enc_input=tf.concat([self.x, self.y], axis=1)
            self.enc_vec = utils.mlp(self.x, mlp_arr, self.non_linearity)

            self.enc_vec = tf.nn.dropout(self.enc_vec, self.keep_prob)
            self.mean=tf.contrib.layers.batch_norm(utils.linear(self.enc_vec, self.n_topic, scope='mean'),is_training=self.training)
            self.alpha = tf.maximum(self.min_alpha, tf.log(1. + tf.exp(self.mean)))

            self.prior = tf.ones((batch_size, self.n_topic), dtype=tf.float32, name='prior') * dir_prior

            self.analytical_kld = tf.lgamma(tf.reduce_sum(self.alpha, axis=1)) - tf.lgamma(
                tf.reduce_sum(self.prior, axis=1))
            self.analytical_kld -= tf.reduce_sum(tf.lgamma(self.alpha), axis=1)
            self.analytical_kld += tf.reduce_sum(tf.lgamma(self.prior), axis=1)
            minus = self.alpha - self.prior
            # test = tf.reduce_sum(minus,1)
            test = tf.reduce_sum(tf.multiply(minus, tf.digamma(self.alpha) - tf.reshape(
                tf.digamma(tf.reduce_sum(self.alpha, 1)), (batch_size, 1))), 1)
            self.analytical_kld += test
            self.analytical_kld = self.mask * self.analytical_kld  # mask paddings

            #self.clss_mlp=utils.mlp(self.x, mlp_arr, self.non_linearity,scope="classifier_mlp")
            #self.clss_mlp=tf.nn.dropout(self.clss_mlp, self.keep_prob)
            self.phi = tf.contrib.layers.batch_norm(utils.linear(self.mean, n_class, scope='phi'),is_training=self.training) #y logits

            # class propabilities
            one_hot_dist=tfd.OneHotCategorical(logits=self.phi)
            hot_out=tf.squeeze(one_hot_dist.sample(1))
            hot_out.set_shape(self.phi.get_shape())
            self.dec_input = tf.cast(hot_out,dtype=tf.float32)

            self.out_y = tf.nn.softmax(self.phi, name="probabilities_y")
            # do NOT use output of softmax here!
            self.clss_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.phi) * 0.1 * N*self.mask

        with tf.variable_scope('decoder'):
            with tf.variable_scope('prob'):
                # Dirichlet
                self.doc_vec = tf.squeeze(tfd.Dirichlet(self.alpha).sample(1))  # tf.shape(self.alpha)
                self.doc_vec.set_shape(self.alpha.get_shape())
            # reconstruction
            if cat_distribution:
                self.merge = tf.cond((tf.reduce_sum(self.lab)) > 0,
                                     lambda: tf.concat([self.doc_vec, self.y], axis=1)
                                     , lambda: tf.concat([self.doc_vec, self.dec_input], axis=1))
            else:
                self.merge = tf.cond((tf.reduce_sum(self.lab)) > 0,
                                     lambda: tf.concat([self.doc_vec, self.y], axis=1)
                                     , lambda: tf.concat([self.doc_vec, self.out_y], axis=1))

            if not back_mlp:
                logits = tf.nn.log_softmax(tf.contrib.layers.batch_norm(
                    utils.linear(self.merge, self.vocab_size, scope='projection', no_bias=True),is_training=self.training))
            else:
                # this might
                logits = tf.nn.log_softmax(utils.mlp(self.merge, list(reversed(mlp_arr))+[self.vocab_size], scope='projection'))


            self.recons_loss = -tf.reduce_sum(tf.multiply(logits, self.x), 1)

            dir1 = tf.contrib.distributions.Dirichlet(self.prior)
            dir2 = tf.contrib.distributions.Dirichlet(self.alpha)

            self.kld = dir2.log_prob(self.doc_vec) - dir1.log_prob(self.doc_vec)

        self.min_l= self.recons_loss + self.warm_up * self.analytical_kld
        self.min_l_analytical=self.recons_loss + self.analytical_kld
        self.out_y_col=tf.transpose(tf.gather(tf.transpose(self.out_y),indices=self.idx))
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.phi, labels=self.out_y)*self.mask

        self.objective = tf.cond((tf.reduce_sum(self.lab)) > 0
                                 , lambda: self.min_l + self.clss_loss
                                 , lambda: self.min_l + self.cross_entropy)#tf.multiply(self.min_l,self.out_y_col)

        self.analytical_objective = tf.cond((tf.reduce_sum(self.lab)) > 0
                                            , lambda: self.min_l_analytical + self.clss_loss
                                            , lambda: self.min_l_analytical + self.cross_entropy)#tf.multiply(self.min_l_analytical,self.out_y_col)



        fullvars = tf.trainable_variables()

        enc_vars = utils.variable_parser(fullvars, 'encoder')
        dec_vars = utils.variable_parser(fullvars, 'decoder')

        # this is the standard gradient for the reconstruction network
        dec_grads = tf.gradients(self.objective, dec_vars)  # I changed this!

        enc_grads = tf.gradients(self.objective, enc_vars)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.adam_beta1,
                                           beta2=self.adam_beta2)
        self.optim_enc = optimizer.apply_gradients(zip(enc_grads, enc_vars))
        self.optim_dec = optimizer.apply_gradients(zip(dec_grads, dec_vars))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optim_all = optimizer.apply_gradients(list(zip(enc_grads, enc_vars)) + list(zip(dec_grads, dec_vars)))
            #self.optim_all=optimizer.minimize(self.objective)
        self.merged = tf.summary.merge_all()

    def run_model(self,labeled_training_rounds,
                  train_batches_with_lab,
                  train_set_with_lab,train_set_y,train_batches_without_lab,
                  train_set_without_lab,debug,semi_supervised,epoch,
                  warm_up,min_alpha,sess,optim=None,keep_prop=0.75,print_statement="",training=None
                  ):
        loss_sum1 = 0.0
        rec_sum1 = 0.0
        kld_sum1 = 0.0
        clss_loss_sum = 0.0
        cross_entropy_sum1 = 0.0
        kld_ana_sum = 0.0
        ana_loss_sum = 0.0
        pred_clss = []
        true_clss = []
        prop_clss = []
        for x in range(int(labeled_training_rounds)):
            for idx_batch in train_batches_with_lab:
                data_batch, mask = utils.fetch_data_new(train_set_with_lab, idx_batch, self.vocab_size)
                data_batch_y = utils.fetch_data_y(train_set_y, idx_batch, self.n_clss)
                input_feed = {self.x.name: data_batch, self.mask.name: mask, self.keep_prob.name: keep_prop,
                              self.warm_up.name: warm_up,self.min_alpha.name: min_alpha,
                              self.lab: np.ones((1)),self.y.name: data_batch_y,self.idx.name:np.zeros((1),dtype=np.int32),self.training.name:training}
                if optim:
                    _, ( loss, recon, kld_train, ana_loss, ana_kld_train, clss_loss,propability,my_phi) = sess.run((optim,
                                                                                                         [ self.objective,
                                                                                                           self.recons_loss,
                                                                                                           self.kld,
                                                                                                           self.analytical_objective,
                                                                                                           self.analytical_kld,
                                                                                                           self.clss_loss, self.out_y,self.phi]),
                                                                                                        input_feed)
                else:
                    loss, recon, kld_train, ana_loss, ana_kld_train, clss_loss, propability = sess.run(
                        [self.objective, self.recons_loss, self.kld,self.analytical_objective, self.analytical_kld,
                         self.clss_loss, self.out_y], input_feed)

                mask_sum = np.sum(mask)
                loss_sum1 += np.sum(loss) / mask_sum
                ana_loss_sum += np.sum(ana_loss) / mask_sum
                if debug:
                    rec_sum1 += np.sum(recon) / mask_sum
                    kld_sum1 += np.sum(kld_train) / mask_sum
                    ana_loss_sum += np.sum(ana_loss) / mask_sum
                    clss_loss_sum += np.sum(clss_loss) / mask_sum
                    kld_ana_sum+=np.sum(ana_kld_train) / mask_sum

                if x == 0:
                    preds = []
                    prop = []
                    j = -1
                    for i in propability:
                        j += 1
                        if idx_batch[j] != -1:
                            preds.append(np.argmax(i))
                            prop.append(np.ndarray.tolist(i))
                    pred_clss.extend(preds)
                    prop_clss.extend(prop)
                    true_clss.extend([train_set_y[i] for i in idx_batch if i != -1])

        if semi_supervised and len(train_batches_without_lab)>0:
            for clss_idx in range(1):
                for idx_batch in train_batches_without_lab:
                    data_batch, count_batch, mask = utils.fetch_data_new(train_set_without_lab,idx_batch, self.vocab_size)
                    data_batch_y = utils.fetch_data_y_dummy(idx_batch, self.n_clss,clss_idx)
                    input_feed = {self.x.name: data_batch, self.y.name: data_batch_y, self.mask.name: mask,
                                  self.keep_prob.name: keep_prop, self.lab: np.zeros((1)),
                                  self.warm_up.name: warm_up, self.min_alpha.name: min_alpha,
                                  self.idx.name:np.ones((1),dtype=np.int32)*clss_idx,self.training.name:training}
                    if optim:
                        _, (optim1, recon1, kld1, c_e1, ana_kld, ana_loss) = sess.run((optim, [self.objective,
                                                                                               self.recons_loss,
                                                                                               self.kld
                            ,self.cross_entropy, self.analytical_kld, self.analytical_objective]), input_feed)
                    else:
                        optim1, recon1, kld1, c_e1, ana_kld, ana_loss=sess.run([self.objective,self.recons_loss,self.kld
                            , self.cross_entropy, self.analytical_kld, self.analytical_objective], input_feed)
                    mask_sum = np.sum(mask)
                    loss_sum1 += np.sum(optim1) / mask_sum
                    ana_loss_sum += np.sum(ana_loss) / mask_sum
                    if debug:
                        rec_sum1 += np.sum(recon1) / mask_sum
                        kld_sum1 += np.sum(kld1) / mask_sum
                        cross_entropy_sum1 += np.sum(c_e1) / mask_sum
                        ana_loss_sum += np.sum(ana_loss) / mask_sum
                        kld_ana_sum += np.sum(ana_kld) / mask_sum
        len_all = (len(train_batches_with_lab) * labeled_training_rounds + len(train_batches_without_lab))
        print_ana_loss = ana_loss_sum / len_all
        print_loss1 = loss_sum1 / len_all
        if debug:
            print_kld1 = kld_sum1 / len_all
            print_rec1 = rec_sum1 / len_all
            print_ana_kld = kld_ana_sum / len_all
            if len(train_batches_without_lab)>0:
                print_c_e1 = cross_entropy_sum1 / len(train_batches_without_lab)
            else:
                print_c_e1 =0
            print_clss_loss = clss_loss_sum / (len(train_batches_with_lab) * labeled_training_rounds)
            print(print_statement,'| Epoch: {:d} |'.format(epoch + 1),
                  '| Loss 1: {:.3f}'.format(print_loss1),
                  '| KLD 1: {:.3f}'.format(print_kld1),
                  '| rec 1: {:.3f}'.format(print_rec1),
                  '| ana loss: {:.3f}'.format(print_ana_loss),
                  '| cross_e.: {:.3f}'.format(print_c_e1),
                  '|clss_sum: {:.3f}'.format(print_clss_loss),
                  '|KLD ana: {:.3f}'.format(print_ana_kld))
        if self.n_clss > 2:
            f1_measure = metrics.f1_score(true_clss, pred_clss, average='weighted')
        else:
            f1_measure = metrics.f1_score(true_clss, pred_clss)
        print(print_statement,"F1: ", f1_measure)
        return print_ana_loss,print_loss1,f1_measure,prop_clss


    def train_x(self,
              dev_set_with_lab,
              dev_set_without_lab,
              dev_set_y,
              train_set_with_lab,
              train_set_without_lab,
              train_set_y,
              test_set,
              test_set_y,
              to_label,
              model_name,  # 10
              lexicon=[],
              warm_up_period=100,
              n_dropout_rounds=100,
              max_learning_iterations=100,
              no_improvement_iterations=15,
              semi_supervised=True, debug=True
              ):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        is_training = False
        dev_batches_with_lab = utils.create_batches(len(dev_set_y), self.batch_size, shuffle=False)
        dev_batches_without_lab = utils.create_batches(len(dev_set_without_lab), self.batch_size, shuffle=False)
        test_batches = utils.create_batches(len(test_set_y), self.batch_size, shuffle=False)
        labeled_training_rounds = math.ceil(float(len(train_set_without_lab)) / float(len(train_set_with_lab)))
        labeled_dev_rounds = math.ceil(float(len(dev_set_without_lab)) / float(len(dev_set_with_lab)))

        # train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
        #                                    sess.graph)
        warm_up = 0
        min_alpha = 0.001  #
        best_print_ana_ppx = 1e10
        no_improvement_iters = 0
        stopped = False
        epoch = -1

        while not stopped:
            epoch += 1
            train_batches_with_lab = utils.create_batches(len(train_set_with_lab), self.batch_size, shuffle=True)
            train_batches_without_lab = utils.create_batches(len(train_set_without_lab), self.batch_size, shuffle=True)
            if warm_up < 1.:
                warm_up += 1. / warm_up_period
            else:
                warm_up = 1.

            optim = self.optim_all

            self.run_model(labeled_training_rounds,
                      train_batches_with_lab,
                      train_set_with_lab, train_set_y, train_batches_without_lab,
                      train_set_without_lab,debug, semi_supervised, epoch,
                        warm_up, min_alpha, sess, optim=optim, keep_prop=0.75, print_statement="training",training=True
                      )


            print_ana_loss, print_loss1, f1_measure,_=self.run_model(labeled_dev_rounds,dev_batches_with_lab,
                      dev_set_with_lab, dev_set_y, dev_batches_without_lab,
                      dev_set_without_lab,  debug, semi_supervised, epoch,
                      warm_up, min_alpha, sess, optim=None, keep_prop=1.0, print_statement="dev",training=is_training
                      )

            if print_ana_loss < best_print_ana_ppx :
                no_improvement_iters = 0
                best_print_ana_ppx = print_ana_loss
                # check on validation set, if ppx better-> save improved model

                #tf.train.Saver().save(sess, model_name + '/improved_model')


            else:
                no_improvement_iters += 1
                print("No improvement: ", no_improvement_iters)
                if no_improvement_iters >= no_improvement_iterations:
                    break

            # -------------------------------
            # test
            if debug:
                self.run_model(1, test_batches,test_set, test_set_y, [],None,  debug, semi_supervised, epoch,
                                                                    warm_up, min_alpha, sess,
                                                                    optim=None, keep_prop=1.0, print_statement="TEST",training=is_training)


        # print("load best dev f1 model...")
        #tf.train.Saver().restore(sess, model_name + '/improved_model')

        _,_,f1_measure,test_pred,=self.run_model(1, test_batches, test_set, test_set_y, [], None, debug, semi_supervised, epoch,
                   warm_up, min_alpha, sess,
                  optim=None, keep_prop=1.0, print_statement="test",training=is_training)

        data_batch, mask = utils.fetch_data_without_idx_new(to_label, self.vocab_size)
        data_batch_y = utils.fetch_data_y_dummy(to_label, self.n_clss,0)
        input_feed = {self.x.name: data_batch, self.y.name: data_batch_y, self.mask.name: mask,
                      self.keep_prob.name: 0.75,
                      self.warm_up.name: warm_up, self.min_alpha.name: min_alpha,
                      self.lab: np.zeros((1)),self.idx.name:np.zeros((1),dtype=np.int32),self.training.name:is_training}

        prediction = [sess.run(([self.out_y]), input_feed) for _ in range(n_dropout_rounds)]
        return test_pred,f1_measure, prediction
