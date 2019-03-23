from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import utils as utils
from sklearn import metrics

"""NVDM Tensorflow implementation by Yishu Miao(https://github.com/ysmiao/nvdm),
 adapted to work with the Dirichlet distribution by Sophie Burkhardt
 ,adapted to work as semi-supervised classifier by Julia Siekiera"""

#use this just with binary class

class NVDM(object):
    def tower(self, x, y, reuse=False, scope="mlp_tower", labeled=False):
        cat_dis = False
        with tf.variable_scope(scope) as sc:
            if reuse:
                sc.reuse_variables()
            enc_input = tf.concat([x, y], axis=1)
            enc_vec = utils.mlp(enc_input, self.mlp_arr, self.non_linearity,is_training=self.training)

            enc_vec = tf.nn.dropout(enc_vec, self.keep_prob)
            mean = tf.contrib.layers.batch_norm(utils.linear(enc_vec, self.n_topic, scope='mean'),
                                                is_training=self.training)
            alpha = tf.maximum(self.min_alpha, tf.log(1. + tf.exp(mean)))

            prior = tf.ones((self.batch_size, self.n_topic), dtype=tf.float32, name='prior') * self.dir_prior

            analytical_kld = tf.lgamma(tf.reduce_sum(alpha, axis=1)) - tf.lgamma(
                tf.reduce_sum(prior, axis=1))
            analytical_kld -= tf.reduce_sum(tf.lgamma(alpha), axis=1)
            analytical_kld += tf.reduce_sum(tf.lgamma(prior), axis=1)
            minus = alpha - prior

            test = tf.reduce_sum(tf.multiply(minus, tf.digamma(alpha) - tf.reshape(
                tf.digamma(tf.reduce_sum(alpha, 1)), (self.batch_size, 1))), 1)
            analytical_kld += test
            analytical_kld = self.mask * analytical_kld  # mask paddings

            clss_mlp = utils.mlp(x, self.mlp_arr, self.non_linearity, scope="classifier_mlp",is_training=self.training)
            clss_mlp = tf.nn.dropout(clss_mlp, self.keep_prob)
            phi = tf.contrib.layers.batch_norm(utils.linear(clss_mlp, self.n_class, scope='phi'),
                                               is_training=self.training)  # y logits
            out_y = tf.nn.softmax(phi, name="probabilities_y")

            if cat_dis:
                one_hot_dist = tfd.OneHotCategorical(logits=phi)
                hot_out = tf.squeeze(one_hot_dist.sample(1))
                hot_out.set_shape(phi.get_shape())
                cat_input = tf.cast(hot_out, dtype=tf.float32)
            else:
                cat_input = out_y

            # Dirichlet
            doc_vec = tf.squeeze(tfd.Dirichlet(alpha).sample(1))  # tf.shape(self.alpha)
            doc_vec.set_shape(alpha.get_shape())
            # reconstruction
            if labeled:
                merge = tf.concat([doc_vec, y], axis=1)
            else:
                merge = tf.concat([doc_vec, cat_input], axis=1)

            logits = tf.nn.log_softmax(tf.contrib.layers.batch_norm(
                utils.linear(merge, self.vocab_size, scope='projection', no_bias=True), is_training=self.training))

            recons_loss = -tf.reduce_sum(tf.multiply(logits, x), 1)

            L_x_y = recons_loss + self.warm_up * analytical_kld

        return L_x_y, phi, out_y

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
        self.mlp_arr = mlp_arr
        self.dir_prior = dir_prior
        tf.reset_default_graph()
        self.vocab_size = vocab_size
        self.n_hidden = mlp_arr[0]
        self.n_topic = n_topic
        self.non_linearity = non_linearity
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_class = n_class
        self.N = N

        self.x_labeled = tf.placeholder(tf.float32, [None, vocab_size], name='x_labeled')
        self.y_labeled = tf.placeholder(tf.float32, [None, n_class], name='y_labeled')

        self.x_unlabeled = tf.placeholder(tf.float32, [None, vocab_size], name='x_unlabeled')

        # TODO extend this to more than 2 classes...

        self.y_neg = tf.placeholder(tf.float32, [None, n_class], name='y_neg')
        self.y_pos = tf.placeholder(tf.float32, [None, n_class], name='y_pos')

        self.mask = tf.placeholder(tf.float32, [None], name='mask')  # mask paddings
        self.warm_up = tf.placeholder(tf.float32, (), name='warm_up')  # warm up
        self.training = tf.placeholder(tf.bool, (), name="training")
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.min_alpha = tf.placeholder(tf.float32, (), name='min_alpha')

        L_x_y_labeled, phi_labeled, y_labeled = self.tower(self.x_labeled, self.y_labeled, reuse=False, labeled=True)
        L_x_y_neg, self.phi_unlabeled, self.out_y = self.tower(self.x_unlabeled, self.y_neg, reuse=True, labeled=False)
        L_x_y_pos, _, _ = self.tower(self.x_unlabeled, self.y_pos, reuse=True, labeled=False)

        self.clss_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_labeled,
                                                                    logits=phi_labeled) * 0.1 * self.N * self.mask  # use only in labeled case
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.phi_unlabeled,
                                                                        labels=self.out_y) * self.mask  # use only in unlabeled case
        out_y_neg = tf.transpose(
            tf.gather(tf.transpose(self.out_y), indices=tf.constant(0, name="neg", dtype=tf.int32)))
        out_y_pos = tf.transpose(
            tf.gather(tf.transpose(self.out_y), indices=tf.constant(1, name="pos", dtype=tf.int32)))

        self.objective = L_x_y_labeled + self.clss_loss + tf.multiply(L_x_y_neg, out_y_neg) + tf.multiply(L_x_y_pos,
                                                                                                          out_y_pos) + self.cross_entropy
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.adam_beta1,
                                                beta2=self.adam_beta2, epsilon=1E-3).minimize(self.objective)

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
                warm_up_period=100,
                n_dropout_rounds=100,
                max_learning_iterations=100,
                no_improvement_iterations=15,
                semi_supervised=True, debug=True, it=1
                ):

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.it = it
        is_training = False
        dev_batches_with_lab, dev_batches_without_lab = utils.create_batches_new(len(dev_set_y),
                                                                                 len(dev_set_without_lab),
                                                                                 self.batch_size, shuffle=False)

        test_batches = utils.create_batches(len(test_set_y), self.batch_size, shuffle=False)
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
            train_batches_with_lab, train_batches_without_lab = utils.create_batches_new(len(train_set_with_lab),
                                                                                         len(train_set_without_lab),
                                                                                         self.batch_size, shuffle=True)
            if warm_up < 1.:
                warm_up += 1. / warm_up_period
            else:
                warm_up = 1.

            self.run_model(train_batches_with_lab,
                           train_set_with_lab, train_set_y, train_batches_without_lab,
                           train_set_without_lab, debug, semi_supervised, epoch,
                           warm_up, min_alpha, sess, optim=self.optim, keep_prop=0.75,
                           print_statement="training", training=True)

            print_ana_loss,_,_ = self.run_model(dev_batches_with_lab,
                                                        dev_set_with_lab, dev_set_y,
                                                        dev_batches_without_lab,
                                                        dev_set_without_lab, debug, semi_supervised, epoch,
                                                        warm_up, min_alpha, sess,
                                                        optim=None, keep_prop=1.0, print_statement="dev",
                                                        training=is_training)
            if debug:
                _, f1_measure_test,_ = self.run_model(test_batches, test_set, test_set_y, test_batches, test_set,
                                                    debug, semi_supervised, epoch,
                                                    warm_up, min_alpha, sess,
                                                    optim=None, keep_prop=1.0, print_statement="test",
                                                    training=is_training)
                print("TEST F1:", f1_measure_test)

            if print_ana_loss < best_print_ana_ppx:
                no_improvement_iters = 0
                best_print_ana_ppx = print_ana_loss
                #tf.train.Saver().save(sess, model_name + '/improved_model')
            else:
                no_improvement_iters += 1
                print("No improvement: ", no_improvement_iters, "epoch:", epoch)
                if no_improvement_iters >= no_improvement_iterations:
                    break
        # print("load best dev f1 model...")
        #tf.train.Saver().restore(sess, model_name + '/improved_model')

        _, f1_measure,prop_clss = self.run_model(test_batches, test_set, test_set_y, test_batches, test_set,
                                       debug, semi_supervised, epoch,
                                       warm_up, min_alpha, sess,
                                       optim=None, keep_prop=1.0, print_statement="test", training=is_training,
                                       model_name=model_name)

        data_batch_labeled, mask = utils.fetch_data_without_idx_new(to_label, self.vocab_size)
        data_batch_y, data_batch_y_neg, data_batch_y_pos = utils.fetch_data_y_dummy_new(to_label, self.n_class)
        input_feed = {self.x_labeled: data_batch_labeled, self.x_unlabeled: data_batch_labeled,
                      self.y_labeled: data_batch_y, self.y_neg: data_batch_y_neg, self.y_pos: data_batch_y_pos,
                      self.mask.name: mask,
                      self.keep_prob: 0.75, self.warm_up: warm_up, self.min_alpha: min_alpha,
                      self.training.name: is_training}

        prediction = [sess.run(([self.out_y]), input_feed) for _ in range(n_dropout_rounds)]
        return f1_measure[1], prediction,prop_clss

    def run_model(self,
                  train_batches_with_lab,
                  train_set_with_lab, train_set_y, train_batches_without_lab,
                  train_set_without_lab, debug, semi_supervised, epoch,
                  warm_up, min_alpha, sess, optim=None, keep_prop=0.75, print_statement="",
                  training=None, model_name=None
                  ):
        loss_sum1 = 0.0
        clss_loss_sum = 0.0
        cross_entropy_sum1 = 0.0
        pred_clss = []
        true_clss = []
        prop_clss = []

        for idx_batch_labeled, idx_batch_unlabeled in zip(train_batches_with_lab, train_batches_without_lab):
            data_batch_labeled, mask = utils.fetch_data_new(train_set_with_lab, idx_batch_labeled, self.vocab_size)
            data_batch_unlabeled, mask = utils.fetch_data_new(train_set_without_lab, idx_batch_unlabeled,
                                                              self.vocab_size)
            data_batch_y, data_batch_y_neg, data_batch_y_pos = utils.fetch_data_y_new(train_set_y, idx_batch_labeled,
                                                                                      self.n_class)

            input_feed = {self.x_labeled: data_batch_labeled, self.x_unlabeled: data_batch_unlabeled,
                          self.y_labeled: data_batch_y, self.y_neg: data_batch_y_neg, self.y_pos: data_batch_y_pos,
                          self.mask: mask, self.keep_prob.name: keep_prop,
                          self.warm_up.name: warm_up, self.min_alpha.name: min_alpha, self.training.name: training}
            if optim:
                _, (loss, clss_loss, entropy, propability) = sess.run(
                    (optim, [self.objective, self.clss_loss, self.cross_entropy, self.out_y]), input_feed)
            else:
                loss, clss_loss, entropy, propability, y_logits = sess.run(
                    [self.objective, self.clss_loss, self.cross_entropy, self.out_y, self.phi_unlabeled], input_feed)

            mask_sum = np.sum(mask)
            loss_sum1 += np.sum(loss) / mask_sum
            if debug:
                clss_loss_sum += np.sum(clss_loss) / mask_sum
                cross_entropy_sum1 += np.sum(entropy) / mask_sum
            if print_statement == "test":
                preds = []
                prop = []
                j = -1
                for c, i in enumerate(propability):
                    j += 1
                    if idx_batch_labeled[j] != -1:
                        preds.append(np.argmax(i))
                        prop.append(i)

                pred_clss.extend(preds)
                prop_clss.extend(prop)

                true_clss.extend([train_set_y[i] for i in idx_batch_labeled if i != -1])
        f1_measure = None
        if print_statement == "test":
            precision, recall, f1_measure, support = metrics.precision_recall_fscore_support(true_clss, pred_clss)


        len_all = (len(train_batches_with_lab) * 2)
        print_loss1 = loss_sum1 / len_all
        if debug:
            print_c_e1 = cross_entropy_sum1 / len(train_batches_without_lab)
            print_clss_loss = clss_loss_sum / len(train_batches_with_lab)
            print(print_statement, '| Epoch: {:d} |'.format(epoch + 1),
                  '| Loss 1: {:.3f}'.format(print_loss1),
                  '| cross_e.: {:.3f}'.format(print_c_e1),
                  '|clss_sum: {:.3f}'.format(print_clss_loss))

        return print_loss1, f1_measure,prop_clss
