## Audrey Olivier
## Build algorithms that correct uncertainty estimates of epistemic_regressors.py
## These algorithms are themselves regressors

import numpy as np
import tensorflow as tf

from .epistemic_regressors import BayesByBackprop, alphaBB
from .general_utils import *


class IScorrector:
    """
    Correct Variational Inference via Importance Sampling
    It requires to have a UQ_regressor for which you can both sample weights from and evaluate its pdf p_reg(w)
    """

    def __init__(self, VI_regressor, ns, ns_keep=None):
        self.VI_regressor = VI_regressor
        if not isinstance(VI_regressor, BayesByBackprop) and not isinstance(VI_regressor, alphaBB):
            raise TypeError('Input UQ_regressor must be a variational inference algorithm.')
        self.ns = ns
        self.ns_keep = ns_keep

        #if n_batches == 1:
        #    ns_per_batch = [self.ns]
        #else:
        #    n_per_batch = ns // n_batches
        #    ns_per_batch = [n_per_batch] * (n_batches-1) + [self.ns-n_per_batch*(n_batches-1)]

        n_layers = len(self.VI_regressor.hidden_units) + 1
        with self.VI_regressor.graph.as_default():

            # Initialize necessary variables and placeholders
            X_IS = tf.placeholder(dtype=tf.float32, name='X_IS', shape=(None,) + self.VI_regressor.input_dim)  # input data
            y_IS = tf.placeholder(dtype=tf.float32, name='y_IS', shape=(None, self.VI_regressor.output_dim))  # output data

            # Sample ns_ weights w from the variational distribution
            self.tf_network_weights = self.VI_regressor.sample_weights_from_variational(ns=self.ns)
            # Compute log(prior(w)) for all ns_ w
            log_prior_pdf = self.VI_regressor.log_prior_pdf(self.tf_network_weights, axis_sum=([-2, -1], [-1]))
            assert len(log_prior_pdf.get_shape().as_list()) == 1
            # Compute log(p(D|w)) for all ns_ w
            self.predictions = self.VI_regressor.compute_predictions(X_IS, self.tf_network_weights, self.ns)
            log_likelihood = - self.VI_regressor.loglike_cost(tf.tile(tf.expand_dims(y_IS, 0), [self.ns, 1, 1]),
                                                              self.predictions, axis_sum=[-1])
            assert len(log_likelihood.get_shape().as_list()) == 1
            # Compute log(q(w)) for all ns_ w
            log_variational_pdf = self.VI_regressor.log_variational_pdf(self.tf_network_weights, self.ns,
                                                                        axis_sum=([-2, -1], [-1]))
            assert len(log_variational_pdf.get_shape().as_list()) == 1
            # Compute log importance weights as log(prior(w)*p(D|w)/q(w)) for all ns_ w
            self.tf_log_importance_weights = log_prior_pdf + log_likelihood - log_variational_pdf
            # Self-normalize the weights so that they sum up to 1
            #self.tf_log_importance_weights = log_importance_weights - tf.reduce_logsumexp(log_importance_weights)

        # Before fitting, the weights are the same for all samples
        self.log_importance_weights = np.log(1 / self.ns) * np.ones(shape=(self.ns,))
        with tf.Session(graph=self.VI_regressor.graph) as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = dict(zip(self.VI_regressor.variational_variables, self.VI_regressor.variational_params))
            self.network_weights = sess.run(self.tf_network_weights, feed_dict=feed_dict)

    def fit(self, X, y):
        """ Sample network parameters omega and compute importance weights w"""

        # Since we do not reset the graph when we run a corrector, several 'X_IS:n' tensors may exist in that graph:
        # use the last one
        #all_IS_names = [tensor.name for tensor in self.VI_regressor.graph.as_graph_def().node
        #                if ('_IS' in tensor.name)]
        #all_IS_names.sort()
        #X_IS = self.VI_regressor.graph.get_tensor_by_name(all_IS_names[len(all_IS_names) // 2 - 1])
        #y_IS = self.VI_regressor.graph.get_tensor_by_name(all_IS_names[-1])
        X_IS = self.VI_regressor.graph.get_tensor_by_name('X_IS:0')
        y_IS = self.VI_regressor.graph.get_tensor_by_name('y_IS:0')

        with tf.Session(graph=self.VI_regressor.graph) as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = dict(zip(self.VI_regressor.variational_variables, self.VI_regressor.variational_params))
            feed_dict.update({X_IS: X, y_IS: y})
            self.network_weights, self.log_importance_weights = sess.run(
                [self.tf_network_weights, self.tf_log_importance_weights], feed_dict=feed_dict)
        from scipy.misc import logsumexp
        self.log_importance_weights -= np.max(self.log_importance_weights)
        self.log_importance_weights -= logsumexp(self.log_importance_weights)
        print(self.log_importance_weights.shape)
        print(np.exp(logsumexp(self.log_importance_weights)))

        # save the same weights as saved within the VI regressor
        weights_to_track = []
        for l in range(len(self.VI_regressor.hidden_units) + 1):
            if self.VI_regressor.n_weights_to_track[l] != 0:
                weights_to_track.append(self.network_weights[2 * l].reshape(
                    (self.ns, -1))[:, :self.VI_regressor.n_weights_to_track[l]])
        if len(weights_to_track) != 0:
            weights_to_track = np.concatenate(weights_to_track, axis=1)
            self.posterior_mean_weights = np.average(weights_to_track, axis=0,
                                                     weights=np.exp(self.log_importance_weights))
            self.posterior_cov_weights = np.cov(weights_to_track, rowvar=False,
                                                ddof=0, aweights=np.exp(self.log_importance_weights))
        else:
            self.posterior_mean_weights = None
            self.posterior_cov_weights = None

    def predict_UQ(self, X, return_mean, return_std):
        """ Predict y for new input X, along with uncertainty """

        #all_X_IS_names = [tensor.name for tensor in self.VI_regressor.graph.as_graph_def().node
        #                if ('X_IS' in tensor.name)]
        #all_X_IS_names.sort()
        #X_IS = self.VI_regressor.graph.get_tensor_by_name(all_X_IS_names[-1])
        X_IS = self.VI_regressor.graph.get_tensor_by_name('X_IS:0')

        with tf.Session(graph=self.VI_regressor.graph) as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = dict(zip(self.tf_network_weights, self.network_weights))
            feed_dict.update({X_IS: X})
            y_MC = sess.run(self.predictions, feed_dict=feed_dict)
        y_mean, y_std = mean_and_std_from_samples(y_MC, var_aleatoric=self.VI_regressor.var_n,
                                                  importance_weights=np.exp(self.log_importance_weights))
        return return_outputs(y_mean, y_std, None, return_mean, return_std, False)
