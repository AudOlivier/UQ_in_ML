

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

    def __init__(self, VI_regressor, ns):
        self.VI_regressor = VI_regressor
        if not isinstance(VI_regressor, BayesByBackprop) and not isinstance(VI_regressor, alphaBB):
            raise TypeError('Input UQ_regressor must be a variational inference algorithm.')
        self.ns = ns
        if isinstance(self.ns, int):
            self.ns = [self.ns]

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

            tf_network_weights = [[] for _ in range(2 * n_layers)]
            log_importance_weights = []
            predictions = []
            for ns_ in self.ns:
                # Sample ns_ weights w from the variational distribution
                tmp_network_weights = self.VI_regressor.sample_weights_from_variational(ns=ns_)
                # Compute log(prior(w)) for all ns_ w
                log_prior_pdf = self.VI_regressor.log_prior_pdf(tmp_network_weights, axis_sum=([-2, -1], [-1]))
                assert len(log_prior_pdf.get_shape().as_list()) == 1
                # Compute log(p(D|w)) for all ns_ w
                tmp_predictions = self.VI_regressor.compute_predictions(X_IS, tmp_network_weights, ns_)
                log_likelihood = - self.VI_regressor.loglike_cost(tf.tile(tf.expand_dims(y_IS, 0), [ns_, 1, 1]),
                                                                  tmp_predictions, axis_sum=[-1])
                assert len(log_likelihood.get_shape().as_list()) == 1
                # Compute log(q(w)) for all ns_ w
                log_variational_pdf = self.VI_regressor.log_variational_pdf(tmp_network_weights, ns_,
                                                                            axis_sum=([-2, -1], [-1]))
                assert len(log_variational_pdf.get_shape().as_list()) == 1
                # Compute log importance weights as log(prior(w)*p(D|w)/q(w)) for all ns_ w
                log_importance_weights.append(log_prior_pdf + log_likelihood - log_variational_pdf)
                [tf_network_weights[i].append(tmp_network_weights[i]) for i in range(2 * n_layers)]
                predictions.append(tmp_predictions)
            # Self-normalize the weights so that they sum up to 1
            log_importance_weights = tf.concat(log_importance_weights, axis=0)
            self.tf_log_importance_weights = log_importance_weights - tf.reduce_logsumexp(log_importance_weights)
            self.tf_network_weights = [tf.concat(tf_network_weights[i], axis=0) for i in range(2 * n_layers)]
            self.predictions = tf.concat(predictions, axis=0)

        # Before fitting, the weights are the same for all samples
        self.log_importance_weights = np.log(1 / sum(self.ns)) * np.ones(shape=(sum(self.ns),))
        with tf.Session(graph=self.VI_regressor.graph) as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = dict(zip(self.VI_regressor.variational_variables, self.VI_regressor.variational_params))
            self.network_weights = sess.run(self.tf_network_weights, feed_dict=feed_dict)

    def fit(self, X, y):
        """ Sample network parameters omega and compute importance weights w"""

        with tf.Session(graph=self.VI_regressor.graph) as sess:
            X_IS = self.VI_regressor.graph.get_tensor_by_name('X_IS:0')
            y_IS = self.VI_regressor.graph.get_tensor_by_name('y_IS:0')

            sess.run(tf.global_variables_initializer())

            feed_dict = dict(zip(self.VI_regressor.variational_variables, self.VI_regressor.variational_params))
            feed_dict.update({X_IS: X, y_IS: y})
            self.network_weights, self.log_importance_weights = sess.run(
                [self.tf_network_weights, self.tf_log_importance_weights], feed_dict=feed_dict)

        # save the same weights as saved within the VI regressor
        weights_to_track = []
        for l in range(len(self.VI_regressor.hidden_units) + 1):
            if self.VI_regressor.n_weights_to_track[l] != 0:
                weights_to_track.append(self.network_weights[2 * l].reshape(
                    (self.ns, -1))[:, :self.VI_regressor.n_weights_to_track[l]])
        if len(weights_to_track) != 0:
            weights_to_track = np.concatenate(weights_to_track, axis=1)
            print(weights_to_track.shape)
            print(np.exp(self.log_importance_weights).shape)
            self.posterior_mean_weights = np.average(weights_to_track, axis=0,
                                                     weights=np.exp(self.log_importance_weights))
            self.posterior_cov_weights = np.cov(weights_to_track, rowvar=False,
                                                ddof=0, aweights=np.exp(self.log_importance_weights))
        else:
            self.posterior_mean_weights = None
            self.posterior_cov_weights = None

    def predict_UQ(self, X, return_mean, return_std):
        """ Predict y for new input X, along with uncertainty """

        X_IS = self.VI_regressor.graph.get_tensor_by_name('X_IS:0')

        with tf.Session(graph=self.VI_regressor.graph) as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = dict(zip(self.tf_network_weights, self.network_weights))
            feed_dict.update({X_IS: X})
            y_MC = sess.run(self.predictions, feed_dict=feed_dict)
        y_mean, y_std = mean_and_std_from_samples(y_MC, var_aleatoric=self.VI_regressor.var_n,
                                                  importance_weights=np.exp(self.log_importance_weights))
        return return_outputs(y_mean, y_std, None, return_mean, return_std, False)



def gaussian_loss_adapt(y_true, y_pred):
    log_var_n = y_pred[:, 1]
    return K.mean(log_var_n + (y_true[:, 0]-y_pred[:, 0])**2 * K.exp(-log_var_n))


class BayesByBackprop:

    def __init__(self, units_per_layer, X_train, prior, var_n=1e-6, activation='tanh', posterior_type='gaussian'):

        # check the inputs
        self.var_n = var_n
        if posterior_type.lower() != 'gaussian':
            raise ValueError('Only supported posterior type is gaussian')
        self.prior = preprocess_prior(prior=prior, n_layers=len(units_per_layer)+1)
        self.kernel_shape, self.bias_shape = compute_weights_shapes(input_dim=X_train.shape[1],
                                                                    units_per_layer=units_per_layer, output_dim=1)

        # create the neural network
        n_data, input_dim = X_train.shape
        mean_X_train, std_X_train = np.mean(X_train, axis=0), np.std(X_train, axis=0)
        # create model using Functional API
        inputs = Input(shape=(input_dim,))
        x = Lambda(lambda x_: (x_ - mean_X_train) / std_X_train)(inputs)
        for j, units in enumerate(units_per_layer):
            x = Dense(units, activation=activation)(x)
        predictions = Dense(1)(x)
        model = Model(inputs=inputs, outputs=predictions)

        # create variables for mu and rho
        self.kernel_mus = []
        self.kernel_rhos = []
        self.bias_mus = []
        self.bias_rhos = []
        for l, layer in enumerate(model.layers[2:]):
            self.kernel_mus.append(tf.Variable(name='kernel_mu_{}'.format(l), trainable=True,
                                               initial_value=tf.random_normal(shape=self.kernel_shape[l], mean=0.0, stddev=0.1)))
            self.kernel_rhos.append(tf.Variable(name='kernel_rho_{}'.format(l), trainable=True,
                                                initial_value=tf.random_normal(shape=self.kernel_shape[l], mean=0.0,
                                                                               stddev=0.1)))
            self.bias_mus.append(tf.Variable(name='bias_mu_{}'.format(l), trainable=True,
                                             initial_value=tf.random_normal(shape=self.bias_shape[l], mean=0.0,
                                                                            stddev=0.1)))
            self.bias_rhos.append(tf.Variable(name='bias_rho_{}'.format(l), trainable=True,
                                              initial_value=tf.random_normal(shape=self.bias_shape[l], mean=0.0,
                                                                             stddev=0.1)))
        self.model = model

    def fit(self, X_train, y_train, epochs=100, nsamples=1, lr=0.001, verbose=0):

        cost = 0.
        # the gradient is computed via MC integration over nsamples
        for _ in range(nsamples):
            # set the weights to their new (random) value - get the prior and posterior terms of the cost
            prior_term = 0.
            q_term = 0.
            for l, layer in enumerate(self.model.layers[2:]):
                # sample kernel
                kernel_sigma = tf.log(1. + tf.exp(self.kernel_rhos[l]))
                w = tf.Variable(initial_value=np.ones(self.kernel_shape[l]), dtype='float32')
                w_ = tf.assign(w, tf.add(self.kernel_mus[l],
                                         kernel_sigma * tf.random_normal(shape=self.kernel_shape[l],
                                                                         mean=0.0, stddev=1.0, dtype='float32')))
                bias_sigma = tf.log(1. + tf.exp(self.bias_rhos[l]))
                b = tf.Variable(initial_value=np.ones(self.bias_shape[l]), dtype='float32')
                b_ = tf.assign(b, tf.add(self.bias_mus[l],
                                         bias_sigma * tf.random_normal(shape=self.bias_shape[l],
                                                                       mean=0.0, stddev=1.0, dtype='float32')))
                #with tf.control_dependencies([w_, b_]):
                prior_term -= tf.reduce_sum(tf.square(w_)) / (2 * self.prior['variance'][l])
                q_term -= tf.reduce_sum(kernel_sigma) + \
                          tf.reduce_sum(tf.square(w_ - self.kernel_mus[l]) / (2 * tf.square(kernel_sigma)))
                # sample bias
                prior_term -= tf.reduce_sum(tf.square(b_)) / (2 * self.prior['variance'][l])
                q_term -= tf.reduce_sum(bias_sigma) + \
                          tf.reduce_sum(tf.square(b_ - self.bias_mus[l]) / (2 * tf.square(bias_sigma)))
                # set the new kernel/bias in the model
                layer.set_weights([K.get_value(w_), K.get_value(b_)])
            # run the model to get a prediction - get the likelihood term of the cost
            pred = self.model.predict(X_train)
            likelihood_term = tf.cast(- tf.reduce_sum(tf.square(pred - y_train)) / (2 * self.var_n), dtype='float32')
            # compute cost
            cost += - likelihood_term # q_term - likelihood_term - prior_term
        cost /= nsamples
        # set-up the optimization procedure
        opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        grads_and_vars = opt.compute_gradients(cost, self.kernel_mus + self.kernel_rhos + self.bias_mus + self.bias_rhos)
        opt_weights = tf.train.GradientDescentOptimizer(learning_rate=lr)
        grads_and_vars = opt.compute_gradients(cost, self.kernel_mus + self.kernel_rhos + self.bias_mus + self.bias_rhos)
        grad_step = opt.apply_gradients(grads_and_vars)

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            loss_history = []
            for e in range(epochs):
                print(e)
                # apply the gradient descent step
                sess.run(grad_step)
                loss_history.append(sess.run(cost))
                if verbose:
                    print('epoch = {}, loss = {}'.format(e, loss_history[-1]))
            self.loss_history = loss_history

    def predict_UQ(self, X, nsamples):

        pred_MC = np.zeros(X.shape[0], nsamples)
        for n in range(nsamples):
            # set the weights to their new (random) value
            for l, layer in enumerate(self.model.layers[2:]):
                # sample kernel
                kernel_sigma = tf.log(1. + tf.exp(self.kernel_rhos[l]))
                w = tf.add(self.kernel_mus[l],
                           kernel_sigma * tf.random_normal(shape=self.kernel_shape[l], mean=0.0, stddev=1.0))
                # sample bias
                bias_sigma = tf.log(1. + tf.exp(self.bias_rhos[l]))
                b = tf.add(self.bias_mus[l],
                           bias_sigma * tf.random_normal(shape=self.bias_shape[l], mean=0.0, stddev=1.0))
                # set the new kernel/bias in the model
                layer.set_weights([w, b])
            # run the model to get a prediction - get the likelihood term of the cost
            pred_MC[:, n] = self.model.predict(X).reshape((-1,))




def prior_reg(weight_matrix, prior):
    if prior['type'].lower() == 'gaussian':
        return K.sum(K.square(weight_matrix)) / (2 * prior['variance'])
    if prior['type'].lower() == 'gaussian_mixture':
        term1 = prior['proba_1'] * (1 / K.sqrt(prior['variance_1']) *
                                    K.exp(K.square(weight_matrix) / (2 * prior['variance_1'])))
        term2 = (1 - prior['proba_1']) * (1 / K.sqrt(prior['variance_2']) *
                                          K.exp(K.square(weight_matrix) / (2 * prior['variance_2'])))
        return K.sum(K.log(term1 + term2))


def prior_reg_with_initial(weight_matrix, prior, initial_weights):
    if prior['type'].lower() == 'gaussian':
        return K.sum(K.square(weight_matrix-initial_weights)) / (2 * prior['variance'])
    if prior['type'].lower() == 'gaussian_mixture':
        term1 = prior['proba_1'] * (1 / K.sqrt(prior['variance_1']) *
                                    K.exp(K.square(weight_matrix-initial_weights) / (2 * prior['variance_1'])))
        term2 = (1 - prior['proba_1']) * (1 / K.sqrt(prior['variance_2']) *
                                          K.exp(K.square(weight_matrix-initial_weights) / (2 * prior['variance_2'])))
        return K.sum(K.log(term1 + term2))



# for bayes by backprop optimisation
#for _ in range(n_samples-1):
#    grads_and_vars = opt.compute_gradients(self.cost, tf.trainable_variables())
#    total_grads_and_vars = [(tgv[0]+gv[0], tgv[1]) for tgv, gv in zip(total_grads_and_vars, grads_and_vars)]
#total_grads_and_vars = [(tgv[0]/n_samples, tgv[1]) for tgv in total_grads_and_vars]