# Audrey Olivier
# 4/16/2019

# This code provides algorithms to estimate uncertainties within neural networks for 1D regression
# (meaning the output is 1-dimensional). The aleatoric noise is assumed to be homoscedastic and known, see input
# var_n to all classes.

import numpy as np
from functools import partial
from scipy.misc import logsumexp
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Dropout
from keras.optimizers import Adam, SGD
from keras.initializers import RandomNormal
from keras.regularizers import l2

from .nn_utils import *
from .general_utils import *
from .mcmc_utils import *

import tensorflow as tf


class Regressor:

    def __init__(self, units_per_layer, prior, pre_model=None, input_dim=None, output_dim=1, var_n=1e-6,
                 activation='tanh'):

        # check the aleatoric noise term and define the loglikelihood term
        self.var_n = var_n
        if isinstance(var_n, np.ndarray): # var_n is provided as a full or diagonal covariance (ndarray)
            if len(var_n) != output_dim:
                raise ValueError('The size of the var_n matrix should be the same as output_dim.')
            if len(var_n.shape) == 1:
                self.var_n = np.diag(var_n)
            self.inv_cov_n = K.constant(np.linalg.inv(self.var_n + 1e-8 * np.eye(output_dim)), dtype='float32')
            self.negloglike = partial(homoscedastic_negloglike_full_covariance, inv_cov_n=self.inv_cov_n)
        elif isinstance(self.var_n, float) or isinstance(self.var_n, int):
            self.negloglike = partial(homoscedastic_negloglike_scalar_variance, var_n=self.var_n)
        # check the prior
        self.prior = preprocess_prior(prior=prior, n_layers=len(units_per_layer) + 1)
        # check input and output dims
        self.output_dim = output_dim
        if input_dim is None and pre_model is None:
            raise ValueError('Either a pre_model or input_dim must be provided.')
        if pre_model is not None:
            input_dim = pre_model.output.get_shape().as_list()[1]
        self.kernel_shapes, self.bias_shapes = compute_weights_shapes(input_dim=input_dim,
                                                                      units_per_layer=units_per_layer,
                                                                      output_dim=output_dim)
        self.glorot_var = compute_glorot_normal_variance(self.kernel_shapes)

    def loglike_cost(self, y_true, y_pred):
        # this is the cost function
        return tf.reduce_sum(self.negloglike(y_true, y_pred), axis=None)


class MAPRegressor(Regressor):

    """ Parent class to do some checks """

    def __init__(self, units_per_layer, prior, pre_model=None, input_dim=None, output_dim=1, var_n=1e-6,
                 activation='tanh'):

        super().__init__(units_per_layer=units_per_layer, prior=prior, pre_model=pre_model, input_dim=input_dim,
                         output_dim=output_dim, var_n=var_n, activation=activation)

        # For this simple regressor, just use dense layers
        if pre_model is None:
            inputs = Input(shape=(input_dim,))
            x = inputs
        else:
            inputs = pre_model.input
            x = pre_model.output
        for l, units in enumerate(units_per_layer + (output_dim,)):
            prior_layer = dict([(key, val[l]) for key, val in prior.items()])
            prior_layer['type'] = prior['type']
            x = Dense(units, activation=activation, name='UQ_layer_{}'.format(l),
                      kernel_regularizer=prior_reg(prior=prior_layer),
                      bias_regularizer=prior_reg(prior=prior_layer),
                      )(x)
        predictions = Dense(units=output_dim, activation=None,
                            kernel_regularizer=l2_reg(weight_regularizer=1 / (2. * prior['variance'][-1]))
                            )(x)
        model = Model(inputs=inputs, outputs=predictions)
        self.estimator = model
        self.ind_layers_UQ = [l for l in range(len(model.layers))
                              if ('uq_layer' in model.layers[l].get_config()['name'].lower())]

    def fit(self, X, y, epochs=100, lr=0.001):
        # always use the full batch
        batch_size = X.shape[0]
        # compile model
        optimizer = Adam(lr=lr)
        self.estimator.compile(loss=self.loglike_cost, optimizer=optimizer)
        # fit to training data
        self.estimator.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.estimator.predict(X).reshape((X.shape[0], self.output_dim))

    def predict_from_prior(self, X, ns=25):
        weights = self.estimator.get_weights()
        y_predictions = np.empty(shape=(X.shape[0], self.output_dim, ns))
        kernels, biases = sample_weights_for_all_layers(self.kernel_shapes, self.bias_shapes, self.prior, ns=ns)
        for n in range(ns):
            # sample from prior, set those weights and predict
            for l, ind_l in enumerate(self.ind_layers_UQ):
                self.estimator.layers[ind_l].set_weights([kernels[l][n], biases[l][n]])
            y_predictions[:, :, n] = self.predict(X)
        # reset the weights to what they were before calling this function
        self.estimator.set_weights(weights)
        return y_predictions


class MCdropoutRegressor(Regressor):

    """ Class that performs MC dropout """

    def __init__(self, units_per_layer, prior, pre_model=None, input_dim=None, output_dim=1,
                 var_n=1e-6, activation='tanh',
                 dropout_rate=None, concrete_dropout=False, dropout_rate_crossval=None, k_crossval=3,
                 ):

        """
        :param concrete_dropout
        :param dropout_rate: use same dropout rate for the whole network
        """
        super().__init__(units_per_layer=units_per_layer, prior=prior, pre_model=pre_model, input_dim=input_dim,
                         output_dim=output_dim, var_n=var_n, activation=activation)
        # check inputs
        if prior['type'].lower() != 'gaussian':
            raise ValueError('For MC dropout, the prior must be gaussian.')
        if concrete_dropout is False and dropout_rate is None and dropout_rate_crossval is None:
            raise ValueError('A dropout_rate, dropout_rate_crossval must be given or concrete_dropout set to True.')
        if concrete_dropout is not None and dropout_rate_crossval is not None:
            raise ValueError('Use either concrete dropout or cross validation, not both.')
        if (concrete_dropout or dropout_rate_crossval is not None) and (dropout_rate is not None):
            print('Warning: using concrete dropout or cross-validation, provided dropout_rate is ignored')
        self.concrete_dropout = concrete_dropout
        self.dropout_rate = dropout_rate
        if self.dropout_rate_crossval is not None:
            self.dropout_rate = dropout_rate_crossval[0]
        self.dropout_rate_crossval = dropout_rate_crossval
        self.k_crossval = k_crossval

        # pre-process the training data
        if pre_model is None:
            inputs = Input(shape=(input_dim,))
            x = inputs
        else:
            inputs = pre_model.input
            x = pre_model.output
        for l, units in enumerate(units_per_layer):
            if concrete_dropout:
                layer_ = Dense(units, activation=activation)
                x = ConcreteDropout(layer_,
                                    weight_regularizer=1 / (2. * prior['variance'][l]),
                                    dropout_regularizer=1.,
                                    name='Concrete_Dropout_{}'.format(l)
                                    )(x)
            else:
                x = Dense(units, activation=activation,
                          kernel_regularizer=l2_reg(weight_regularizer=(1-dropout_rate) / (2. * prior['variance'][l]))
                          )(x)
                x = Dropout(rate=dropout_rate)(x, training=True)
        predictions = Dense(units=output_dim, activation=None,
                            kernel_regularizer=l2_reg(weight_regularizer=1 / (2. * prior['variance'][-1]))
                            )(x)
        model = Model(inputs=inputs, outputs=predictions)
        self.estimator = model

    def fit(self, X, y, epochs=100, lr=0.001):
        # always use the full batch
        batch_size = X.shape[0]
        # compile the model
        optimizer = Adam(lr=lr)
        self.estimator.compile(loss=self.loglike_cost, optimizer=optimizer)
        # fit the model to data
        if self.concrete_dropout:  # do concrete dropout
            dropout_history = DropoutRateHistory()
            train_history = self.estimator.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0,
                                               callbacks=[dropout_history])
            self.loss_history = train_history.history['loss']
            self.dropout_history = np.array(dropout_history.dropout_rates, dtype=float) #rows are epochs, columns are layers
        elif self.dropout_rate_crossval is not None:  # do cross-validation
            loglike_validation = [0]*len(self.dropout_rate_crossval)
            all_inds = np.random.choice(batch_size, replace=False, size=batch_size)
            W = self.estimator.get_weights()
            for d, dropout_rate in enumerate(self.dropout_rate_crossval): # loop over all the parameter values
                for j in range(self.k_crossval):
                    # split the data
                    if j == self.k_crossval-1:
                        val_inds = all_inds[int(j * batch_size / self.k_crossval):]
                    else:
                        val_inds = all_inds[int(j*batch_size/self.k_crossval):int((j+1)*batch_size/self.k_crossval)]
                    # return to untrained configuration and train on new data
                    train_inds = np.setdiff1d(all_inds, val_inds)
                    self.estimator.set_weights(W)
                    dropout_setup = DropoutRateSetup(dropout_rate=dropout_rate)
                    self.estimator.fit(X[train_inds], y[train_inds], epochs=epochs, batch_size=batch_size, verbose=0,
                                       callbacks=[dropout_setup])
                    # evaluate on remaining data
                    loglike = self.predict_loglike(X[val_inds], y[val_inds])
                    loglike_validation[d] = loglike_validation[d] + loglike / self.k_crossval
            self.loglike_validation = loglike_validation
            self.dropout_rate = self.dropout_rate_crossval[np.argmax(loglike_validation)]
            # re-train on all available data
            self.estimator.set_weights(W)
            dropout_setup = DropoutRateSetup(dropout_rate=self.dropout_rate)
            train_history = self.estimator.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0,
                                               callbacks=[dropout_setup])
            self.loss_history = train_history.history['loss']
        else:
            train_history = self.estimator.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
            self.loss_history = train_history.history['loss']

    def predict_UQ(self, X, ns=100, output_MC=False):
        # predict for unseen X, mean and standard deviations
        pred_MC = np.zeros((X.shape[0], self.output_dim, ns))
        for j in range(ns):
            pred_MC[:, :, j] = self.estimator.predict(X).reshape((X.shape[0], self.output_dim))
        estimated_mean = np.mean(pred_MC, axis=2)
        aleatoric_noise = self.var_n
        if isinstance(aleatoric_noise, np.ndarray):
            aleatoric_noise = np.diag(aleatoric_noise).reshape((1, self.output_dim))
        estimated_var = aleatoric_noise + np.var(pred_MC, axis=2)
        if output_MC:
            return estimated_mean, np.sqrt(estimated_var), pred_MC
        return estimated_mean, np.sqrt(estimated_var), None

    def predict_loglike(self, X, y, ns=100):
        # compute predictive log-likelihood of the data
        pred_MC = np.zeros((X.shape[0], ns))
        for j in range(ns):
            pred_MC[:, j] = self.estimator.predict(X).reshape((-1,))
        return logsumexp(- (y.reshape((X.shape[0], 1)) - pred_MC) ** 2 / (2 * self.var_n), axis=1) \
               - np.log(ns) - 1/2 * np.log(2 * np.pi * self.var_n)


class BayesByEnsemble(Regressor):

    def __init__(self, units_per_layer, prior, pre_model=None, input_dim=None, output_dim=1,
                 var_n=1e-6, activation='tanh', n_estimators=1, prior_sampling=False):

        # check the inputs
        super().__init__(units_per_layer=units_per_layer, prior=prior, pre_model=pre_model, input_dim=input_dim,
                         output_dim=output_dim, var_n=var_n, activation=activation)

        self.n_estimators = n_estimators
        if prior_sampling:
            kernels, biases = sample_weights_for_all_layers(self.kernel_shapes, self.bias_shapes, self.prior,
                                                            ns=self.n_estimators)

        self.estimators = []
        for n in range(n_estimators):
            if pre_model is None:
                inputs = Input(shape=(input_dim,))
                x = inputs
            else:
                inputs = pre_model.input
                x = pre_model.output

            for l, units in enumerate(units_per_layer + (output_dim,)):
                # sample from the prior
                prior_layer = dict([(key, val[l]) for key, val in prior.items()])
                prior_layer['type'] = prior['type']
                if prior_sampling:
                    kernel_regularizer = prior_reg_with_initial(prior=prior_layer, initial_weights=kernels[l][n])
                    bias_regularizer = prior_reg_with_initial(prior=prior_layer, initial_weights=biases[l][n])
                    kernel_initializer = Constant(kernels[l][n])
                    bias_initializer = Constant(biases[l][n])
                else:
                    kernel_regularizer = prior_reg(prior=prior_layer)
                    bias_regularizer = prior_reg(prior=prior_layer)
                    kernel_initializer = 'glorot_normal'
                    bias_initializer = 'glorot_normal'
                if l == len(units_per_layer):
                    activation_layer = None
                else:
                    activation_layer = activation
                x = Dense(units, activation=activation_layer,
                          kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                          kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer
                          )(x)
            model = Model(inputs=inputs, outputs=x)
            self.estimators.append(model)

    def fit(self, X, y, epochs=100, lr=0.001, verbose=0, bootstrap_sampling=False, parametric_sampling=False):
        # fit all predictors, possibly on various samples of the data
        self.loss_history = []
        for estimator in self.estimators:
            # compile the model
            optimizer = Adam(lr=lr)
            estimator.compile(loss=self.loglike_cost, optimizer=optimizer)
            # resample the data ?
            if bootstrap_sampling:
                indices_to_keep = np.random.choice(X.shape[0], size=(X.shape[0], ), replace=True)
                X = X[indices_to_keep, :]
                y = y[indices_to_keep, :]
            if parametric_sampling:
                y = y + np.sqrt(self.var_n) * np.random.normal(loc=0., scale=1., size=y.shape)
            batch_size = X.shape[0]
            train_history = estimator.fit(X, y, epochs=epochs, verbose=verbose, batch_size=batch_size)
            self.loss_history.append(train_history.history['loss'])
        self.loss_history = np.array(self.loss_history)

    def predict_UQ(self, X, output_MC=False):
        # predict for unseen X, mean and standard deviations
        pred_MC = np.zeros((X.shape[0], self.output_dim, self.n_estimators))
        for n, estimator in enumerate(self.estimators):
            # run the model to get a prediction for each estimator
            pred_MC[:, :, n] = estimator.predict(X).reshape((X.shape[0], self.output_dim))
        estimated_mean = np.mean(pred_MC, axis=2)
        aleatoric_noise = self.var_n
        if isinstance(aleatoric_noise, np.ndarray):
            aleatoric_noise = np.diag(aleatoric_noise).reshape((1, self.output_dim))
        estimated_var = aleatoric_noise + np.var(pred_MC, axis=2)
        if output_MC:
            return estimated_mean, np.sqrt(estimated_var), pred_MC
        return estimated_mean, np.sqrt(estimated_var), None


class BayesByBackprop(Regressor):
    """ BayesByBackprop algorithm, from 'Weight Uncertainty in Neural Networks', Blundell et al., 2015. """

    def __init__(self, units_per_layer, prior, pre_model=None, input_dim=None, output_dim=1,
                 var_n=1e-6, activation='tanh', posterior_type='gaussian', n_weights_to_track=None):
        """ Initialize the network and define the cost function. """

        tf.reset_default_graph()

        # Do the initial checks and computations for the network
        super().__init__(units_per_layer=units_per_layer, prior=prior, pre_model=pre_model, input_dim=input_dim,
                         output_dim=output_dim, var_n=var_n, activation=activation)
        if n_weights_to_track is None:  # do not save any weights
            n_weights_to_track = [0] * (len(units_per_layer) + 1)
        elif isinstance(n_weights_to_track, int):
            n_weights_to_track = [n_weights_to_track] * (len(units_per_layer) + 1)
        if (not isinstance(n_weights_to_track, list)) or (len(n_weights_to_track) != len(units_per_layer) + 1):
            raise TypeError('Input weights_to_track should be a list of length len(units_per_layer)+1 or an integer.')
        self.n_weights_to_track = n_weights_to_track

        # Initialize necessary variables and placeholders
        X_ = tf.placeholder(dtype=tf.float32, name='X_', shape=(None, input_dim))  # input data
        y_ = tf.placeholder(dtype=tf.float32, name='y_', shape=(None, output_dim))  # output data
        ns_ = tf.placeholder(dtype=tf.int32, name='ns_', shape=())  # number of samples in MC approximation of cost
        ndata = tf.shape(X_)[0]  # number of independent data points
        self.cost_prior = 0.  # contribution of prior to cost: -log(p(w))
        self.cost_var_post = 0.  # contribution of variational posterior: log(q_{theta}(w))
        self.means_ = []
        self.stds_ = []

        x = X_
        if pre_model is not None:
            for layer in pre_model.layers[1:]:  # layer 0 will be an input layer
                x = layer(x)
        x = tf.tile(tf.expand_dims(x, 0), [ns_, 1, 1])
        # add dense layers, add contributions of each layer to prior and variational posterior costs
        for l, units in enumerate(units_per_layer + (output_dim, )):
            prior_layer = dict([(key, val[l]) for key, val in prior.items()])
            prior_layer['type'] = prior['type']

            # Define the parameters of the variational distribution to be trained: theta={mu, rho} for kernel and bias
            mu_kernel = tf.Variable(
                tf.random_normal(shape=self.kernel_shapes[l], mean=0., stddev=0.1),
                name='mu_kernel_layer_{}'.format(l), trainable=True, dtype=tf.float32)
            rho_kernel = tf.Variable(
                -6 * tf.ones(shape=self.kernel_shapes[l], dtype=tf.float32),
                name='rho_kernel_layer_{}'.format(l), trainable=True, dtype=tf.float32)
            sigma_kernel = tf.log(1. + tf.exp(rho_kernel))

            mu_bias = tf.Variable(
                tf.random_normal(shape=self.bias_shapes[l], mean=0., stddev=0.1),
                name='mu_bias_layer_{}'.format(l), trainable=True, dtype=tf.float32)
            rho_bias = tf.Variable(
                -6 * tf.ones(shape=self.bias_shapes[l], dtype=tf.float32),
                name='rho_bias_layer_{}'.format(l), trainable=True, dtype=tf.float32)
            sigma_bias = tf.log(1. + tf.exp(rho_bias))

            # Keep track of some of the weights
            tmp = tf.reshape(mu_kernel[:, :], shape=(-1,))
            self.means_.extend(tmp[i] for i in range(n_weights_to_track[l]))
            tmp = tf.reshape(sigma_kernel[:, :], shape=(-1,))
            self.stds_.extend(tmp[i] for i in range(n_weights_to_track[l]))

            # Samples ns_ biases and kernels from the variational distribution q_{theta}(w)
            kernel = tf.add(tf.tile(tf.expand_dims(mu_kernel, 0), [ns_, 1, 1]),
                            tf.multiply(tf.tile(tf.expand_dims(sigma_kernel, 0), [ns_, 1, 1]),
                                        tf.random_normal(shape=(ns_,) + self.kernel_shapes[l], mean=0., stddev=1.))
                            )
            bias = tf.add(tf.tile(tf.expand_dims(mu_bias, 0), [ns_, 1]),
                          tf.multiply(tf.tile(tf.expand_dims(sigma_bias, 0), [ns_, 1]),
                                      tf.random_normal(shape=(ns_,) + self.bias_shapes[l], mean=0., stddev=1.))
                          )

            # Compute activation(W X + b) for this layer, with w={W, b} sampled from q_{theta}(w)
            x = tf.add(tf.matmul(x, kernel),
                       tf.tile(tf.expand_dims(bias, 1), [1, ndata, 1]))
            assert len(x.get_shape().as_list()) == 3
            if l < len(units_per_layer):
                if activation is 'tanh':
                    x = tf.nn.tanh(x)
                elif activation is 'relu':
                    x = tf.nn.relu(x)

            # Compute the cost from the prior term -log(p(w))
            if prior['type'].lower() == 'gaussian':
                kwargs = {'mu': 0., 'std': np.sqrt(prior['variance'][l]).astype(np.float32)}
                self.cost_prior -= log_gaussian(x=kernel, **kwargs)
                self.cost_prior -= log_gaussian(x=bias, **kwargs)
            elif prior['type'].lower() == 'gaussian_mixture':
                kwargs = {'mu1': 0., 'mu2': 0., 'std1': np.sqrt(prior['variance_1'][l].astype(np.float32)),
                          'std2': np.sqrt(prior['variance_2'][l]).astype(np.float32),
                          'pi1': prior['proba_1'][l].astype(np.float32)}
                self.cost_prior -= log_gaussian_mixture(x=kernel, **kwargs)
                self.cost_prior -= log_gaussian_mixture(x=bias, **kwargs)
            else:
                raise ValueError('Prior must be gaussian or gaussian_mixture.')

            # Compute the cost from the variational distribution term log(q_{theta}(w))
            if posterior_type.lower() == 'gaussian':
                self.cost_var_post += log_gaussian(x=kernel,
                                                   mu=tf.tile(tf.expand_dims(mu_kernel, axis=0), [ns_, 1, 1]),
                                                   std=tf.tile(tf.expand_dims(sigma_kernel, axis=0), [ns_, 1, 1]))
                self.cost_var_post += log_gaussian(x=bias,
                                                   mu=tf.tile(tf.expand_dims(mu_bias, axis=0), [ns_, 1]),
                                                   std=tf.tile(tf.expand_dims(sigma_bias, axis=0), [ns_, 1]))
            else:
                raise ValueError('Variational posterior must be gaussian.')

        # Compute contribution of likelihood to cost: - log p(data|w)
        self.predictions = x
        assert len(self.predictions.get_shape().as_list()) == 3
        self.cost_likelihood = self.loglike_cost(tf.tile(tf.expand_dims(y_, 0), [ns_, 1, 1]), self.predictions)

        # all costs have been summed over ns_ samples, take the average
        self.cost_likelihood /= tf.to_float(ns_)
        self.cost_prior /= tf.to_float(ns_)
        self.cost_var_post /= tf.to_float(ns_)

        # the global cost is the sum of all three costs
        self.cost = self.cost_prior + self.cost_likelihood + self.cost_var_post

    def fit(self, X, y, epochs=100, ns=10, verbose=0, lr=0.001):
        """ Fit the network to data, i.e., learn the parameters theta of the variational distribution. """

        # Set-up the training procedure
        #print(tf.trainable_variables())
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        total_grads_and_vars = opt.compute_gradients(self.cost, tf.trainable_variables())
        grad_step = opt.apply_gradients(total_grads_and_vars)

        # Initilize tensorflow session (the same will be used later for ) and required variables
        X_ = tf.get_default_graph().get_tensor_by_name('X_:0')
        y_ = tf.get_default_graph().get_tensor_by_name('y_:0')
        ns_ = tf.get_default_graph().get_tensor_by_name('ns_:0')
        init_op = tf.global_variables_initializer()
        loss_history, means_history, stds_history = [], [], []
        self.sess = tf.Session()
        self.sess.run(init_op)

        # Run training loop
        for e in range(epochs):
            # apply the gradient descent step
            self.sess.run(grad_step, feed_dict={X_: X, y_: y, ns_: ns})
            # save the loss
            loss_history_ = self.sess.run([self.cost, self.cost_var_post, self.cost_prior, self.cost_likelihood],
                                          feed_dict={X_: X, y_: y, ns_: ns})
            loss_history.append(loss_history_)
            # save some of the weights
            if any(self.n_weights_to_track) != 0:
                mean, std = self.sess.run([self.means_, self.stds_], feed_dict={X_: X, y_: y, ns_: ns})
                means_history.append(mean)
                stds_history.append(std)
            # print comments on terminal
            if verbose:
                print('epoch = {}, loss = {}'.format(e, loss_history[-1][-1]))

        # self.loss_history is a matrix of size (epochs, 4)
        self.loss_history = np.array(loss_history)
        # self.theta_history is a matrix of size (epochs, sum(self.n_weights_to_track), 2)
        if any(self.n_weights_to_track) != 0:
            self.theta_history = np.stack([np.array(means_history), np.array(stds_history)], axis=-1)
        else:
            self.theta_history = None

    def predict_UQ(self, X, ns=100, output_MC=False):
        """ Predict output and uncertainty for new input data. """

        X_ = tf.get_default_graph().get_tensor_by_name('X_:0')
        ns_ = tf.get_default_graph().get_tensor_by_name('ns_:0')
        # pred_MC is a ndarray of shape (ns, ndata, ny)
        pred_MC = self.sess.run(self.predictions, feed_dict={X_: X, ns_: ns})

        # get the mean as average over all predictions
        estimated_mean = np.mean(pred_MC, axis=0)

        # get the aleatoric + epistemic uncertainty (variance) in each dimension of the output
        aleatoric_noise = self.var_n
        if isinstance(aleatoric_noise, np.ndarray):
            aleatoric_noise = np.diag(aleatoric_noise).reshape((1, self.output_dim))
        estimated_var = aleatoric_noise + np.var(pred_MC, axis=0)

        # return all the posterior runs or just the mean and std
        if output_MC:
            return estimated_mean, np.sqrt(estimated_var), pred_MC
        return estimated_mean, np.sqrt(estimated_var), None


class alphaBB(Regressor):
    """alpha-BlackBox algorithm, from 'Black-Box α-Divergence Minimization', Hernández-Lobato, 2016"""

    def __init__(self, units_per_layer, prior, alpha, pre_model=None, input_dim=None, output_dim=1,
                 var_n=1e-6, activation='tanh', posterior_type='gaussian', n_weights_to_track=None):
        """ Initialize the network and define the cost function. """

        tf.reset_default_graph()

        # Do the initial checks and computations for the network
        super().__init__(units_per_layer=units_per_layer, prior=prior, pre_model=pre_model, input_dim=input_dim,
                         output_dim=output_dim, var_n=var_n, activation=activation)
        self.alpha = alpha
        if self.alpha <= 0. or self.alpha >= 1.:
            raise ValueError('Input alpha must be between 0 and 1')
        if n_weights_to_track is None:  # do not save any weights
            n_weights_to_track = [0] * (len(units_per_layer) + 1)
        elif isinstance(n_weights_to_track, int):
            n_weights_to_track = [n_weights_to_track] * (len(units_per_layer) + 1)
        if (not isinstance(n_weights_to_track, list)) or (len(n_weights_to_track) != len(units_per_layer) + 1):
            raise TypeError('Input weights_to_track should be a list of length len(units_per_layer)+1 or an integer.')
        self.n_weights_to_track = n_weights_to_track

        # Initialize necessary variables and placeholders
        X_ = tf.placeholder(dtype=tf.float32, name='X_', shape=(None, input_dim))  # input data
        y_ = tf.placeholder(dtype=tf.float32, name='y_', shape=(None, output_dim))  # output data
        ns_ = tf.placeholder(dtype=tf.int32, name='ns_', shape=())  # number of samples in MC approximation of cost
        ndata = tf.shape(X_)[0]  # number of independent data points
        self.normalization_term = 0.  # contribution of normalization terms to cost: -log(Z(prior))-log(Z(q))
        self.log_factor = 0.  # log of the site approximations f: log(f(w))=s(w).T lambda_{f}
        self.means_ = []
        self.stds_ = []

        x = X_
        if pre_model is not None:
            for layer in pre_model.layers[1:]:
                x = layer(x)
        x = tf.tile(tf.expand_dims(x, 0), [ns_, 1, 1])
        # add dense layers and compute normalization term of the cost for all layers
        for l, units in enumerate(units_per_layer + (output_dim,)):
            prior_layer = dict([(key, val[l]) for key, val in prior.items()])
            prior_layer['type'] = prior['type']

            # Define the parameters of the variational distribution to be trained: theta={mu, rho} for kernel and bias
            mu_kernel = tf.Variable(
                    tf.random_normal(shape=self.kernel_shapes[l], mean=0., stddev=0.1),
                    name='mu_kernel_layer_{}'.format(l), trainable=True, dtype=tf.float32)
            rho_kernel = tf.Variable(
                    -6 * tf.ones(shape=self.kernel_shapes[l], dtype=tf.float32),
                    name='rho_kernel_layer_{}'.format(l), trainable=True, dtype=tf.float32)
            sigma_kernel = tf.log(1. + tf.exp(rho_kernel))

            mu_bias = tf.Variable(
                    tf.random_normal(shape=self.bias_shapes[l], mean=0., stddev=0.1),
                    name='mu_bias_layer_{}'.format(l), trainable=True, dtype=tf.float32)
            rho_bias = tf.Variable(
                    -6 * tf.ones(shape=self.bias_shapes[l], dtype=tf.float32),
                    name='rho_bias_layer_{}'.format(l), trainable=True, dtype=tf.float32)
            sigma_bias = tf.log(1. + tf.exp(rho_bias))

            # Keep track of some of the weights
            tmp = tf.reshape(mu_kernel[:, :], shape=(-1,))
            self.means_.extend(tmp[i] for i in range(n_weights_to_track[l]))
            tmp = tf.reshape(sigma_kernel[:, :], shape=(-1,))
            self.stds_.extend(tmp[i] for i in range(n_weights_to_track[l]))

            # compute normalization term: -log(Z(prior))-log(Z(q))
            self.normalization_term += tf.reduce_sum(0.5 * tf.log(self.prior['variance'][l])
                                                     - tf.log(sigma_kernel)
                                                     - 0.5 * tf.square(tf.divide(mu_kernel, sigma_kernel)))
            self.normalization_term += tf.reduce_sum(0.5 * tf.log(self.prior['variance'][l])
                                                     - tf.log(sigma_bias)
                                                     - 0.5 * tf.square(tf.divide(mu_bias, sigma_bias)))

            # Samples ns_ biases and kernels from the variational distribution q_{theta}(w)
            kernel = tf.add(tf.tile(tf.expand_dims(mu_kernel, 0), [ns_, 1, 1]),
                            tf.multiply(tf.tile(tf.expand_dims(sigma_kernel, 0), [ns_, 1, 1]),
                                        tf.random_normal(shape=(ns_,) + self.kernel_shapes[l], mean=0., stddev=1.))
                            )
            bias = tf.add(tf.tile(tf.expand_dims(mu_bias, 0), [ns_, 1]),
                          tf.multiply(tf.tile(tf.expand_dims(sigma_bias, 0), [ns_, 1]),
                                      tf.random_normal(shape=(ns_,) + self.bias_shapes[l], mean=0., stddev=1.))
                          )

            # Compute activation(W X + b) for this layer, with w={W, b} sampled from q_{theta}(w)
            x = tf.add(tf.matmul(x, kernel),
                       tf.tile(tf.expand_dims(bias, 1), [1, ndata, 1]))
            assert len(x.get_shape().as_list()) == 3
            if l < len(units_per_layer):
                if activation is 'tanh':
                    x = tf.nn.tanh(x)
                elif activation is 'relu':
                    x = tf.nn.relu(x)

            # compute the factor parameters lambda_f, then the log of the site approximations
            var_f_kernel, mu_over_var_f_kernel = factor_params_gaussian(var_prior=prior['variance'][l],
                                                                        mu_post=mu_kernel,
                                                                        var_post=tf.square(sigma_kernel),
                                                                        ndata=tf.to_float(ndata))
            var_f_bias, mu_over_var_f_bias = factor_params_gaussian(var_prior=prior['variance'][l],
                                                                    mu_post=mu_bias,
                                                                    var_post=tf.square(sigma_bias),
                                                                    ndata=tf.to_float(ndata))
            self.log_factor += tf.reduce_sum(
                tf.subtract(tf.multiply(tf.tile(tf.expand_dims(mu_over_var_f_kernel, 0), [ns_, 1, 1]), kernel),
                            tf.multiply(0.5 / tf.tile(tf.expand_dims(var_f_kernel, 0), [ns_, 1, 1]), tf.square(kernel))
                            ),
                axis=[1, 2])
            self.log_factor += tf.reduce_sum(
                tf.subtract(tf.multiply(tf.tile(tf.expand_dims(mu_over_var_f_bias, 0), [ns_, 1]), bias),
                            tf.multiply(0.5 / tf.tile(tf.expand_dims(var_f_bias, 0), [ns_, 1]), tf.square(bias))
                            ),
                axis=[1])

        # compute cost due to likelihood
        self.predictions = x
        assert len(self.predictions.get_shape().as_list()) == 3
        # tie factors: use same site approximation for all data points
        self.log_factor = tf.tile(tf.expand_dims(self.log_factor, 1), [1, ndata])
        # compute likelihood of all data points n separately
        loglike = -1 * self.negloglike(tf.tile(tf.expand_dims(y_, 0), [ns_, 1, 1]), self.predictions)
        assert len(loglike.get_shape().as_list()) == 2
        # compute log(E_{q}[(like_n(w)/f(w)) ** alpha]) for all data points n,
        # expectation is computed by averaging over the ns_ samples
        logexpectation = tf.add(tf.log(1. / tf.to_float(ns_)),
                                tf.reduce_logsumexp(self.alpha * tf.subtract(
                                    loglike, self.log_factor
                                ), axis=0)
                                )
        # in the cost, sum over all the data points n
        self.cost = self.normalization_term - 1. / self.alpha * tf.reduce_sum(logexpectation)

    def fit(self, X, y, epochs=100, ns=10, verbose=0, lr=0.001):
        """ Fit the network to data, i.e., learn the parameters theta of the variational distribution. """

        # Set-up the training procedure
        #print(tf.trainable_variables())
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        total_grads_and_vars = opt.compute_gradients(self.cost, tf.trainable_variables())
        grad_step = opt.apply_gradients(total_grads_and_vars)

        # Initilize tensorflow session (the same will be used later for ) and required variables
        X_ = tf.get_default_graph().get_tensor_by_name('X_:0')
        y_ = tf.get_default_graph().get_tensor_by_name('y_:0')
        ns_ = tf.get_default_graph().get_tensor_by_name('ns_:0')
        init_op = tf.global_variables_initializer()
        loss_history, means_history, stds_history = [], [], []
        self.sess = tf.Session()
        self.sess.run(init_op)

        # Run training loop
        for e in range(epochs):
            # apply the gradient descent step
            self.sess.run(grad_step, feed_dict={X_: X, y_: y, ns_: ns})
            # save the loss
            loss_history_ = self.sess.run(self.cost, feed_dict={X_: X, y_: y, ns_: ns})
            loss_history.append(loss_history_)
            # save some of the weights
            if any(self.n_weights_to_track) != 0:
                mean, std = self.sess.run([self.means_, self.stds_], feed_dict={X_: X, y_: y, ns_: ns})
                means_history.append(mean)
                stds_history.append(std)
            # print comments on terminal
            if verbose:
                print('epoch = {}, loss = {}'.format(e, loss_history[-1]))

        # self.loss_history is a matrix of size (epochs, )
        self.loss_history = np.array(loss_history)
        # self.theta_history is a matrix of size (epochs, sum(self.n_weights_to_track), 2)
        if any(self.n_weights_to_track) != 0:
            self.theta_history = np.stack([np.array(means_history), np.array(stds_history)], axis=-1)
        else:
            self.theta_history = None

    def predict_UQ(self, X, ns=100, output_MC=False):
        """ Predict output and uncertainty for new input data. """

        X_ = tf.get_default_graph().get_tensor_by_name('X_:0')
        ns_ = tf.get_default_graph().get_tensor_by_name('ns_:0')
        # pred_MC is a ndarray of shape (ns, ndata, ny)
        pred_MC = self.sess.run(self.predictions, feed_dict={X_: X, ns_: ns})

        # get the mean as average over all predictions
        estimated_mean = np.mean(pred_MC, axis=0)

        # get the aleatoric + epistemic uncertainty (variance) in each dimension of the output
        aleatoric_noise = self.var_n
        if isinstance(aleatoric_noise, np.ndarray):
            aleatoric_noise = np.diag(aleatoric_noise).reshape((1, self.output_dim))
        estimated_var = aleatoric_noise + np.var(pred_MC, axis=0)

        # return all the posterior runs or just the mean and std
        if output_MC:
            return estimated_mean, np.sqrt(estimated_var), pred_MC
        return estimated_mean, np.sqrt(estimated_var), None


class MCMCRegressor(Regressor):

    def __init__(self, units_per_layer, prior, ns, pre_model=None, input_dim=None, output_dim=1,
                 var_n=1e-6, activation='tanh',
                 algorithm='am', n_chains=1, burnin=0, jump=1):

        super().__init__(units_per_layer=units_per_layer, prior=prior, pre_model=pre_model, input_dim=input_dim,
                         output_dim=output_dim, var_n=var_n, activation=activation)

        self.algorithm = algorithm
        self.n_chains = n_chains
        if algorithm.lower() == 'de-mc' and n_chains < 2:
            raise ValueError('With DE-MC, use at least 2 chains.')
        self.ns = ns
        self.burnin = burnin
        self.jump = jump

        if pre_model is None:
            inputs = Input(shape=(input_dim,))
            x = inputs
        else:
            inputs = pre_model.input
            x = pre_model.output
        for l, units in enumerate(units_per_layer + (output_dim,)):
            # sample from the prior
            prior_layer = dict([(key, val[l]) for key, val in prior.items()])
            prior_layer['type'] = prior['type']
            kernel, bias = sample_weights_for_one_layer(self.kernel_shapes[l], self.bias_shapes[l], prior_layer)
            kernel_regularizer = prior_reg_with_initial(prior=prior_layer, initial_weights=kernel)
            bias_regularizer = prior_reg_with_initial(prior=prior_layer, initial_weights=bias)
            kernel_initializer = Constant(kernel)
            bias_initializer = Constant(bias)
            if l == len(units_per_layer):
                activation_layer = None
            else:
                activation_layer = activation
            x = Dense(units, activation=activation_layer, name='UQ_layer_{}'.format(l),
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer
                      )(x)
        model = Model(inputs=inputs, outputs=x)
        self.estimator = model

        # generate the first vector of weights
        self.ind_layers_UQ = [l for l in range(len(model.layers))
                              if ('uq_layer' in model.layers[l].get_config()['name'].lower())]
        seed = sample_weights_for_all_layers(self.kernel_shapes, self.bias_shapes, prior=self.prior, ns=1,
                                             as_vector=True)
        print(seed.shape)
        self.seed_vector = np.array(seed).reshape((-1,))
        print(self.seed_vector.shape)

    def fit(self, X, y, epochs=100, n_samples=1, verbose=0, lr=0.001):

        # do MCMC
        T = self.burnin + self.ns * self.jump
        if self.algorithm == 'am':
            samples, _, acc = am(log_pdf=partial(self.log_like, X_=X, y_=y), seed=self.seed_vector, T=T, verbose=False)
        print('Acceptance ratio is {}'.format(acc))
        self.samples = post_process_samples(samples, burnin=self.burnin, jump=self.jump, concatenate_chains=True)

    def predict_UQ(self, X, output_MC=False):
        # predict for unseen X, mean and standard deviations
        pred_MC = np.zeros((X.shape[0], self.output_dim, self.ns))
        for n, weight_vector in enumerate(self.samples):
            _ = weights_from_vector_to_layers(weight_vector, nn=self.estimator, ind_layers=self.ind_layers_UQ,
                                              kernel_shapes=self.kernel_shapes,
                                              bias_shapes=self.bias_shapes,
                                              prior=None)
            pred_MC[:, :, n] = self.estimator.predict(X).reshape((X.shape[0], self.output_dim))
        # get the mean as average over all predictions
        estimated_mean = np.mean(pred_MC, axis=-1)
        # get the uncertainty: aleatoric + epistemic
        aleatoric_noise = self.var_n
        if isinstance(aleatoric_noise, np.ndarray):
            aleatoric_noise = np.diag(aleatoric_noise).reshape((1, self.output_dim))
        estimated_var = aleatoric_noise + np.var(pred_MC, axis=2)
        if output_MC:
            return estimated_mean, np.sqrt(estimated_var), pred_MC
        return estimated_mean, np.sqrt(estimated_var), None

    def log_like(self, weight_vector, X_, y_):
        # function that computes the log likelihood of some data given some weights
        log_prior_proba = weights_from_vector_to_layers(weight_vector, nn=self.estimator, ind_layers=self.ind_layers_UQ,
                                                        kernel_shapes=self.kernel_shapes,
                                                        bias_shapes=self.bias_shapes,
                                                        prior=self.prior)
        y_pred = self.estimator.predict(X_)
        return K.eval((-1) * self.loglike_cost(y_true=y_, y_pred=y_pred) + log_prior_proba)

