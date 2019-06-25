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
from keras.layers import InputLayer, Dense, Lambda, Dropout
from keras.optimizers import Adam, SGD
from keras.initializers import RandomNormal
from keras.regularizers import l2

from .nn_utils import *
from .general_utils import *
from .mcmc_utils import *

import tensorflow as tf


class Regressor:
    """
    Base Regressor class
    This class does some initial checks of the inputs and defines methods pertaining to the prior.
    """

    def __init__(self, hidden_units, output_dim=1, input_dim=1, pre_model=None, input_uq_dim=None,
                 var_n=1e-6, activation='tanh', prior_type='gaussian', prior_params=(0., 1.)):

        self.hidden_units = hidden_units
        self.activations = [activation] * len(self.hidden_units) + ['linear']
        self.pre_model = pre_model
        if self.pre_model is not None:
            if (not isinstance(self.pre_model, Model)) or (not isinstance(self.pre_model.layers[0], InputLayer)):
                raise TypeError('Input pre_model must be a keras model with first layer an InputLayer.')
        # check input and output dims
        self.output_dim = output_dim
        self.input_uq_dim = input_uq_dim
        self.input_dim = input_dim
        if not isinstance(self.output_dim, int):
            raise TypeError('Input output_dim must be an integer.')
        if self.pre_model is not None:
            self.input_dim = tuple(self.pre_model.input.get_shape().as_list()[1:])
            self.input_uq_dim = self.pre_model.output.get_shape().as_list()[1]
        if self.input_uq_dim is None:
            if isinstance(self.input_dim, int):
                self.input_uq_dim = self.input_dim
            elif isinstance(self.input_dim, tuple) and len(self.input_dim) == 1:
                    self.input_uq_dim = self.input_dim[0]
            else:
                raise TypeError('Error in defining input_uq_dim.')
        if isinstance(self.input_dim, int):
            self.input_dim = (self.input_dim, )
        if not isinstance(self.input_dim, tuple):
            raise TypeError('Input input_dim must be an integer or a tuple of integers.')
        if self.input_dim is None or self.output_dim is None or self.input_uq_dim is None:
            raise ValueError('Review the definitions of the input/output dimensions.')

        # check the aleatoric noise term and define the loglikelihood term
        self.var_n = var_n
        if isinstance(var_n, np.ndarray): # var_n is provided as a full or diagonal covariance (ndarray)
            if len(var_n) != self.output_dim:
                raise ValueError('The size of the var_n matrix should be the same as output_dim.')
            if len(var_n.shape) == 1:
                self.var_n = np.diag(var_n)
            self.inv_cov_n = K.constant(np.linalg.inv(self.var_n + 1e-8 * np.eye(self.output_dim)), dtype='float32')
            self.negloglike = partial(homoscedastic_negloglike_full_covariance, inv_cov_n=self.inv_cov_n)
        elif isinstance(self.var_n, float) or isinstance(self.var_n, int):
            self.negloglike = partial(homoscedastic_negloglike_scalar_variance, var_n=self.var_n)
        # check the prior
        self.prior_type = prior_type
        self.prior_params = preprocess_prior(prior_type=self.prior_type, prior_params=prior_params,
                                             n_layers=len(self.hidden_units) + 1)
        print(self.prior_params)
        self.kernel_shapes, self.bias_shapes = compute_weights_shapes(hidden_units=self.hidden_units,
                                                                      input_dim=self.input_uq_dim,
                                                                      output_dim=self.output_dim)
        self.glorot_var = compute_glorot_normal_variance(self.kernel_shapes)

    def loglike_cost(self, y_true, y_pred, axis_sum=None):
        # this is the cost function
        return tf.reduce_sum(self.negloglike(y_true, y_pred), axis=axis_sum)

    def sample_weights_from_prior(self, ns, layer_ind=None):
        if layer_ind is None:
            list_layers = list(range(len(self.hidden_units) + 1))
        else:
            list_layers = [layer_ind]
        weights = []
        for layer in list_layers:
            # sample weights from prior for one layer
            kernel = lhs_pdf(shape=self.kernel_shapes[layer], ns=ns, pdf_type=self.prior_type,
                             pdf_params=self.prior_params[layer])
            bias = lhs_pdf(shape=self.bias_shapes[layer], ns=ns, pdf_type=self.prior_type,
                           pdf_params=self.prior_params[layer])
            weights.extend([kernel, bias])
        return weights

    def build_network_keras(self):
        # Build the prior network in keras
        if self.pre_model is None:
            inputs = Input(shape=self.input_dim)
            x = inputs
        else:
            inputs = self.pre_model.input
            x = self.pre_model.output
        for l, units in enumerate(self.hidden_units + (self.output_dim,)):
            x = Dense(units, activation=self.activations[l], name='uq_layer_{}'.format(l))(x)
        network = Model(inputs=inputs, outputs=x)
        # freeze all layers that do not account for uncertainties
        layers_freezed = [layer for layer in network.layers if ('uq_layer' not in layer.get_config()['name'].lower())]
        for layer in layers_freezed:
            layer.trainable = False
        return network

    def display_network(self):
        self.build_network_keras().summary()

    def predict_UQ_from_prior(self, X, ns=25, return_mean=False, return_std=False, return_MC=True):
        # Predict from the network using the prior pdf over the weights
        prior_model = self.build_network_keras()
        ind_layers_UQ = [l for l in range(len(prior_model.layers))
                         if ('uq_layer' in prior_model.layers[l].get_config()['name'].lower())]
        # sample weights from prior
        network_weights = self.sample_weights_from_prior(ns=ns)
        # set those weights within the network and predict
        y_MC = np.empty(shape=(ns, X.shape[0], self.output_dim))
        for n in range(ns):
            # sample from prior, set those weights and predict
            for l, model_layer in enumerate(ind_layers_UQ):
                kernel, bias = network_weights[2 * l][n], network_weights[2 * l + 1][n]
                prior_model.layers[model_layer].set_weights([kernel, bias])
            y_MC[n, :, :] = prior_model.predict(X)
        y_mean, y_std = mean_and_std_from_samples(y_MC, var_aleatoric=self.var_n)
        return return_outputs(y_mean, y_std, y_MC, return_mean, return_std, return_MC)


class MAPRegressor(Regressor):

    """ MAP regressor: the cost is - log likelihood + prior
     This regressor is coded in tensorflow"""

    def __init__(self, hidden_units, pre_model=None, input_dim=1, input_uq_dim=None, output_dim=1, var_n=1e-6,
                 activation='tanh', prior_type='gaussian', prior_params=(0., 1.)):

        super().__init__(hidden_units=hidden_units, pre_model=pre_model, input_dim=input_dim, input_uq_dim=input_uq_dim,
                         output_dim=output_dim, var_n=var_n, activation=activation,
                         prior_type=prior_type, prior_params=prior_params)

        # For this simple regressor, just use dense layers
        if self.pre_model is None:
            inputs = Input(shape=self.input_dim)
            x = inputs
        else:
            inputs = self.pre_model.input
            x = self.pre_model.output
        for l, units in enumerate(self.hidden_units + (self.output_dim, )):
            x = Dense(units, activation=self.activations[l], name='uq_layer_{}'.format(l),
                      kernel_regularizer=lambda omega: -1 * log_pdf(omega, pdf_type=self.prior_type,
                                                                    pdf_params=self.prior_params[l]),
                      bias_regularizer=lambda omega: -1 * log_pdf(omega, pdf_type=self.prior_type,
                                                                  pdf_params=self.prior_params[l]),
                      )(x)
        model = Model(inputs=inputs, outputs=x)
        # freeze all layers that do not account for uncertainties
        layers_freezed = [layer for layer in model.layers if ('uq_layer' not in layer.get_config()['name'].lower())]
        for layer in layers_freezed:
            layer.trainable = False
        self.estimator = model

    def fit(self, X, y, epochs=100, lr=0.001, verbose=False):
        # always use the full batch
        batch_size = X.shape[0]
        # compile model
        optimizer = Adam(lr=lr)
        self.estimator.compile(loss=self.loglike_cost, optimizer=optimizer)
        # fit to training data
        train_history = self.estimator.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        self.loss_history = np.array(train_history.history['loss'])

    def predict_UQ(self, X, return_mean=True, return_std=True):
        y_mean = self.estimator.predict(X).reshape((X.shape[0], self.output_dim))
        if isinstance(self.var_n, int) or isinstance(self.var_n, float):
            y_std = np.tile(np.array(np.sqrt(self.var_n)).reshape((1, 1)), [X.shape[0], self.output_dim])
        elif isinstance(self.var_n, np.ndarray):
            y_std = np.tile(np.sqrt(np.diag(self.var_n)).reshape((1, -1)), [X.shape[0], 1])
        else:
            raise TypeError('Input var_n should be a float, int or ndarray.')
        return return_outputs(y_mean, y_std, None, return_mean, return_std, False)


class MCdropoutRegressor(Regressor):

    """ Class that performs MC dropout """

    def __init__(self, hidden_units, pre_model=None, input_dim=1, input_uq_dim=None, output_dim=1, var_n=1e-6,
                 activation='tanh', prior_type='gaussian', prior_params=(0., 1.),
                 dropout_rate=None, concrete_dropout=False, dropout_rate_crossval=None, k_crossval=3,
                 ):
        """
        :param concrete_dropout
        :param dropout_rate: use same dropout rate for the whole network
        """
        super().__init__(hidden_units=hidden_units, pre_model=pre_model, input_dim=input_dim, input_uq_dim=input_uq_dim,
                         output_dim=output_dim, var_n=var_n, activation=activation,
                         prior_type=prior_type, prior_params=prior_params)
        # check inputs
        if self.prior_type.lower() != 'gaussian':
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
            inputs = Input(shape=(input_uq_dim,))
            x = inputs
        else:
            inputs = pre_model.input
            x = pre_model.output
        for l, units in enumerate(hidden_units):
            if concrete_dropout:
                layer_ = Dense(units, activation=activation)
                x = ConcreteDropout(layer_,
                                    weight_regularizer=1. / (2. * self.prior_params[l][1]**2),
                                    dropout_regularizer=1.,
                                    name='Concrete_Dropout_{}'.format(l)
                                    )(x)
            else:
                x = Dense(units, activation=activation,
                          kernel_regularizer=l2_reg(weight_regularizer=(1.-dropout_rate)/(2.*self.prior_params[l][1]**2))
                          )(x)
                x = Dropout(rate=dropout_rate)(x, training=True)
        predictions = Dense(units=output_dim, activation=None,
                            kernel_regularizer=l2_reg(weight_regularizer=1 / (2. * self.prior_params[-1]))
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
        pred_MC = np.zeros((ns, X.shape[0], self.output_dim))
        for j in range(ns):
            pred_MC[j, :, :] = self.estimator.predict(X).reshape((X.shape[0], self.output_dim))
        estimated_mean = np.mean(pred_MC, axis=0)
        aleatoric_noise = self.var_n
        if isinstance(aleatoric_noise, np.ndarray):
            aleatoric_noise = np.diag(aleatoric_noise).reshape((1, self.output_dim))
        estimated_var = aleatoric_noise + np.var(pred_MC, axis=0)
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

    def __init__(self, hidden_units, pre_model=None, input_dim=1, input_uq_dim=None, output_dim=1, var_n=1e-6,
                 activation='tanh', prior_type='gaussian', prior_params=1.,
                 n_estimators=1, prior_sampling=False):

        # check the inputs
        super().__init__(hidden_units=hidden_units, pre_model=pre_model, input_dim=input_dim, input_uq_dim=input_uq_dim,
                         output_dim=output_dim, var_n=var_n, activation=activation,
                         prior_type=prior_type, prior_params=prior_params)

        self.n_estimators = n_estimators
        if prior_sampling:
            kernels, biases = sample_weights_for_all_layers(self.kernel_shapes, self.bias_shapes, self.prior,
                                                            ns=self.n_estimators)

        self.estimators = []
        for n in range(n_estimators):
            if pre_model is None:
                inputs = Input(shape=(input_uq_dim,))
                x = inputs
            else:
                inputs = pre_model.input
                x = pre_model.output

            for l, units in enumerate(hidden_units + (output_dim,)):
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
                if l == len(hidden_units):
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


class VariationalInferenceRegressor(Regressor):
    """ UQ based on variational inference """

    def __init__(self, hidden_units, pre_model=None, input_dim=1, input_uq_dim=None, output_dim=1, var_n=1e-6,
                 activation='tanh', prior_type='gaussian', prior_params=(0., 1.),
                 n_weights_to_track=None):
        """ Initialize the network and define the cost function. """

        self.graph = tf.Graph()
        with self.graph.as_default():
            # Do the initial checks and computations for the network
            super().__init__(hidden_units=hidden_units, pre_model=pre_model, input_dim=input_dim, input_uq_dim=input_uq_dim,
                             output_dim=output_dim, var_n=var_n, activation=activation,
                             prior_type=prior_type, prior_params=prior_params)
            self.variational_pdf_type = 'gaussian'
            if n_weights_to_track is None:  # do not save any weights
                n_weights_to_track = [0] * (len(hidden_units) + 1)
            elif isinstance(n_weights_to_track, int):
                n_weights_to_track = [n_weights_to_track] * (len(hidden_units) + 1)
            if (not isinstance(n_weights_to_track, list)) or (len(n_weights_to_track) != len(hidden_units) + 1):
                raise TypeError('Input weights_to_track should be a list of length len(units_per_layer)+1 or an integer.')
            self.n_weights_to_track = n_weights_to_track

            # Initialize necessary variables and placeholders
            self.means_ = []
            self.stds_ = []
            self.variational_variables = []
            self.trainable_variables = []
            self.variational_params = []

            # add dense layers, add contributions of each layer to prior and variational posterior costs
            for l, units in enumerate(hidden_units + (output_dim,)):
                # before fitting to data, variational params are the prior
                rho_prior = np.log(np.exp(self.prior_params[l][1])-1)
                self.variational_params.extend([np.zeros(shape=self.kernel_shapes[l]),
                                                rho_prior * np.ones(shape=self.kernel_shapes[l]),
                                                np.zeros(shape=self.bias_shapes[l]),
                                                rho_prior * np.ones(shape=self.bias_shapes[l])
                                                ])

                # Define the parameters of the variational distribution to be trained: theta={mu, rho} for kernel and bias
                mu_kernel = tf.Variable(
                    tf.random_normal(shape=self.kernel_shapes[l], mean=0., stddev=0.1),
                    name='mu_kernel_layer_{}'.format(l), trainable=True, dtype=tf.float32)
                rho_kernel = tf.Variable(
                    -4 * tf.ones(shape=self.kernel_shapes[l], dtype=tf.float32),
                    name='rho_kernel_layer_{}'.format(l), trainable=True, dtype=tf.float32)
                sigma_kernel = tf.log(1. + tf.exp(rho_kernel))

                mu_bias = tf.Variable(
                    tf.random_normal(shape=self.bias_shapes[l], mean=0., stddev=0.1),
                    name='mu_bias_layer_{}'.format(l), trainable=True, dtype=tf.float32)
                rho_bias = tf.Variable(
                    -4 * tf.ones(shape=self.bias_shapes[l], dtype=tf.float32),
                    name='rho_bias_layer_{}'.format(l), trainable=True, dtype=tf.float32)
                sigma_bias = tf.log(1. + tf.exp(rho_bias))
                self.variational_variables.extend([mu_kernel, sigma_kernel, mu_bias, sigma_bias])
                self.trainable_variables.extend([mu_kernel, rho_kernel, mu_bias, rho_bias])

                # Keep track of some of the weights
                tmp = tf.reshape(mu_kernel, shape=(-1,))
                self.means_.extend(tmp[i] for i in range(n_weights_to_track[l]))
                tmp = tf.reshape(sigma_kernel, shape=(-1,))
                self.stds_.extend(tmp[i] for i in range(n_weights_to_track[l]))

    def compute_predictions(self, X, network_weights, ns):
        ndata = tf.shape(X)[0]  # number of independent data points
        x = X
        if self.pre_model is not None:
            for layer in self.pre_model.layers[1:]:  # layer 0 is an Input layer
                x = layer(x)
        x = tf.tile(tf.expand_dims(x, 0), [ns, 1, 1])
        for l, units in enumerate(self.hidden_units + (self.output_dim,)):
            # Compute activation(W X + b) for this layer, with w={W, b} sampled from q_{theta}(w)
            x = tf.add(tf.matmul(x, network_weights[2 * l]),
                       tf.tile(tf.expand_dims(network_weights[2 * l + 1], 1), [1, ndata, 1]))
            assert len(x.get_shape().as_list()) == 3
            if self.activations[l].lower() == 'tanh':
                x = tf.nn.tanh(x)
            elif self.activations[l].lower() == 'relu':
                x = tf.nn.relu(x)
            elif self.activations[l].lower() == 'linear':
                x = x
            else:
                raise ValueError('Activations should be relu, tanh or linear.')
        return x

    def sample_weights_from_variational(self, ns, layer_ind=None):
        if layer_ind is None:
            list_layers = list(range(len(self.hidden_units) + 1))
        else:
            list_layers = [layer_ind]
        weights = []
        for l, layer in enumerate(list_layers):
            # Samples ns_ biases and kernels from the variational distribution q_{theta}(w)
            kernel = tf.add(
                tf.tile(tf.expand_dims(self.variational_variables[4 * layer], 0), [ns, 1, 1]),
                tf.multiply(tf.tile(tf.expand_dims(self.variational_variables[4 * layer + 1], 0), [ns, 1, 1]),
                            tf.random_normal(shape=(ns,) + self.kernel_shapes[layer], mean=0., stddev=1.))
                )
            bias = tf.add(tf.tile(tf.expand_dims(self.variational_variables[4 * layer + 2], 0), [ns, 1]),
                          tf.multiply(tf.tile(tf.expand_dims(self.variational_variables[4 * layer + 3], 0), [ns, 1]),
                                      tf.random_normal(shape=(ns,) + self.bias_shapes[layer], mean=0., stddev=1.))
                          )
            weights.extend([kernel, bias])
        return weights

    def log_variational_pdf(self, weights, ns, axis_sum=(None, None), layer_ind=None):
        if layer_ind is None:
            list_layers = list(range(len(self.hidden_units) + 1))
        else:
            list_layers = [layer_ind]
        log_q = 0
        assert(len(weights) == 2 * len(list_layers))
        for l, layer in enumerate(list_layers):
            # kernels
            mu = tf.tile(tf.expand_dims(self.variational_variables[4 * layer], axis=0), [ns, 1, 1])
            sigma = tf.tile(tf.expand_dims(self.variational_variables[4 * layer + 1], axis=0), [ns, 1, 1])
            log_q += log_gaussian(x=weights[2 * l], mean=mu, std=sigma, axis_sum=axis_sum[0])
            # biases
            mu = tf.tile(tf.expand_dims(self.variational_variables[4 * layer + 2], axis=0), [ns, 1])
            sigma = tf.tile(tf.expand_dims(self.variational_variables[4 * layer + 3], axis=0), [ns, 1])
            log_q += log_gaussian(x=weights[2 * l + 1], mean=mu, std=sigma, axis_sum=axis_sum[0])
        return log_q

    def log_prior_pdf(self, weights, axis_sum=(None, None), layer_ind=None):
        if layer_ind is None:
            list_layers = list(range(len(self.hidden_units) + 1))
        else:
            list_layers = [layer_ind]
        log_q = 0
        assert (len(weights) == 2 * len(list_layers))
        for l, layer in enumerate(list_layers):
            # kernels
            log_q += log_pdf(x=weights[2 * l], pdf_type=self.prior_type,
                             pdf_params=self.prior_params[layer], axis_sum=axis_sum[0])
            # biases
            log_q += log_pdf(x=weights[2 * l + 1], pdf_type=self.prior_type,
                             pdf_params=self.prior_params[layer], axis_sum=axis_sum[0])
        return log_q


class BayesByBackprop(VariationalInferenceRegressor):
    """ BayesByBackprop algorithm, from 'Weight Uncertainty in Neural Networks', Blundell et al., 2015. """

    def __init__(self, hidden_units, pre_model=None, input_dim=1, input_uq_dim=None, output_dim=1, var_n=1e-6,
                 activation='tanh', prior_type='gaussian', prior_params=(0., 1.),
                 n_weights_to_track=None):
        """ Initialize the network and define the cost function. """

        # Do the initial checks and computations for the network
        super().__init__(hidden_units=hidden_units, pre_model=pre_model, input_dim=input_dim, input_uq_dim=input_uq_dim,
                         output_dim=output_dim, var_n=var_n, activation=activation,
                         prior_type=prior_type, prior_params=prior_params, n_weights_to_track=n_weights_to_track)

        with self.graph.as_default():

            # Initialize necessary variables and placeholders
            X_ = tf.placeholder(dtype=tf.float32, name='X_', shape=(None, ) + self.input_dim)  # input data
            y_ = tf.placeholder(dtype=tf.float32, name='y_', shape=(None, output_dim))  # output data
            ns_ = tf.placeholder(dtype=tf.int32, name='ns_', shape=())  # number of samples in MC approximation of cost

            # sample weights from variational distribution
            network_weights = self.sample_weights_from_variational(ns=ns_)

            # Compute the cost from the prior term -log(p(w))
            self.cost_prior = - self.log_prior_pdf(network_weights, axis_sum=(None, None))

            # Compute the cost from the variational distribution term log(q_{theta}(w))
            self.cost_var_post = self.log_variational_pdf(network_weights, ns_, axis_sum=(None, None))

            # Compute contribution of likelihood to cost: -log(p(data|w))
            self.predictions = self.compute_predictions(X_, network_weights, ns_)
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

        if isinstance(lr, int) or isinstance(lr, float):
            lr = [lr] * epochs
        # Set-up the training procedure
        with self.graph.as_default():
            #print(tf.trainable_variables())
            #print(self.trainable_variables)
            learning_rate = tf.placeholder(tf.float32, shape=())
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            total_grads_and_vars = opt.compute_gradients(self.cost, self.trainable_variables)
            grad_step = opt.apply_gradients(total_grads_and_vars)

        # Initilize tensorflow session (the same will be used later for ) and required variables
        X_ = self.graph.get_tensor_by_name('X_:0')
        y_ = self.graph.get_tensor_by_name('y_:0')
        ns_ = self.graph.get_tensor_by_name('ns_:0')
        loss_history, means_history, stds_history = [], [], []
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            # Run training loop
            for e in range(epochs):
                # apply the gradient descent step
                sess.run(grad_step, feed_dict={X_: X, y_: y, ns_: ns, learning_rate: lr[e]})
                # save the loss
                loss_history_ = sess.run([self.cost, self.cost_var_post, self.cost_prior, self.cost_likelihood],
                                          feed_dict={X_: X, y_: y, ns_: ns})
                loss_history.append(loss_history_)
                # save some of the weights
                if any(self.n_weights_to_track) != 0:
                    mean, std = sess.run([self.means_, self.stds_], feed_dict={X_: X, y_: y, ns_: ns})
                    means_history.append(mean)
                    stds_history.append(std)
                # print comments on terminal
                if verbose:
                    print('epoch = {}, loss = {}'.format(e, loss_history[-1][-1]))

            # Save the final variational parameters
            self.variational_params = sess.run(self.variational_variables, feed_dict={X_: X, y_: y, ns_: ns})

        # self.loss_history is a matrix of size (epochs, 4)
        self.loss_history = np.array(loss_history)
        # self.variational_params_history is a matrix of size (epochs, sum(self.n_weights_to_track), 2)
        if any(self.n_weights_to_track) != 0:
            self.variational_params_history = np.stack([np.array(means_history), np.array(stds_history)], axis=-1)
        else:
            self.variational_params_history = None

    def predict_UQ(self, X, ns=100, return_mean=True, return_std=True, return_MC=False):
        """ Predict output and uncertainty for new input data. """

        X_ = self.graph.get_tensor_by_name('X_:0')
        ns_ = self.graph.get_tensor_by_name('ns_:0')
        with tf.Session(graph=self.graph) as sess:
            feed_dict = dict(zip(self.variational_variables, self.variational_params))
            feed_dict.update({X_: X, ns_: ns})
            sess.run(tf.global_variables_initializer())
            y_MC = sess.run(self.predictions, feed_dict=feed_dict)

        y_mean, y_std = mean_and_std_from_samples(y_MC, var_aleatoric=self.var_n)
        return return_outputs(y_mean, y_std, y_MC, return_mean, return_std, return_MC)


class alphaBB(VariationalInferenceRegressor):
    """alpha-BlackBox algorithm, from 'Black-Box α-Divergence Minimization', Hernández-Lobato, 2016"""

    def __init__(self, alpha, hidden_units, pre_model=None, input_dim=1, input_uq_dim=None, output_dim=1, var_n=1e-6,
                 activation='tanh', prior_type='gaussian', prior_params=(0., 1.),
                 n_weights_to_track=None):
        """ Initialize the network and define the cost function. """

        # Do the initial checks and computations for the network
        super().__init__(hidden_units=hidden_units, pre_model=pre_model, input_dim=input_dim, input_uq_dim=input_uq_dim,
                         output_dim=output_dim, var_n=var_n, activation=activation,
                         prior_type=prior_type, prior_params=prior_params, n_weights_to_track=n_weights_to_track)

        # check alpha
        self.alpha = alpha
        if self.alpha <= 0. or self.alpha >= 1.:
            raise ValueError('Input alpha must be between 0 and 1')

        with self.graph.as_default():
            # Initialize necessary variables and placeholders
            X_ = tf.placeholder(dtype=tf.float32, name='X_', shape=(None,) + self.input_dim)  # input data
            y_ = tf.placeholder(dtype=tf.float32, name='y_', shape=(None, output_dim))  # output data
            ns_ = tf.placeholder(dtype=tf.int32, name='ns_', shape=())  # number of samples in MC approximation of cost

            ndata = tf.shape(X_)[0]  # number of independent data points
            self.normalization_term = 0.  # contribution of normalization terms to cost: -log(Z(prior))-log(Z(q))
            self.log_factor = 0.  # log of the site approximations f: log(f(w))=s(w).T lambda_{f}

            # sample weights from variational distribution
            network_weights = self.sample_weights_from_variational(ns=ns_)
            # add dense layers and compute normalization term of the cost for all layers
            for l, units in enumerate(hidden_units + (output_dim,)):

                mu_kernel = self.variational_variables[4 * l]
                sigma_kernel = self.variational_variables[4 * l + 1]
                mu_bias = self.variational_variables[4 * l + 2]
                sigma_bias = self.variational_variables[4 * l + 3]
                kernel = network_weights[2 * l]
                bias = network_weights[2 * l + 1]

                # compute normalization term: -log(Z(prior))-log(Z(q))
                self.normalization_term += tf.reduce_sum(tf.log(self.prior_params[l][1])
                                                         - tf.log(sigma_kernel)
                                                         - 0.5 * tf.square(tf.divide(mu_kernel, sigma_kernel)))
                self.normalization_term += tf.reduce_sum(tf.log(self.prior_params[l][1])
                                                         - tf.log(sigma_bias)
                                                         - 0.5 * tf.square(tf.divide(mu_bias, sigma_bias)))

                # compute the factor parameters lambda_f, then the log of the site approximations
                var_f_kernel, mu_over_var_f_kernel = factor_params_gaussian(var_prior=self.prior_params[l][1]**2,
                                                                            mu_post=mu_kernel,
                                                                            var_post=tf.square(sigma_kernel),
                                                                            ndata=tf.to_float(ndata))
                var_f_bias, mu_over_var_f_bias = factor_params_gaussian(var_prior=self.prior_params[l][1]**2,
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
            self.predictions = self.compute_predictions(X_, network_weights, ns_)
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

        if isinstance(lr, int) or isinstance(lr, float):
            lr = [lr] * epochs
        # Set-up the training procedure
        with self.graph.as_default():
            #print(tf.trainable_variables())
            #print(self.trainable_variables)
            learning_rate = tf.placeholder(tf.float32, shape=())
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            total_grads_and_vars = opt.compute_gradients(self.cost, self.trainable_variables)
            grad_step = opt.apply_gradients(total_grads_and_vars)

        # Initilize tensorflow session (the same will be used later for ) and required variables
        X_ = self.graph.get_tensor_by_name('X_:0')
        y_ = self.graph.get_tensor_by_name('y_:0')
        ns_ = self.graph.get_tensor_by_name('ns_:0')
        loss_history, means_history, stds_history = [], [], []
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            # Run training loop
            for e in range(epochs):
                # apply the gradient descent step
                sess.run(grad_step, feed_dict={X_: X, y_: y, ns_: ns, learning_rate: lr[e]})
                # save the loss
                loss_history_ = sess.run(self.cost, feed_dict={X_: X, y_: y, ns_: ns})
                loss_history.append(loss_history_)
                # save some of the weights
                if any(self.n_weights_to_track) != 0:
                    mean, std = sess.run([self.means_, self.stds_], feed_dict={X_: X, y_: y, ns_: ns})
                    means_history.append(mean)
                    stds_history.append(std)
                # print comments on terminal
                if verbose:
                    print('epoch = {}, loss = {}'.format(e, loss_history[-1]))

            # Save the final variational parameters
            self.variational_params = sess.run(self.variational_variables, feed_dict={X_: X, y_: y, ns_: ns})

        # self.loss_history is a matrix of size (epochs, 4)
        self.loss_history = np.array(loss_history)
        # self.variational_params_history is a matrix of size (epochs, sum(self.n_weights_to_track), 2)
        if any(self.n_weights_to_track) != 0:
            self.variational_params_history = np.stack([np.array(means_history), np.array(stds_history)], axis=-1)
        else:
            self.variational_params_history = None

    def predict_UQ(self, X, ns=100, return_mean=True, return_std=True, return_MC=False):
        """ Predict output and uncertainty for new input data. """

        X_ = self.graph.get_tensor_by_name('X_:0')
        ns_ = self.graph.get_tensor_by_name('ns_:0')
        with tf.Session(graph=self.graph) as sess:
            feed_dict = dict(zip(self.variational_variables, self.variational_params))
            feed_dict.update({X_: X, ns_: ns})
            sess.run(tf.global_variables_initializer())
            y_MC = sess.run(self.predictions, feed_dict=feed_dict)

        y_mean, y_std = mean_and_std_from_samples(y_MC, var_aleatoric=self.var_n)
        return return_outputs(y_mean, y_std, y_MC, return_mean, return_std, return_MC)


class MCMCRegressor(Regressor):

    def __init__(self, ns_per_chain, hidden_units, pre_model=None, input_dim=1, input_uq_dim=None, output_dim=1, var_n=1e-6,
                 activation='tanh', prior_type='gaussian', prior_params=(0., 1.),
                 algorithm='am', n_chains=1, burnin=0, jump=1):

        super().__init__(hidden_units=hidden_units, pre_model=pre_model, input_dim=input_dim, input_uq_dim=input_uq_dim,
                         output_dim=output_dim, var_n=var_n, activation=activation,
                         prior_type=prior_type, prior_params=prior_params)

        self.algorithm = algorithm
        self.n_chains = n_chains
        if algorithm.lower() == 'de-mc' and n_chains < 2:
            raise ValueError('With DE-MC, use at least 2 chains.')
        self.ns_per_chain = ns_per_chain
        self.burnin = burnin
        self.jump = jump

        self.estimator = self.build_network_keras()

        # initialize by sampling the necessary seeds
        weights = self.sample_weights_from_prior(n_chains, layer_ind=None)
        seed = []
        for n in range(n_chains):
            weight_vector = np.concatenate([w[n].reshape((-1, 1)) for w in weights], axis=0)
            seed.append(weight_vector)
        self.seeds = np.concatenate(seed, axis=1)

    def fit(self, X, y, verbose=True):

        # do MCMC
        T = self.burnin + self.ns_per_chain * self.jump
        if self.algorithm == 'am':
            samples, _, acc = am(log_pdf=partial(self.evaluate_weight_vector, X_=X, y_=y),
                                 seed=self.seeds, T=T, verbose=verbose)
        if self.algorithm == 'de-mc':
            samples, _, acc = de_mc(log_pdf=partial(self.evaluate_weight_vector, X_=X, y_=y),
                                 seed=self.seeds, T=T, verbose=verbose)
        else:
            raise ValueError
        self.samples = post_process_samples(samples, burnin=self.burnin, jump=self.jump, concatenate_chains=True)

    def predict_UQ(self, X, return_mean=True, return_std=True, return_MC=False):
        # predict for unseen X, mean and standard deviations
        y_MC = np.zeros((self.samples.shape[0], X.shape[0], self.output_dim))
        for n, weight_vector in enumerate(self.samples):
            weights = weights_from_vector_to_layers(weight_vector, kernel_shapes=self.kernel_shapes,
                                                    bias_shapes=self.bias_shapes)
            self.estimator.set_weights(weights)
            y_MC[n, :, :] = self.estimator.predict(X).reshape((X.shape[0], self.output_dim))
        y_mean, y_std = mean_and_std_from_samples(y_MC, var_aleatoric=self.var_n)
        return return_outputs(y_mean, y_std, y_MC, return_mean, return_std, return_MC)

    def evaluate_weight_vector(self, weight_vector, X_, y_):
        # function that computes the log(p(D|w))+log(prior(w)) of some data given a weight vector w
        weights = weights_from_vector_to_layers(weight_vector, kernel_shapes=self.kernel_shapes,
                                                bias_shapes=self.bias_shapes)
        # compute value of prior log(prior(w))
        log_prior_value = 0
        for l in range(len(self.hidden_units) + 1):
            log_prior_value += log_pdf(weights[2 * l], pdf_type=self.prior_type, pdf_params=self.prior_params[l])
            log_prior_value += log_pdf(weights[2 * l + 1], pdf_type=self.prior_type, pdf_params=self.prior_params[l])
        # compute value of log(p(D|w))
        self.estimator.set_weights(weights)
        y_pred = self.estimator.predict(X_)
        log_likelihood_value = K.eval((-1) * self.loglike_cost(y_true=y_, y_pred=y_pred))
        return log_likelihood_value + log_prior_value