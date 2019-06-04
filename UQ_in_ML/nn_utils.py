# Audrey Olivier
# 4/16/2019
# Utility functions for neural networks

import numpy as np
from keras import backend as K
from keras.layers import Layer, Wrapper, Input, Lambda
from keras.initializers import RandomUniform, Constant
from keras.engine import InputSpec
from keras.callbacks import Callback
from keras import activations, initializers
from keras.regularizers import Regularizer
from keras.models import Model

import tensorflow as tf


# Losses

# these compute p(xn|theta) for all xn independently, returns a matrix. Need to sum over to get the cost
def homoscedastic_negloglike_scalar_variance(y_true, y_pred, var_n):
    # var_n is a fixed known scalar
    indpt_outputs = tf.reduce_sum(tf.square(y_true-y_pred) / (2 * var_n) + 0.5 * tf.log(2 * np.pi * var_n), axis=-1)
    return indpt_outputs


def homoscedastic_negloglike_full_covariance(y_true, y_pred, inv_cov_n):
    # var_n is a fixed known full covariance, inv_cov_n is the inverse of the covariance matrix
    Y = y_true-y_pred
    indpt_outputs = tf.reduce_sum(tf.multiply(tf.tensordot(Y, inv_cov_n, axes=[[-1], [0]]), Y), axis=-1)
    return indpt_outputs


# Regularizers

class l2_reg(Regularizer):
    """Regularizer for L2 regularization.
    # Arguments
        weight_regularizer: Float; L2 regularization factor.
    """
    def __init__(self, weight_regularizer=0.):
        self.weight_regularizer = K.cast_to_floatx(weight_regularizer)
    def __call__(self, x):
        return self.weight_regularizer * K.sum(K.square(x))
    def get_config(self):
        return {'weight_regularizer': float(self.weight_regularizer)}


class l2_reg_with_initial(Regularizer):
    """Regularizer for L2 regularization.
    # Arguments
        weight_regularizer: Float; L2 regularization factor.
    """
    def __init__(self, weight_regularizer, initial_weights):
        self.weight_regularizer = K.cast_to_floatx(weight_regularizer)
        self.initial_weights = K.cast_to_floatx(initial_weights)
    def __call__(self, x):
        return self.weight_regularizer * K.sum(K.square(x-self.initial_weights))
    def get_config(self):
        return {'weight_regularizer': float(self.weight_regularizer),
                'initial_weights': self.initial_weights}


class prior_reg(Regularizer):
    """Regularizer for L2 regularization.
    # Arguments
        weight_regularizer: Float; L2 regularization factor.
    """
    def __init__(self, prior):
        self.prior = prior
    def __call__(self, x):
        if self.prior['type'].lower() == 'gaussian':
            return K.sum(K.square(x)) / (2 * self.prior['variance'])
        if self.prior['type'].lower() == 'gaussian_mixture':
            term1 = self.prior['proba_1'] * (1 / K.sqrt(self.prior['variance_1']) *
                                        K.exp(-1 * K.square(x) / (2 * self.prior['variance_1'])))
            term2 = (1 - self.prior['proba_1']) * (1 / K.sqrt(self.prior['variance_2']) *
                                              K.exp(-1 * K.square(x) / (2 * self.prior['variance_2'])))
            return (-1) * K.sum(K.log(term1 + term2))
    def get_config(self):
        return {'prior': self.prior}


class prior_reg_with_initial(Regularizer):
    """Regularizer for L2 regularization.
    # Arguments
        weight_regularizer: Float; L2 regularization factor.
    """
    def __init__(self, prior, initial_weights):
        self.prior = prior
        self.initial_weights = initial_weights
    def __call__(self, x):
        if self.prior['type'].lower() == 'gaussian':
            return K.sum(K.square(x)) / (2 * self.prior['variance'])
        if self.prior['type'].lower() == 'gaussian_mixture':
            term1 = self.prior['proba_1'] * (1 / K.sqrt(self.prior['variance_1']) *
                                             K.exp(K.square(x-self.initial_weights) / (2 * self.prior['variance_1'])))
            term2 = (1 - self.prior['proba_1']) * (1 / K.sqrt(self.prior['variance_2']) *
                                                   K.exp(K.square(x-self.initial_weights) / (2 * self.prior['variance_2'])))
            return K.sum(K.log(term1 + term2))
    def get_config(self):
        return {'prior': self.prior}


# Initializers


def my_init(shape, dtype=None, prior_dict=None):
    if prior_dict['type'] == 'gaussian':
        return K.sqrt(prior_dict['variance']) * K.random_normal(shape, dtype=dtype)
    elif prior_dict['type'] == 'gaussian_mixture':
        binomial = K.random_binomial(shape=shape, p=prior_dict['proba_1'])
        variances = prior_dict['variance_2'] + (prior_dict['variance_1'] - prior_dict['variance_2']) * binomial
        return K.sqrt(variances) * K.random_normal(shape, dtype=dtype)



# Classes for Dropout


class MyDropout(Wrapper):
    """
    Dropout function to resemble the Concrete Dropout one, see following
    """
    def __init__(self, layer, dropout_rate=0.2, weight_regularizer=1e-6, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(MyDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = K.cast_to_floatx(weight_regularizer)
        self.supports_masking = True
        self.p = K.cast_to_floatx(dropout_rate)

    def build(self, input_shape=None):
        assert len(input_shape) == 2
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(MyDropout, self).build()  # this is very weird... we must call super before we add new losses
        # initialise regulariser / prior KL term
        #self.p = self.layer.add_weight(name='p',
        #                               shape=(1,),
        #                               initializer=Constant(self.p),
        #                               trainable=False)  # ~0.1 to ~0.5 in logit space.
        input_dim = input_shape[-1] # we drop only last dim
        weight = self.layer.kernel
        # Note: we divide by (1 - p) because we scaled layer output by (1 - p)
        kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) * (1. - self.p)
        regularizer = K.sum(kernel_regularizer)
        self.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def my_dropout(self, inputs):
        return K.dropout(inputs, self.p, noise_shape=None)

    def call(self, inputs, training=None):
        return self.layer.call(self.my_dropout(inputs))


class ConcreteDropout(Wrapper):
    """
    This code is based on code from Yarin Gal github:
    https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout-keras.ipynb

    This wrapper allows to learn the dropout probability for any given input Dense layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers which have 2D
    kernels, not just `Dense`. However, Conv2D layers require different
    weighing of the regulariser (use SpatialConcreteDropout instead).
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()  # this is very weird.. we must call super before we add new losses

        # initialise p
        self.p_logit = self.layer.add_weight(name='p_logit',
                                            shape=(1,),
                                            initializer=RandomUniform(self.init_min, self.init_max),
                                            trainable=True)
        self.p = K.sigmoid(self.p_logit[0])

        # initialise regulariser / prior KL term
        assert len(input_shape) == 2, 'this wrapper only supports Dense layers'
        input_dim = np.prod(input_shape[-1])  # we drop only last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * (1. - self.p) * K.sum(K.square(weight))
        dropout_regularizer = self.p * K.log(self.p)
        dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1

        unif_noise = K.random_uniform(shape=K.shape(x))
        drop_prob = (
            K.log(self.p + eps)
            - K.log(1. - self.p + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs))
            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)


class DropoutRateHistory(Callback):
    """
    Callback to store the values of the learnt dropout rate during training (or at the beginning / end)
    """
    def on_train_begin(self, logs=None):
        self.concrete_dropout_layers_indices = [l for (l, layer) in enumerate(self.model.layers)
                                                if ('concrete_dropout' in layer.get_config()['name'].lower())]
        self.dropout_rates = []
        #self.dropout_rates.append([self.model.layers[l].get_weights()[2][0]
        #                           for l in self.concrete_dropout_layers_indices])
        self.dropout_rates.append([K.eval(self.model.layers[l].p)
                                   for l in self.concrete_dropout_layers_indices])

    def on_batch_end(self, batch=None, logs=None):
        self.dropout_rates.append([K.eval(self.model.layers[l].p)
                                   for l in self.concrete_dropout_layers_indices])


class DropoutRateSetup(Callback):
    """
    Callback to store the values of the learnt dropout rate during training (or at the beginning / end)
    """
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        super(DropoutRateSetup, self).__init__()

    def on_train_begin(self, logs=None):
        for layer in self.model.layers:
            if 'dropout' in layer.get_config()['name'].lower():
                layer.rate = self.dropout_rate


# Bayes By Backprop


class BayesDense(Layer):
    """
    Adding Bayesian to densely-connected NN layer.
    """

    def __init__(self, units, prior,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='glorot_uniform',
                 mu_initializer='glorot_normal',
                 rho_initializer=RandomUniform(-5, -3),
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(BayesDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.mu_initializer = initializers.get(mu_initializer)
        self.rho_initializer = initializers.get(rho_initializer)
        if prior['type'].lower() not in ['gaussian', 'gaussian_mixture']:
            raise ValueError('Prior type should be gaussian or gaussian_mixture.')
        self.prior = prior
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.mu_kernel = self.add_weight(shape=(input_dim, self.units),
                                         initializer=self.mu_initializer,
                                         name='mu_kernel',
                                         trainable=True)
        self.rho_kernel = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.rho_initializer,
                                          name='rho_kernel',
                                          trainable=True)

        self.mu_bias = self.add_weight(shape=(self.units,),
                                       initializer=self.mu_initializer,
                                       name='mu_bias',
                                       trainable=True)
        self.rho_bias = self.add_weight(shape=(self.units,),
                                        initializer=self.rho_initializer,
                                        name='rho_bias',
                                        trainable=True)

        sigma_kernel = K.log(1. + K.exp(self.rho_kernel))
        self.kernel = tf.Variable(initial_value=K.random_normal(shape=self.mu_kernel.shape, mean=0.0, stddev=1.0),
                                  name='kernel', trainable=False)
        self.kernel = self.mu_kernel + sigma_kernel * self.kernel
        sigma_bias = K.log(1. + K.exp(self.rho_bias))
        self.bias = tf.Variable(initial_value=K.random_normal(shape=self.mu_bias.shape, mean=0.0, stddev=1.0),
                                name='bias', trainable=False)
        self.bias = self.mu_bias + sigma_bias * self.bias

        # add the losses that correspond to the q_term and prior_term
        # kernel losses
        #prior_term_kernel = K.sum(K.square(self.kernel)) / (2 * self.prior['variance'])
        #q_term_kernel = - K.sum(K.log(sigma_kernel)) - \
        #                1 / 2 * K.sum(K.square((self.kernel - self.mu_kernel) / sigma_kernel))
        # bias losses
        #prior_term_bias = K.sum(K.square(self.bias)) / (2 * self.prior['variance'])
        #q_term_bias = - K.sum(K.log(sigma_bias)) - \
        #              1 / 2 * K.sum(K.square((self.bias - self.mu_bias) / sigma_bias))
        #self.add_loss(prior_term_kernel + prior_term_bias)
        #self.add_loss(q_term_kernel + q_term_bias)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, training=None):

        #sigma_kernel = K.log(1. + K.exp(self.rho_kernel))
        #self.kernel = self.mu_kernel + sigma_kernel * K.random_normal(shape=self.mu_kernel.shape, mean=0.0, stddev=1.0)
        #sigma_bias = K.log(1. + K.exp(self.rho_bias))
        #self.bias = self.mu_bias + sigma_bias * K.random_normal(shape=self.mu_bias.shape, mean=0.0, stddev=1.0)

        output = K.dot(inputs, self.kernel)
        output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


# Create pre-networks


def build_scaling_layers(X_train):
    n_data, input_dim = X_train.shape
    mean_X_train, std_X_train = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    # create model using Functional API
    inputs = Input(shape=(input_dim,))
    outputs = Lambda(lambda x_: (x_ - mean_X_train) / std_X_train)(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def load_VGG16(input_shape, pruning=None):

    # create the base pre-trained model
    base_model = VGG16(weights='imagenet', include_top=False, pooling=None, input_shape=input_shape)
    inputs = base_model.inputs
    # replace max pooling with average pooling layers
    layer_id = [3, 6, 10, 14]
    layers = [l for l in base_model.layers]
    x = layers[0].output
    count_pooling_layers = 0
    #if pruning is None:
    #    last_layer = len(layers)-1
    #elif pruning in [1, 2, 3]:
    #    last_layer = len(layers)-1-4*pruning
    #else:
    #    raise ValueError('Audrey :( pruning=None, 1, 2, or 3 (3 you prune a lot !!!)')
    for i in range(1, len(layers)-1):
        if i in layer_id:
            count_pooling_layers += 1
            x = AveragePooling2D(pool_size=(2, 2), name='block{}_avpool'.format(count_pooling_layers))(x)
        else:
            x = layers[i](x)
    # add a global spatial average pooling layer
    x = AveragePooling2D(pool_size=(int(x.shape[1]), int(x.shape[2])),
                         name='block{}_avpool'.format(count_pooling_layers+1))(x)
    # reshape it so it is ready for classification/regression
    x = Reshape(target_shape=(int(x.shape[3]),))(x)
    base_model = Model(inputs=inputs, outputs=x)
    #base_model.summary()

    # Last layer not to train
    if pruning is None:
        last_layer_untrainable = 18
    elif pruning in [1, 2, 3]:
        last_layer_untrainable = 18-4*pruning
    return base_model, last_layer_untrainable





