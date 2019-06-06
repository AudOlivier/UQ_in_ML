# Audrey Olivier
# 4/16/2019
# Plotting utility functions

import matplotlib.pyplot as plt
import numpy as np
from pyDOE import lhs
from scipy.stats.distributions import norm
import tensorflow as tf

# Random utils functions


def preprocess_prior(prior, n_layers=None):
    if not isinstance(prior, dict):
        raise ValueError('The prior should be given as a dictionary.')
    if 'type' not in prior.keys():
        raise ValueError('The prior dictionary should contain a "type" key.')
    if prior['type'].lower() not in ['gaussian', 'gaussian_mixture']:
        raise ValueError('The prior type should be "gaussian" or "gaussian_mixture".')
    if prior['type'].lower() == 'gaussian':
        if 'variance' not in prior.keys():
            raise ValueError('For a gaussian prior, the variance should be provided.')
        if not isinstance(prior['variance'], list):
            prior['variance'] = [prior['variance']]*n_layers
        prior['variance'] = [float(v) for v in prior['variance']]
    if prior['type'].lower() == 'gaussian_mixture':
        if ('variance_1' not in prior.keys()) or ('variance_2' not in prior.keys()) or ('proba_1' not in prior.keys()):
            raise ValueError('For a gaussian_mixture prior, variance_1, variance_2, proba_1 should be provided.')
        for key in ['variance_1', 'variance_2', 'proba_1']:
            if not isinstance(prior[key], list):
                prior[key] = [prior[key]]*n_layers
        for key in ['variance_1', 'variance_2', 'proba_1']:
            prior[key] = [float(v) for v in prior['key']]
    return prior


def compute_weights_shapes(input_dim, units_per_layer, output_dim):
    # compute shape of kernels and biases
    all_units = units_per_layer + (output_dim, )
    kernel_shapes = [(input_dim, all_units[0])]
    for j, units in enumerate(all_units[:-1]):
        kernel_shapes.append((units, all_units[j+1]))
    bias_shapes = [(units, ) for units in all_units]
    return kernel_shapes, bias_shapes


def compute_glorot_normal_variance(kernel_shapes):
    # Glorot Normal variance = 2/(fan_in+fan_out)
    variances = [2 / (shape_kernel[0] + shape_kernel[1]) for shape_kernel in kernel_shapes]
    return variances


def sample_weights_for_one_layer(kernel_shape, bias_shape, prior_layer, ns=1):
    assert len(kernel_shape) == 2
    assert len(bias_shape) == 1
    if ns == 1:
        rv_kernel = np.random.normal(loc=0., scale=1, size=(1,) + kernel_shape)
        rv_bias = np.random.normal(loc=0., scale=1, size=(1,) + bias_shape)
    else:
        lhd = lhs(int(np.prod(kernel_shape)), samples=ns)
        rv_kernel = norm(loc=0., scale=1.).ppf(lhd).reshape(((ns,) + kernel_shape))
        lhd = lhs(int(np.prod(bias_shape)), samples=ns)
        rv_bias = norm(loc=0., scale=1.).ppf(lhd).reshape(((ns,) + bias_shape))
    if prior_layer['type'].lower() == 'gaussian':
        new_kernel = np.sqrt(prior_layer['variance']) * rv_kernel
        new_bias = np.sqrt(prior_layer['variance']) * rv_bias
    elif prior_layer['type'].lower() == 'gaussian_mixture':
        vars_kernel = np.random.choice(a=[prior_layer['variance_1'], prior_layer['variance_2']],
                                       p=[prior_layer['proba_1'], 1. - prior_layer['proba_1']],
                                       size=(ns,) + kernel_shape)
        new_kernel = np.sqrt(vars_kernel) * rv_kernel
        vars_bias = np.random.choice(a=[prior_layer['variance_1'], prior_layer['variance_2']],
                                     p=[prior_layer['proba_1'], 1. - prior_layer['proba_1']],
                                     size=(ns,) + bias_shape)
        new_bias = np.sqrt(vars_bias) * rv_bias
    else:
        raise ValueError('Prior must be gaussian or gaussian_mixture')
    if ns == 1:
        new_kernel, new_bias = new_kernel[0], new_bias[0]
    return new_kernel, new_bias


def sample_weights_for_all_layers(kernel_shapes, bias_shapes, prior, ns=1, as_vector=False):
    new_kernels, new_biases = [], []
    new_vector = []
    for l in range(len(kernel_shapes)):
        prior_layer = dict([(key, val[l]) for key, val in prior.items()])
        prior_layer['type'] = prior['type']
        new_kernel, new_bias = sample_weights_for_one_layer(kernel_shapes[l], bias_shapes[l], prior_layer, ns=ns)
        new_kernels.append(new_kernel)
        new_biases.append(new_bias)
        if as_vector:
            new_vector.append(np.concatenate([new_kernel.reshape((ns, -1)), new_bias.reshape((ns, -1))], axis=1))
    if as_vector:
        return np.concatenate(new_vector, axis=1)
    return new_kernels, new_biases


def weights_from_layers_to_vector(nn, ind_layers, prior=None):
    weight_vector = []
    log_prior_proba = 0
    for i, l in enumerate(ind_layers):
        weights = nn.layers[l].get_weights()
        kernel, bias = weights[0].reshape((-1,)), weights[1].reshape((-1,))
        weight_vector.append(list(kernel))
        weight_vector.append(list(bias))
        if prior is not None and prior['type'].lower() == 'gaussian':
            log_prior_proba -= np.sum(kernel ** 2, axis=None) / (2 * prior['variance'][i])
            log_prior_proba -= np.sum(bias ** 2, axis=None) / (2 * prior['variance'][i])
    return np.array(weight_vector), log_prior_proba


def weights_from_vector_to_layers(weight_vector, nn, ind_layers, kernel_shapes, bias_shapes, prior=None):
    c_ = 0
    log_prior_proba = 0
    for i, l in enumerate(ind_layers):
        kernel = weight_vector[c_:c_+np.prod(kernel_shapes[i])].reshape(kernel_shapes[i])
        c_ += np.prod(kernel_shapes[i])
        bias = weight_vector[c_:c_+np.prod(bias_shapes[i])].reshape(bias_shapes[i])
        c_ += np.prod(bias_shapes[i])
        nn.layers[l].set_weights([kernel, bias])
        if prior is not None and prior['type'].lower() == 'gaussian':
            log_prior_proba -= np.sum(kernel ** 2, axis=None) / (2 * prior['variance'][i])
            log_prior_proba -= np.sum(bias ** 2, axis=None) / (2 * prior['variance'][i])
    return log_prior_proba


# Probability distributions

def log_gaussian(x, mu, std, axis_sum=None):
    log2pi = np.log(2 * np.pi).astype(np.float32)
    return tf.reduce_sum(- 0.5 * log2pi - tf.log(std) - 0.5 * tf.square(tf.divide(tf.subtract(x, mu), std)),
                         axis=axis_sum)


def log_gaussian_mixture(x, mu1, mu2, std1, std2, pi1, axis_sum=None):
    sqrt2pi = np.sqrt(2 * np.pi).astype(np.float32)
    term1 = tf.multiply(pi1,
                        tf.multiply(1./(sqrt2pi * std1),
                                    tf.exp(-0.5 * tf.square(tf.divide(tf.subtract(x, mu1), std1)))
                                    )
                        )
    term2 = tf.multiply(1. - pi1,
                        tf.multiply(1. / (sqrt2pi * std2),
                                    tf.exp(-0.5 * tf.square(tf.divide(tf.subtract(x, mu2), std2)))
                                    )
                        )
    return tf.reduce_sum(tf.log(tf.add(term1, term2)), axis=axis_sum)


# Functions for alpha black box

def factor_params_gaussian(var_prior, mu_post, var_post, ndata):
    var_factor = 1. / (1. / ndata * (1. / var_post - 1. / var_prior))
    mu_over_var_factor = 1. / ndata * tf.divide(mu_post, var_post)
    return var_factor, mu_over_var_factor


# Plot functions

def plot_mean_2sigma(x, y_mean, y_std, ax, var_n=None,
                     color_mean='black', color_std='gray', alpha_std=.3):

    assert len(x.shape) == 1
    assert len(y_mean.shape) == 1
    assert len(y_std.shape) == 1

    ax.plot(x, y_mean, color=color_mean, label='mean prediction')
    if var_n is not None:
        # plot mean +/- 2 std. dev. of the noise term only, in darker shade
        ax.fill(np.concatenate([x, x[::-1]]),
                np.concatenate([y_mean - 2. * np.sqrt(var_n), (y_mean + 2. * np.sqrt(var_n))[::-1]]),
                alpha=alpha_std * 2, fc=color_std, ec='None', label='aleatoric uncertainty')
    # plot mean +/- 2 std. dev.
    ax.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_mean - 2. * y_std, (y_mean + 2. * y_std)[::-1]]),
            alpha=alpha_std, fc=color_std, ec='None', label='aleatoric + epistemic')


def plot_mean_MC(x, y_mean, y_MC, ax,
                 color_mean='black', color_MC='gray', alpha_std=.2):

    assert len(x.shape) == 1
    assert len(y_mean.shape) == 1
    assert len(y_MC.shape) == 2

    ax.plot(x, y_mean, color=color_mean, label='mean prediction')
    for j in range(y_MC.shape[-1]):
        if j == 0:
            ax.plot(x, y_MC[:, j], color=color_MC, alpha=alpha_std, label='aleatoric + epistemic')
        else:
            ax.plot(x, y_MC[:, j], color=color_MC, alpha=alpha_std)



def plot_UQ(regressor, X, domain=None, ax=None, figsize=(8, 6),
           plot_2sig=True, plot_MC=False, plot_one_posterior=False, plot_one_prior=False):
    """
    Plot the mean prediction and uncertainty at inputs X

    :param regressor: instance of a UQ regressor
    :param X: input where to evaluate/plot the regressor
    :param domain: 1d domain corresponding to X that allows plotting
    :param ax:
    :param figsize:
    :param plot_2sig: plot mean +/- 2 std. dev. intervals?
    :param plot_MC: plot MC draws from posterior?
    :param plot_one_posterior: plot one draw from posterior?
    :param plot_one_prior: plot one draw from prior?
    :return:
    """
    if plot_2sig and plot_MC:
        raise ValueError('plot_2sig and plot_MC cannot both be True: the user must choose between plotting'
                         ' mean +/- 2 std. dev. or MC draws from the posterior.')
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if (domain is None) and (len(X.shape) == 2 and X.shape[1] == 1):
        domain = X
    if (domain is None) and (len(X.shape) > 2 or X.shape[1] != 1):
        raise ValueError('For input X that is not 1-dimensional, provide a 1d domain for the plots')
    ymean, ystd, yMC = regressor.predict_UQ(domain, output_MC=(plot_MC or plot_one_posterior))
    ax.plot(domain, ymean, color='black', label='mean prediction')
    # plot mean +/- 2 std. dev. of the noise term only, in darker shade
    ax.fill(np.concatenate([domain, domain[::-1]]),
            np.concatenate([ymean - 2. * np.sqrt(regressor.var_n), (ymean + 2. * np.sqrt(regressor.var_n))[::-1]]),
            alpha=.7, fc='gray', ec='None', label='aleatoric uncertainty')
    if not plot_MC:  # predict and plot the variance
        ystd = np.reshape(ystd, ymean.shape)
        # plot mean +/- 2 std. dev.
        ax.fill(np.concatenate([domain, domain[::-1]]),
                np.concatenate([ymean - 2. * ystd, (ymean + 2. * ystd)[::-1]]),
                alpha=.3, fc='gray', ec='None', label='aleatoric + epistemic')
    else:
        for j in range(yMC.shape[-1]):
            if j == 0:
                ax.plot(domain, yMC[:, j], color='grey', alpha=0.1, label='aleatoric + epistemic')
            else:
                ax.plot(domain, yMC[:, j], color='grey', alpha=0.1)
    if plot_one_posterior:
        # plot one of the post in red
        ax.plot(domain, yMC[:, 0], color='red', label='one realization of posterior', alpha=0.5)
    #if plot_one_prior:
    #    # plot one of the post in red
    #    ax.plot(domain, yprior[:, 0], color='red', linestyle='-.', label='one realization of prior', alpha=0.3)
    return ax

