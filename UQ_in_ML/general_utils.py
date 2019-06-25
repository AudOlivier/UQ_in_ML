# Audrey Olivier
# 4/16/2019
# Plotting utility functions

import matplotlib.pyplot as plt
import numpy as np
from pyDOE import lhs
from scipy.stats.distributions import norm
import tensorflow as tf


# Probability distributions

def log_gaussian(x, mean, std, axis_sum=None):
    log2pi = np.log(2 * np.pi).astype(np.float32)
    return tf.reduce_sum(- 0.5 * log2pi - tf.log(std) - 0.5 * tf.square(tf.divide(tf.subtract(x, mean), std)),
                         axis=axis_sum)


def log_gaussian_mixture(x, mean1, mean2, std1, std2, prob1, axis_sum=None):
    sqrt2pi = np.sqrt(2 * np.pi).astype(np.float32)
    term1 = tf.multiply(prob1,
                        tf.multiply(1./(sqrt2pi * std1),
                                    tf.exp(-0.5 * tf.square(tf.divide(tf.subtract(x, mean1), std1)))
                                    )
                        )
    term2 = tf.multiply(1. - prob1,
                        tf.multiply(1. / (sqrt2pi * std2),
                                    tf.exp(-0.5 * tf.square(tf.divide(tf.subtract(x, mean2), std2)))
                                    )
                        )
    return tf.reduce_sum(tf.log(tf.add(term1, term2)), axis=axis_sum)


def log_pdf(x, pdf_type, pdf_params, axis_sum=None):
    if pdf_type.lower() == 'gaussian':
        return log_gaussian(x, mean=pdf_params[0], std=pdf_params[1], axis_sum=axis_sum)
    elif pdf_type.lower() == 'gaussian_mixture':
        return log_gaussian_mixture(x, mean1=pdf_params[0], mean2=pdf_params[1],
                                    std1=pdf_params[2], std2=pdf_params[3],
                                    prob1=pdf_params[4], axis_sum=axis_sum)
    else:
        return ValueError('This function supports only gaussian and gaussian_mixture pdfs.')


def lhs_gaussian(shape, ns, mean=0., std=1.):
    lhd = lhs(int(np.prod(shape)), samples=ns)
    rv = norm(loc=0., scale=1.).ppf(lhd).reshape(((ns,) + shape))
    return mean + std * rv


def lhs_gaussian_mixture(shape, ns, prob1, mean1=0., std1=1., mean2=0., std2=1.):
    lhd = lhs(int(np.prod(shape)), samples=ns)
    rv = norm(loc=0., scale=1.).ppf(lhd).reshape(((ns,) + shape))
    inds = np.random.binomial(1, prob1, size=(ns,) + shape)
    means = mean1 * inds + mean2 * (1-inds)
    stds = std1 * inds + std2 * (1-inds)
    return means + stds * rv


def lhs_pdf(shape, ns, pdf_type, pdf_params):
    if pdf_type.lower() == 'gaussian':
        return lhs_gaussian(shape, ns, mean=pdf_params[0], std=pdf_params[1])
    elif pdf_type.lower() == 'gaussian_mixture':
        return lhs_gaussian_mixture(shape, ns, mean1=pdf_params[0], mean2=pdf_params[1],
                                    std1=pdf_params[2], std2=pdf_params[3],
                                    prob1=pdf_params[4])
    else:
        return ValueError('This function supports only gaussian and gaussian_mixture pdfs.')


def batch(iterable, n):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


# Random utils functions


def preprocess_prior(prior_type, prior_params, n_layers):
    # preprocess the prior inputs
    if prior_type.lower() not in ['gaussian', 'gaussian_mixture']:
        raise ValueError('The prior type should be "gaussian" or "gaussian_mixture".')
    if not isinstance(prior_params, tuple) and (not isinstance(prior_params, list)):
        raise ValueError('Input prior_params should be a tuple or list of tuples.')
    if isinstance(prior_params, list):
        if len(prior_params) != n_layers:
            raise ValueError
        if any([not isinstance(params, tuple) for params in prior_params]):
            raise TypeError
    if prior_type.lower() == 'gaussian':
        if isinstance(prior_params, tuple):
            prior_params = [(float(prior_params[0]), float(prior_params[1]))] * n_layers
        elif isinstance(prior_params, list):
            prior_params = [(float(params[0]), float(params[1])) for params in prior_params]
        else:
            raise TypeError
    if prior_type.lower() == 'gaussian_mixture':
        if isinstance(prior_params, tuple):
            prior_params = [(float(prior_params[0]), float(prior_params[1]), float(prior_params[2]),
                             float(prior_params[3]), float(prior_params[4]))] * n_layers
        elif isinstance(prior_params, list):
            prior_params = [(float(params[0]), float(params[1]), float(params[2]), float(params[3]), float(params[4]))
                            for params in prior_params]
        else:
            raise TypeError
    return prior_params


def compute_weights_shapes(hidden_units, input_dim, output_dim):
    # compute shape of kernels and biases
    all_units = hidden_units + (output_dim, )
    kernel_shapes = [(input_dim, all_units[0])]
    for j, units in enumerate(all_units[:-1]):
        kernel_shapes.append((units, all_units[j + 1]))
    bias_shapes = [(units,) for units in all_units]
    return kernel_shapes, bias_shapes


def compute_glorot_normal_variance(kernel_shapes):
    # Glorot Normal variance = 2/(fan_in+fan_out)
    variances = [2 / (shape_kernel[0] + shape_kernel[1]) for shape_kernel in kernel_shapes]
    return variances


def mean_and_std_from_samples(y_MC, var_aleatoric=0., importance_weights=None):
    y_mean = np.average(y_MC, axis=0, weights=importance_weights)

    ndata = y_MC.shape[1]
    if isinstance(var_aleatoric, int) or isinstance(var_aleatoric, float):
        aleatoric_noise = np.tile(np.array(var_aleatoric).reshape((1, 1)), [ndata, 1])
    elif isinstance(var_aleatoric, np.ndarray):
        aleatoric_noise = np.tile(np.diag(var_aleatoric).reshape((1, -1)), [ndata, 1])
    else:
        raise TypeError('Input var_aleatoric should be a float, int or ndarray.')
    epistemic_noise = np.average((y_MC-np.tile(np.expand_dims(y_mean, 0), [y_MC.shape[0], 1, 1])) ** 2,
                                 axis=0, weights=importance_weights)
    y_std = np.sqrt(aleatoric_noise + epistemic_noise)
    return y_mean, y_std


def return_outputs(y_mean, y_std, y_MC, return_mean, return_std, return_MC):
    outputs = []
    if return_mean:
        outputs.append(y_mean)
    if return_std:
        outputs.append(y_std)
    if return_MC:
        outputs.append(y_MC)
    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


def std_from_samples(ndata, y_MC=None, var_aleatoric=None, importance_weights=None):
    aleatoric_noise, epistemic_noise = 0., 0.
    if var_aleatoric is not None:
        if isinstance(var_aleatoric, int) or isinstance(var_aleatoric, float):
            aleatoric_noise = np.tile(np.array(var_aleatoric).reshape((1, 1)), [ndata, 1])
        if isinstance(var_aleatoric, np.ndarray):
            aleatoric_noise = np.tile(np.diag(var_aleatoric).reshape((1, -1)), [ndata, 1])
    if y_MC is not None:
        epistemic_noise = np.var(y_MC, axis=0)
    return np.sqrt(aleatoric_noise + epistemic_noise)


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


def weights_from_layers_to_vector(weights):
    weights = np.vstack([w.reshape((-1,)) for w in weights])
    return weights


def weights_from_vector_to_layers(weight_vector, kernel_shapes, bias_shapes):
    weights = []
    position = 0
    for l in range(len(kernel_shapes)):
        weights.append(weight_vector[position:position + np.prod(kernel_shapes[l])].reshape(kernel_shapes[l]))
        position += np.prod(kernel_shapes[l])
        weights.append(weight_vector[position:position + np.prod(bias_shapes[l])].reshape(bias_shapes[l]))
        position += np.prod(kernel_shapes[l])
    return weights


# Functions for alpha black box

def factor_params_gaussian(var_prior, mu_post, var_post, ndata):
    var_factor = 1. / (1. / ndata * (1. / var_post - 1. / var_prior))
    mu_over_var_factor = 1. / ndata * tf.divide(mu_post, var_post)
    return var_factor, mu_over_var_factor


# Plot functions

def plot_mean_std(x, y_std, ax, y_mean=None, var_aleatoric=None,
                     color_mean='black', linestyle_mean='-.', color_std='gray', alpha_std=.3,
                     label_mean='mean prediction', label_std='aleatoric + epistemic',
                     label_var_n='epistemic_uncertainty'):
    assert (len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[1] == 1))
    assert (len(y_std.shape) == 1 or (len(y_std.shape) == 2 and y_std.shape[1] == 1))

    if y_mean is not None:
        ax.plot(x, y_mean, color=color_mean, linestyle=linestyle_mean, label=label_mean)
    if var_aleatoric is not None:
        # plot mean +/- 2 std. dev. of the noise term only, in darker shade
        ax.fill(np.concatenate([x, x[::-1]]),
                np.concatenate([y_mean - 2. * np.sqrt(var_aleatoric), (y_mean + 2. * np.sqrt(var_aleatoric))[::-1]]),
                alpha=alpha_std * 2.5, fc=color_std, ec='None', label=label_var_n)
    # plot mean +/- 2 std. dev.
    ax.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_mean - 2. * y_std, (y_mean + 2. * y_std)[::-1]]),
            alpha=alpha_std, fc=color_std, ec='None', label=label_std)


def plot_mean_MC(x, y_MC, ax, y_mean=None, color_mean='black', linestyle_mean='-.', color_MC='gray', alpha_MC=.4,
                 label_mean='mean prediction', label_MC='epistemic uncertainty'):

    assert (len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[1] == 1))
    assert (len(y_MC.shape) == 2 or (len(y_MC.shape) == 3) and y_MC.shape[-1] == 1)

    if y_mean is not None:
        ax.plot(x, y_mean, color=color_mean, label=label_mean, linestyle=linestyle_mean)
    for j in range(y_MC.shape[0]):
        if j == 0:
            ax.plot(x, y_MC[j], color=color_MC, alpha=alpha_MC, label=label_MC)
        else:
            ax.plot(x, y_MC[j], color=color_MC, alpha=alpha_MC)



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

