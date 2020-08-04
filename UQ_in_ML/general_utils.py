# Audrey Olivier
# 4/16/2019
# Plotting utility functions

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal, rankdata
from pyDOE import lhs
from scipy.stats.distributions import norm
import tensorflow as tf


# Losses

# these compute -p(xn|theta) for all xn independently, returns a matrix. Need to sum over to get the cost
def homoscedastic_negloglike_scalar_variance(y_true, y_pred, var_n):
    # var_n is a fixed known scalar
    indpt_outputs = tf.reduce_sum(tf.square(y_true-y_pred) / (2. * var_n) + 0.5 * tf.log(2. * np.pi * var_n), axis=-1)
    return indpt_outputs


def homoscedastic_negloglike_full_covariance(y_true, y_pred, inv_cov_n, logdet_cov_n):
    # var_n is a fixed known full covariance, inv_cov_n is the inverse of the covariance matrix
    Y = y_true - y_pred
    indpt_outputs = 0.5 * tf.reduce_sum(tf.multiply(tf.tensordot(Y, inv_cov_n, axes=[[-1], [0]]), Y), axis=-1) + \
                    0.5 * inv_cov_n.shape[0] * tf.log(2 * np.pi) + 0.5 * logdet_cov_n
    return indpt_outputs

# Probability distributions


def log_gaussian(x, mean=0., std=1., axis_sum=None, mask_to_keep=None, module='tf'):
    if isinstance(x, np.ndarray):
        x = x.astype(np.float32)
    log2pi = np.log(2 * np.pi).astype(np.float32)
    if module == 'np':
        multiv_gauss = - 0.5 * log2pi - np.log(std) - 0.5 * ((x - mean) / std) ** 2
        if mask_to_keep is None:
            return np.sum(multiv_gauss, axis=axis_sum)
        return np.sum(mask_to_keep * multiv_gauss, axis=axis_sum)
    multiv_gauss = - 0.5 * log2pi - tf.log(std) - 0.5 * ((x - mean) / std) ** 2
    if mask_to_keep is None:
        return tf.reduce_sum(multiv_gauss, axis=axis_sum)
    return tf.reduce_sum(tf.cast(mask_to_keep, tf.float32) * multiv_gauss, axis=axis_sum)


def log_multiv_gaussian(x, mean, cov_chol, axis_sum=None):
    d = tf.to_float(cov_chol.shape[0])
    log2pi = np.log(2 * np.pi).astype(np.float32)
    inv_chol = tf.matrix_inverse(cov_chol)
    inv_cov = tf.matmul(tf.transpose(inv_chol), inv_chol)
    logdet_cov = 2. * tf.reduce_sum(tf.log(tf.diag_part(cov_chol)))    # log det cov = 2 * sum_{i}(log L_{ii})
    X = x - mean
    multiv_gauss = - 0.5 * tf.reduce_sum(tf.multiply(tf.tensordot(X, inv_cov, axes=[[-1], [0]]), X), axis=-1) \
                   - 0.5 * d * log2pi - 0.5 * logdet_cov
    return tf.reduce_sum(multiv_gauss, axis=axis_sum)


def lhs_gaussian(shape, ns, mean=0., std=1.):
    lhd = lhs(int(np.prod(shape)), samples=ns)
    rv = norm(loc=0., scale=1.).ppf(lhd).reshape(((ns,) + shape))
    return mean + std * rv


def neg_entropy_gauss(std_1, axis_sum=None):
    # Compute E_{p1}[log(p1)] (negative entropy) for a multivariate gaussian with independent dimensions
    neg_entropy = - 0.5 - tf.log(tf.sqrt(2 * np.pi) * std_1)
    return tf.reduce_sum(neg_entropy, axis=axis_sum)


def kl_gauss_gauss(mean_1, std_1, mean_2, std_2, axis_sum=None):
    # Compute KL(p1||p2)=E_{p1}[log(p1/p2)] between multivariate gaussians with independent dimensions
    kl_all = tf.log(std_2 / std_1) + (std_1 ** 2 + (mean_1 - mean_2) ** 2) / (2. * std_2 ** 2) - 0.5
    return tf.reduce_sum(kl_all, axis=axis_sum)


def kl_gauss_mixgauss(mean_1, std_1, mean_2, std_2, compmix_2):
    # Compute KL(p1||p2)=E_{p1}[log(p1/p2)] between a gaussian p1 and a mixture of gaussians p2
    ncomp = tf.to_int32(tf.shape(mean_2)[0])
    nd = len(mean_1.get_shape().as_list())
    kl_gauss = kl_gauss_gauss(
        tf.tile(tf.expand_dims(mean_1, 0), [ncomp, ] + [1, ] * nd),
        tf.tile(tf.expand_dims(std_1, 0), [ncomp, ] + [1, ] * nd),
        mean_2, std_2, axis_sum=list(range(1, 1+nd)))
    weighted_kl_gauss = tf.log(compmix_2) - kl_gauss
    return - tf.reduce_logsumexp(weighted_kl_gauss)


# bootstrap thingy

def sample_data_bootstrap(X, y, type_bootstrap, sample_aleatoric_func=None, nb=1):
    all_xn, all_yn, all_weights = np.empty((nb, ) + X.shape), np.empty((nb, ) + y.shape), np.empty((nb, X.shape[0]))
    # Sample the data
    ndata = X.shape[0]
    for cb in range(nb):
        if type_bootstrap == 'frequentist':
            indices = np.random.choice(ndata, replace=True, size=ndata)
            x_n = np.array([X[i] for i in indices])
            y_n = np.array([y[i] for i in indices])
            weights_data = np.ones((ndata, ))
        elif type_bootstrap == 'bayesian':
            weights_data = np.random.dirichlet([1.] * ndata) * ndata
            x_n, y_n = X.copy(), y.copy()
        elif type_bootstrap == 'none':
            data_tot = np.concatenate([X, y], axis=-1)
            np.random.shuffle(data_tot)
            x_n, y_n = data_tot[:, :X.shape[1]], data_tot[:, X.shape[1]:]
            weights_data = np.ones((ndata,))
        else:
            raise ValueError
        if sample_aleatoric_func is not None:
            y_n = y_n + sample_aleatoric_func(size=(ndata,))
        if nb == 1:
            return x_n, y_n, weights_data
        all_xn[cb, ...] = x_n
        all_yn[cb, ...] = y_n
        all_weights[cb, ...] = weights_data
    return all_xn, all_yn, all_weights


# Preprocessing functions


def compute_weights_shapes(hidden_units, input_dim, output_dim):
    # compute shape of kernels and biases
    all_units = hidden_units + (output_dim, )
    kernel_shapes = [(input_dim, all_units[0])]
    for j, units in enumerate(all_units[:-1]):
        kernel_shapes.append((units, all_units[j + 1]))
    bias_shapes = [(units,) for units in all_units]
    return kernel_shapes, bias_shapes


def compute_standard_sigmas(hidden_units, input_dim, output_dim, scale=1., mode='fan_avg'):
    kernel_shapes, _ = compute_weights_shapes(hidden_units, input_dim, output_dim)
    if mode == 'fan_avg':
        # Glorot Normal std = sqrt(2/(fan_in+fan_out))
        result = [scale * np.sqrt(2 / (shape_kernel[0] + shape_kernel[1])) for shape_kernel in kernel_shapes]
    elif mode == 'fan_in':
        # Lecun std = sqrt(1/fan_in)
        result = [scale * np.sqrt(1 / (shape_kernel[0])) for shape_kernel in kernel_shapes]
    elif mode == 'fan_out':
        result = [scale * np.sqrt(1 / (shape_kernel[1])) for shape_kernel in kernel_shapes]
    else:
        raise ValueError
    return result


def kl_div_gaussians(mean_prior, std_prior, mean_posterior, std_posterior):
    # Compute KL(posterior||prior) between univariate gaussians
    return np.log(std_prior / std_posterior) + \
           0.5 * (std_posterior ** 2 + (mean_posterior-mean_prior) ** 2) / std_prior ** 2 - 0.5


def check_list_input(list_to_check, correct_len, types):
    if not isinstance(list_to_check, (tuple, list)):
        return False
    if len(list_to_check) != correct_len:
        return False
    if any(not isinstance(elmt, types) for elmt in list_to_check):
        return False
    return True


def extract_mask_from_vi_ranking(rank_metric, metric_values, threshold_on_number=None, threshold_on_metric_perc=None,
                                 threshold_on_metric=None):
    """ Given importance metric values and thresholds, compute the mask of important weights """

    # keep only a given number of parameters
    if threshold_on_number is not None:
        if not isinstance(threshold_on_number, int):
            raise TypeError
        ranking = rankdata(-1 * metric_values, method='average')  # rank: 1 is the weight with max value
        ranking_bool = (ranking <= threshold_on_number)
    elif threshold_on_metric_perc is not None:  # keep parameters to achieve a given perc. of the metric
        if not isinstance(threshold_on_metric_perc, (float, int)):
            raise TypeError
        result_vec_sorted = np.sort(metric_values)[::-1]
        if rank_metric.lower() == 'snr':
            result_vec_sorted = np.cumsum(result_vec_sorted ** 2) / np.sum(result_vec_sorted ** 2)
        else:
            result_vec_sorted = np.cumsum(result_vec_sorted) / np.sum(result_vec_sorted)
        n_to_keep = np.sum(result_vec_sorted <= threshold_on_metric_perc)
        ranking = rankdata(-1 * metric_values, method='average')  # rank: 1 is the weight with max value
        ranking_bool = (ranking <= n_to_keep)
    elif threshold_on_metric is not None:
        if not isinstance(threshold_on_metric, (float, int)):
            raise TypeError
        ranking_bool = (metric_values >= threshold_on_metric)
    else:
        #ranking = rankdata(-1 * result_vector, method='average')  # rank: 1 is the weight with max value
        raise ValueError
    return ranking_bool


def do_reset_vi(VI_regressor):
    """
    Reset VI to avoid increase in computational complexity (deleted some of its attributes)
    """
    from .epistemic_regressors import BayesByBackprop, alphaBB, BayesByBackpropMixture
    alpha_value, ncomp = None, None
    if isinstance(VI_regressor, alphaBB):
        alpha_value = VI_regressor.alpha
    if isinstance(VI_regressor, BayesByBackpropMixture):
        ncomp = VI_regressor.ncomp
    VI_inputs = {'input_dim': VI_regressor.input_dim,
                 'output_dim': VI_regressor.output_dim,
                 'var_n': VI_regressor.var_n,
                 'hidden_units': VI_regressor.hidden_units,
                 'activation': VI_regressor.activation}
    if VI_regressor.learn_prior:
        VI_inputs.update({'prior_means': None, 'prior_stds': None})
    else:
        VI_inputs.update({'prior_means': VI_regressor.prior_means,
                          'prior_stds': VI_regressor.prior_stds})
    VI_outputs = (VI_regressor.variational_mu, VI_regressor.variational_sigma, VI_regressor.training_data)
    if VI_regressor.learn_prior:
        VI_outputs_prior = (VI_regressor.variational_prior_mu, VI_regressor.variational_prior_sigma)
    del VI_regressor
    if alpha_value is not None:
        VI_regressor = alphaBB(alpha=alpha_value, **VI_inputs)
    elif ncomp is not None:
        VI_regressor = BayesByBackpropMixture(ncomp=ncomp, **VI_inputs)
    else:
        VI_regressor = BayesByBackprop(**VI_inputs)
    VI_regressor.variational_mu = VI_outputs[0]
    VI_regressor.variational_sigma = VI_outputs[1]
    VI_regressor.training_data = VI_outputs[2]
    if VI_regressor.learn_prior:
        VI_regressor.variational_prior_mu = VI_outputs_prior[0]
        VI_regressor.variational_prior_sigma = VI_outputs_prior[1]
    return VI_regressor


# Utils to process outputs

def generate_seed(nfigures=4, previous_seeds=()):
    """ Generate an integer that can serve as seed
     It uses nfigures and must be different from previous_seeds"""
    if not isinstance(previous_seeds, (list, tuple)):
        raise ValueError
    seed = list(np.random.choice(['1', '2', '3', '4', '5', '6', '7', '8', '9'], replace=False, size=nfigures))
    seed = int(''.join(seed))
    while seed in previous_seeds:
        seed = list(np.random.choice(['1', '2', '3', '4', '5', '6', '7', '8', '9'], replace=False, size=nfigures))
        seed = int(''.join(seed))
    return seed


def generate_seeds(nseeds, nfigures=4, previous_seeds=()):
    seeds = []
    if isinstance(previous_seeds, tuple):
        previous_seeds = list(previous_seeds)
    elif not isinstance(previous_seeds, list):
        raise ValueError
    for _ in range(nseeds):
        seed = generate_seed(nfigures=nfigures, previous_seeds=seeds + previous_seeds)
        seeds.append(seed)
    return seeds


def mean_and_std_from_samples(y_MC, var_aleatoric=0., importance_weights=None):
    nMC, ndata, ny = y_MC.shape
    # Compute mean
    y_mean = np.average(y_MC, axis=0, weights=importance_weights)
    # Compute std (aleatoric + epistemic uncertainty)
    if isinstance(var_aleatoric, int) or isinstance(var_aleatoric, float):
        aleatoric_noise = np.tile(np.array(var_aleatoric).reshape((1, 1)), [ndata, 1])
    elif isinstance(var_aleatoric, np.ndarray):
        aleatoric_noise = np.tile(np.diag(var_aleatoric).reshape((1, -1)), [ndata, 1])
    else:
        raise TypeError('Input var_aleatoric should be a float, int or ndarray.')
    epistemic_noise = np.average((y_MC - y_mean) ** 2, axis=0, weights=importance_weights)
    y_std = np.sqrt(aleatoric_noise + epistemic_noise)
    return y_mean, y_std


def percentiles_from_samples(y_MC, var_aleatoric=0., importance_weights=None, percentiles=(2.5, 97.5)):
    nMC, ndata, ny = y_MC.shape
    new_samples = np.copy(y_MC)
    if importance_weights is not None:
        new_samples = resample(new_samples, weights=importance_weights)
    if var_aleatoric != 0.:
        new_samples = add_aleatoric_noise(y_MC, var_aleatoric)
    result = np.empty(shape=(len(percentiles), ndata, ny))
    for n in range(ny):
        result[:, :, n] = np.percentile(new_samples[:, :, n], q=percentiles, axis=0, overwrite_input=True,
                                        interpolation='linear', keepdims=False)
    return result


def resample(samples, weights, method='multinomial', size=None):
    nsamples = samples.shape[0]
    if size is None:
        size = nsamples
    if method == 'multinomial':
        idx = np.random.choice(nsamples, size=size, p=weights, replace=True)
        output = samples[idx]
        return output
    else:
        raise ValueError('Exit code: Current available method: multinomial')


def add_aleatoric_noise(y_MC, var_aleatoric):
    nMC, ndata, ny = y_MC.shape
    if isinstance(var_aleatoric, int) or isinstance(var_aleatoric, float):
        noise_aleatoric = np.sqrt(var_aleatoric) * np.random.normal(size=(nMC, ndata, ny))
    elif isinstance(var_aleatoric, np.ndarray):
        noise_aleatoric = np.random.multivariate_normal(mean=np.zeros(ny), cov=var_aleatoric, size=(nMC, ndata))
    else:
        raise TypeError
    return y_MC + noise_aleatoric


def compute_and_return_outputs(y_MC, return_std=False, return_percentiles=(), return_MC=0, var_aleatoric=0.,
                               importance_weights=None, aleatoric_in_std_perc=True, aleatoric_in_MC=True):
    outputs = []
    if not aleatoric_in_std_perc:
        var_aleatoric_ = 0.
    else:
        var_aleatoric_ = var_aleatoric
    y_mean, y_std = mean_and_std_from_samples(y_MC, var_aleatoric=var_aleatoric_, importance_weights=importance_weights)
    outputs.append(y_mean)
    if return_std:
        outputs.append(y_std)
    if len(return_percentiles) > 0:
        percentiles = percentiles_from_samples(y_MC, var_aleatoric=var_aleatoric_,
                                               importance_weights=importance_weights, percentiles=return_percentiles)
        outputs.append(percentiles)
    if return_MC > 0:
        if aleatoric_in_MC:
            outputs.append(add_aleatoric_noise(y_MC[:return_MC], var_aleatoric))
        else:
            outputs.append(y_MC[:return_MC])
    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


# Plot functions

def plot_mean_std(x, y_std, ax, y_mean=None, var_aleatoric=None,
                  color_mean='black', linestyle_mean='-.', label_mean='mean prediction',
                  color_uncertainty='gray', alpha_uncertainty=.3, label_std='aleatoric + epistemic',
                  label_var_n='aleatoric uncertainty'):
    assert (len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[1] == 1))
    assert (len(y_std.shape) == 1 or (len(y_std.shape) == 2 and y_std.shape[1] == 1))

    if y_mean is not None:
        ax.plot(x, y_mean, color=color_mean, linestyle=linestyle_mean, label=label_mean)
    if var_aleatoric is not None:
        # plot mean +/- 2 std. dev. of the noise term only, in darker shade
        ax.fill(np.concatenate([x, x[::-1]]),
                np.concatenate([y_mean - 2. * np.sqrt(var_aleatoric), (y_mean + 2. * np.sqrt(var_aleatoric))[::-1]]),
                alpha=alpha_uncertainty * 2.5, fc=color_uncertainty, ec='None', label=label_var_n)
    # plot mean +/- 2 std. dev.
    ax.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_mean - 2. * y_std, (y_mean + 2. * y_std)[::-1]]),
            alpha=alpha_uncertainty, fc=color_uncertainty, ec='None', label=label_std)
    ax.set_xlabel(r'input $X$', fontsize=14)
    ax.set_ylabel(r'output $y$', fontsize=14)


def plot_mean_percentiles(x, y_perc, ax, y_mean=None, var_aleatoric=None, percentiles=(2.5, 97.5),
                          color_mean='black', linestyle_mean='-.', label_mean='mean prediction',
                          color_uncertainty='gray', alpha_uncertainty=.3, label_std='aleatoric + epistemic',
                          label_var_n='aleatoric uncertainty'):
    assert len(percentiles) == 2
    assert (len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[1] == 1))
    assert (len(y_perc.shape) == 2) or (len(y_perc.shape) == 3 and y_perc.shape[-1] == 1)
    assert (y_perc.shape[0] == len(percentiles))

    if y_mean is not None:
        ax.plot(x, y_mean, color=color_mean, linestyle=linestyle_mean, label=label_mean)
    if (var_aleatoric is not None) and percentiles == (2.5, 97.5):
        # plot mean +/- 2 std. dev. (equiv. to the 2.5 and 97.5 percentiles) of the noise term only, in darker shade
        ax.fill(np.concatenate([x, x[::-1]]),
                np.concatenate([y_mean - 2. * np.sqrt(var_aleatoric), (y_mean + 2. * np.sqrt(var_aleatoric))[::-1]]),
                alpha=alpha_uncertainty * 2.5, fc=color_uncertainty, ec='None', label=label_var_n)
    # plot mean +/- percentiles
    ax.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_perc[0], y_perc[1][::-1]]),
            alpha=alpha_uncertainty, fc=color_uncertainty, ec='None', label=label_std)
    ax.set_xlabel(r'input $X$', fontsize=14)
    ax.set_ylabel(r'output $y$', fontsize=14)


def plot_mean_MC(x, y_MC, ax, y_mean=None, color_mean='black', linestyle_mean='-.', label_mean='mean prediction',
                 color_MC='gray', alpha_MC=.4, label_MC='one posterior draw'):

    assert (len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[1] == 1))
    assert (len(y_MC.shape) == 2 or (len(y_MC.shape) == 3) and y_MC.shape[-1] == 1)

    if not isinstance(color_MC, list):
        color_MC = [color_MC]
    if y_mean is not None:
        ax.plot(x, y_mean, color=color_mean, label=label_mean, linestyle=linestyle_mean)
    ax.plot(x, y_MC[0], color=color_MC[0], alpha=alpha_MC, label=label_MC)
    for j in range(1, len(color_MC)):
        ax.plot(x, y_MC[j], color=color_MC[j], alpha=alpha_MC)
    for j in range(len(color_MC), y_MC.shape[0]):
        ax.plot(x, y_MC[j], color=color_MC[-1], alpha=alpha_MC)
    ax.set_xlabel(r'input $X$', fontsize=14)
    ax.set_ylabel(r'output $y$', fontsize=14)


def plot_2d_gaussian(mean, cov, xlim, ylim, levels=5, ax=None, figsize=(5, 4), filled=False, cmap='Greys',
                     xlabel=r'$x_{1}$', ylabel=r'$x_{2}$', fontsize=14):
    x, y = np.mgrid[np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100)]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    rv = multivariate_normal(mean, cov)
    return_both = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return_both = True
    if filled:
        ax.contourf(x, y, rv.pdf(pos), levels, cmap=cmap)
    else:
        ax.contour(x, y, rv.pdf(pos), levels, cmap=cmap)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.axis('square')
    ax.set_xlabel(xlabel, fontsize=fontsize); ax.set_ylabel(ylabel, fontsize=fontsize)
    if return_both:
        return fig, ax
    return ax


def cov_to_corr(cov):
    Dinv = np.diag(1/np.sqrt(np.diag(cov)))
    return np.matmul(np.matmul(Dinv, cov), Dinv)


def corr_to_cov(corr, stds):
    D = np.diag(stds)
    return np.matmul(np.matmul(D, corr), D)


def plot_covariance_matrix(cov, type_cov='correlation', ax=None, labels=None, fontsize=14, cmap=None,
                           vmin=None, vmax=None):
    d = cov.shape[0]
    if labels is None:
        labels = [r'$x_{}$'.format(i+1) for i in range(d)]
    assert len(labels) == d
    return_both=False
    if ax is None:
        fig, ax = plt.subplots(figsize=(d + 0.2 * d, d))
        return_both = True
    if type_cov == 'covariance' and cmap is None:
        cmap = 'Blues'
    if type_cov == 'correlation' and cmap is None:
        cmap = 'RdYlBu'
    cm = plt.cm.get_cmap(cmap)
    x, y = np.mgrid[range(d), range(d)]
    x, y = x.reshape((-1,)), y.reshape((-1,))
    z = np.reshape(np.flip(cov.T, axis=1), x.shape)
    if type_cov == 'covariance':
        mask = [True if xi >= yi else False for (xi, yi) in zip(x, np.flip(y, axis=0))]
        sc = ax.scatter(x=x[mask], y=y[mask], marker='s', c=z[mask], s=700 * np.sqrt(d), cmap=cm)
    elif type_cov == 'correlation':
        mask = [True if xi > yi else False for (xi, yi) in zip(x, np.flip(y, axis=0))]
        sc = ax.scatter(x=x[mask], y=y[mask], marker='s', c=z[mask], s=700 * np.sqrt(d), cmap=cm, vmin=-1, vmax=1)
    else:
        raise ValueError
    ax.set_xticks(range(d), minor=False)
    ax.set_yticks(range(d), minor=False)
    ax.set_xticks([d_ - 0.5 for d_ in range(d + 1)], minor=True)
    ax.set_yticks([d_ - 0.5 for d_ in range(d + 1)], minor=True)
    ax.xaxis.tick_top()
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_yticklabels(labels[::-1], fontsize=fontsize)
    ax.grid(False, 'major'); ax.grid(True, 'minor')
    plt.colorbar(sc, ax=ax)
    if return_both:
        return fig, ax
    return ax


from scipy.stats import norm
import statsmodels.api as sm
def plot_univariate_pdf(samples=None, weights=None, mean=None, std=None, ax=None, figsize=(4, 4), xrange=None,
                        **kwargs):
    return_both = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return_both = True
    if samples is None:
        if xrange is None:
            xrange = [mean - 4 * std, mean + 4 * std]
        domain = np.linspace(xrange[0], xrange[1], 200)
        rv = norm(loc=mean, scale=std)
        ax.plot(domain, rv.pdf(domain), **kwargs)
    else:
        if xrange is None:
            range = np.amax(samples)-np.amin(samples)
            xrange = [np.amin(samples)-0.02*range, np.amax(samples)+0.02*range]
        if weights is not None:
            indices = np.random.choice(samples.size, p=weights, size=2000)
            samples = samples[indices]
        dens = sm.nonparametric.KDEUnivariate(samples.astype(np.float64))
        dens.fit()
        domain = np.linspace(xrange[0], xrange[1], 200)
        pdf = [dens.evaluate(d) for d in domain]
        ax.plot(domain, pdf, **kwargs)
    if return_both:
        return fig, ax
    return ax


def plot_ranking_weights(ranking_weights, figsize=(7, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    xlabels = []
    for j, rw in enumerate(ranking_weights):
        ax.scatter((j+1) * np.ones((rw.size, )), rw.reshape((-1, )), marker='x', color='grey')
        if j % 2 == 0:
            xlabels.append(r'$W_{}$'.format(j//2+1))
        else:
            xlabels.append(r'$b_{}$'.format(j//2+1))
    ax.set_xticks(range(1, len(ranking_weights)+1), minor=False)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.grid(True, 'major')
    return fig, ax


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

