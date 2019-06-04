# Audrey Olivier, 05/01/2019
# Functions for MCMC
# See: Markov chain Monte Carlo simulation using the DREAM software package: Theory, concepts, and MATLAB
# implementation, Vrugt, 2016


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import chi2, norm


def am(log_pdf, seed, T, verbose=False):
    """Adaptive Metropolis Hastings"""
    d = seed.shape[-1]
    x = np.empty(shape=(T, d))
    logp_x = np.empty(shape=(T, ))
    # first iteration
    C = (2.38 / np.sqrt(d)) ** 2 * np.eye(d)
    x[0, :] = seed
    logp_x[0] = log_pdf(x[0, :])
    # dynamic part: evolution of chain
    acceptance_ratio = 0
    for iter_ in range(1, T):
        # adapt covariance
        if iter_ % 10 == 0:
            C = (2.38/np.sqrt(d))**2 * (np.cov(x[:iter_, :].T) + 1e-4 * np.eye(d))
        # sample and evaluate candidate
        candidate = x[iter_-1, :] + np.random.multivariate_normal(mean=np.zeros(shape=(d,)), cov=C, size=1).reshape((d,))
        logp_candidate = log_pdf(candidate)
        p_acc = min(1, np.exp(logp_candidate - logp_x[iter_-1]))
        if p_acc > np.random.rand():  # accept candidate
            x[iter_, :] = candidate
            logp_x[iter_] = logp_candidate
            acceptance_ratio += 1
        else:
            x[iter_, :] = x[iter_-1, :]
            logp_x[iter_] = logp_x[iter_-1]
    acceptance_ratio /= T
    if verbose:
        print('AM done, acceptance ratio of chain = {}'.format(acceptance_ratio))
    return x, logp_x, acceptance_ratio


def de_mc(log_pdf, seed, T, verbose=False):
    """Differential Evolution Markov Chain (DE-MC) algorithm"""
    assert (len(seed.shape)==2) and (seed.shape[0] >= 2)
    N, d = seed.shape
    gamma_rwm = 2.38/np.sqrt(2*d)
    x = np.empty(shape=(T, N, d))
    logp_x = np.empty(shape=(T, N))
    # first iteration
    x[0, :, :] = seed
    logp_x[0, :] = [log_pdf(x[0, j, :]) for j in range(N)]
    R = np.array([np.setdiff1d(np.arange(N), j) for j in range(N)])
    # dynamic part: evolution of chain
    acceptance_ratios = [0] * N
    for iter_ in range(1, T):
        draw = np.argsort(np.random.rand(N-1, N), axis=0)
        g = np.random.choice([gamma_rwm, 1], size=1, replace=True, p=[0.9, 0.1])
        for j in range(N):
            a, b = R[j, draw[0, j]], R[j, draw[1, j]]
            candidate = x[iter_-1, j, :] + g * (x[iter_-1, a, :]-x[iter_-1, b, :]) + 1e-6 * np.random.normal(size=(d,))
            logp_candidate = log_pdf(candidate)
            p_acc = min(1, np.exp(logp_candidate-logp_x[iter_-1, j]))
            if p_acc > np.random.rand():
                x[iter_, j, :] = candidate
                logp_x[iter_, j] = logp_candidate
                acceptance_ratios[j] = acceptance_ratios[j] + 1
            else:
                x[iter_, j, :] = x[iter_-1, j, :]
                logp_x[iter_, j] = logp_x[iter_-1, j]
    acceptance_ratios = [acceptance_ratio/T for acceptance_ratio in acceptance_ratios]
    if verbose:
        print('DE-MC done, acceptance ratios of chains = '+', '.join([str(acc) for acc in acceptance_ratios]))
    return x, logp_x, acceptance_ratios


def post_process_samples(chains, burnin=0, jump=1, concatenate_chains=True):
    tmp = chains[burnin::jump]
    if (len(chains.shape) == 3) and concatenate_chains:
        return tmp.reshape((tmp.shape[0] * tmp.shape[1], tmp.shape[2]))
    else:
        return tmp


def diagnostics_several_chains(chains, verbose=True):
    """See Gelman and Rubin (1992)"""
    if len(chains.shape) == 2:
        chains = chains.reshape((chains.shape[0], chains.shape[1], 1))
    elif chains.shape[2] > 5:
        raise ValueError("Can't run this diagnostics for more than 5 outputs.")
    nsamples, N, d = chains.shape
    Rhats = np.empty(shape=(d,))
    for d_ in range(d):
        variances = [np.var(chains[:, i, d_]) for i in range(N)]
        W = np.mean(variances)
        means = [np.mean(chains[:, i, d_]) for i in range(N)]
        B = nsamples * np.var(means)
        Vhat = (1 - 1/nsamples) * W + B / nsamples
        Rhat = np.sqrt(Vhat / W)
        if verbose:
            print('Rhat = {}; if > 1.2, continue running the chain!'.format(Rhat))
        Rhats[d_] = Rhat
    return Rhats


def diagnostics_one_chain(chain, verbose=True, figsize=(8, 4), eps_ESS=0.05, alpha_ESS=0.05):
    if len(chain.shape) == 1:
        chain = chain.reshape((chain.shape[0], 1))
    elif chain.shape[2] > 5:
        raise ValueError("Can't run this diagnostics for more than 5 outputs.")
    nsamples, d = chain.shape
    # Computation of ESS and min ESS
    bn = np.ceil(nsamples ** (1 / 2))  # nb of samples per bin
    an = int(np.ceil(nsamples / bn))  # nb of bins, for computation of
    idx = np.array_split(np.arange(nsamples), an)
    means_subdivisions = np.empty((an, chain.shape[1]))
    for i, idx_i in enumerate(idx):
        x_sub = chain[idx_i, :]
        means_subdivisions[i, :] = np.mean(x_sub, axis=0)
    Omega = np.cov(chain.T)
    Sigma = np.cov(means_subdivisions.T)
    joint_ESS = nsamples * np.linalg.det(Omega) ** (1 / d) / np.linalg.det(Sigma) ** (1 / d)
    chi2_value = chi2.ppf(1 - alpha_ESS, df=d)
    min_joint_ESS = 2 ** (2 / d) * np.pi / (d * gamma(d / 2)) ** (
            2 / d) * chi2_value / eps_ESS ** 2
    marginal_ESS = np.empty((d,))
    min_marginal_ESS = np.empty((d,))
    for j in range(d):
        marginal_ESS[j] = nsamples * Omega[j, j] / Sigma[j, j]
        min_marginal_ESS[j] = 4 * norm.ppf(alpha_ESS / 2) ** 2 / eps_ESS ** 2
    if verbose:
        print('Univariate Effective Sample Size in each dimension:')
        for j in range(d):
            print('Parameter # {}: ESS = {}, minimum ESS recommended = {}'.
                  format(j + 1, marginal_ESS[j], min_marginal_ESS[j]))
        print('\nMultivariate Effective Sample Size:')
        print('Multivariate ESS = {}, minimum ESS recommended = {}'.format(joint_ESS, min_joint_ESS))
    # Output plots
    if figsize is None:
        figsize = (20, 4 * d)
    fig, ax = plt.subplots(nrows=d, ncols=3, figsize=figsize)
    for j in range(chain.shape[1]):
        ax[j, 0].plot(np.arange(nsamples), chain[:, j])
        ax[j, 0].set_title('chain - parameter # {}'.format(j + 1))
        ax[j, 1].plot(np.arange(nsamples), np.cumsum(chain[:, j]) / np.arange(nsamples))
        ax[j, 1].set_title('parameter convergence')
        ax[j, 2].acorr(chain[:, j] - np.mean(chain[:, j]), maxlags=50, normed=True)
        ax[j, 2].set_title('correlation between samples')
    return fig, ax
