# From https://brendanhasz.github.io/2018/12/03/tfp-regression.html#variational-model
# and https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/HamiltonianMonteCarlo
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import tensorflow_probability as tfp

from UQ_in_ML.epistemic_regressors import Regressor
from UQ_in_ML.general_utils import plot_mean_uq

name = 'sinusoid'

if name == 'cubic':
    # Data settings
    var_n = 0.05 ** 2
    f = lambda x, noisy: 2 * x ** 3 + noisy*np.sqrt(var_n)*np.random.normal(size=x.shape)
    n_data = 20
    xn = np.array([-0.02519606, -0.29152739, -0.60474655, 0.31944225, -0.08100553, -0.24830156, 0.57461577,
                   0.50232181, 0.60433894, -0.02046175, 0.53479088, -0.65367602, -0.06110107, 0.46652892,
                   -0.66163461, 0.26793157, 0.20481661, -0.24144274, -0.42398829, -0.52080597]).reshape((-1, 1))
    yn = np.array([0.04928457864952569, -0.11915410490457669, -0.405097551770553, 0.029554098140267056,
                   -0.013086956159543405, -0.017770100521146612, 0.42280077037504055, 0.1944984572601308,
                   0.4534092801344878, -0.05744532400253988, 0.27416952296635494, -0.6450129511010473,
                   -0.00434618253501617, 0.16330603887330705, -0.5274704221475347, 0.02189741180766931,
                   0.012647796994763167, 0.08367359752673682, -0.10875986459325471,
                   -0.2964629150726794]).reshape((-1, 1))
    x_plot = np.linspace(-1, 1, 100).reshape((-1, 1))
    x_away_data = np.array([-0.95, -0.75, 0.1, 0.75, 0.95]).reshape((-1, 1))
    # NN setting
    nn = {'input_dim': 1,
          'output_dim': 1,
          'var_n': var_n,
          'hidden_units': (100,),
          'activation': tf.nn.relu,
          'prior_means': 0.,
          'prior_stds': 1.}
    # HMC Settings
    n_burnin = int(2e3)  # number of burn-in steps
    n_jump = 20 #15
    step_size = 0.002
    num_leapfrog_steps = 10
elif name == 'sinusoid':
    var_n = 0.05 ** 2
    f = lambda x, noisy: 0.4 * np.sin(4 * x) + 0.5 * np.cos(12 * x) + noisy * np.sqrt(var_n) * np.random.normal(
        size=x.shape)
    n_data = 20
    xn = 1 / 5 * np.array(
        [[-1.23374131], [-2.80075168], [-3.44975161], [-3.21186531], [-3.02209586], [-2.21373764], [-4.37775872],
         [-1.86404408], [-1.69559337], [-1.37603453], [3.72937], [4.88684951], [2.23649116], [2.69073348],
         [1.29613003], [3.95513359], [3.83697005], [1.45216711], [4.42576355], [2.07597818]])
    yn = np.array([-0.8181956697783953, 0.13885402687227927, -0.3862388281350327, -0.20234915578844812,
                   0.03000639994622538, -0.06849287803937654, -0.07642553448272046, -0.5232257730716847,
                   -0.7023192220709934, -0.8190169865472378, -0.3285415162946372, 0.05488363929789496,
                   0.6563808735923208, 0.6941414577967285, -0.08382934737250632, -0.47515928440184424,
                   -0.5146741885584126, -0.11555280503711199, -0.31140762228445024, 0.4945424732107972]).reshape(
        (-1, 1))
    x_plot = np.linspace(-1., 1., 100).reshape((-1, 1))
    x_away_data = np.array([-0.95, -0.75, -0.1, 0.05, 0.65]).reshape((-1, 1))
    # NN setting
    nn = {'input_dim': 1,
          'output_dim': 1,
          'var_n': var_n,
          'hidden_units': (20, 20, 20),
          'activation': tf.nn.relu,
          'prior_means': 0.,
          'prior_stds': 1.}
    # HMC Settings
    n_burnin = int(2e3)  # number of burn-in steps
    n_jump = 40
    step_size = 0.0005
    num_leapfrog_steps = 10
else:
    raise ValueError
reg_0 = Regressor(**nn)


def target_log_prob_fn(w):
    nchains = w.shape[0]
    network_weights = []
    nc = 0
    for ws, wd in zip(reg_0.weights_shape, reg_0.weights_dim):
        #w_ = tf.expand_dims(tf.reshape(w[nc:nc + wd], ws), 0)
        w_ = tf.reshape(w[:, nc:nc + wd], (nchains, ) + ws)
        network_weights.append(w_)
        nc += wd
    preds = reg_0.compute_predictions(X=tf.constant(xn.astype(np.float32)), network_weights=network_weights)
    log_likelihood = -1. * reg_0.neg_log_like(
        y_true=tf.constant(yn.astype(np.float32)), y_pred=preds, do_sum=True, axis_sum=-1)
    log_prior_value = reg_0.log_prior_pdf(network_weights=network_weights, sum_over_ns=False)
    return log_prior_value + log_likelihood


def compute_diagnostics(states_, x_away_data_=None):
    if x_away_data_ is not None:
        # Diagnostics on outputs
        with tf.compat.v1.Session() as sess:
            y_MC = np.zeros((states_.shape[0], states_.shape[1], x_away_data_.size))
            for j in range(nchains):
                network_weights = []
                nc = 0
                for ws, wd in zip(reg_0.weights_shape, reg_0.weights_dim):
                    network_weights.append(states_[:, j, nc:nc + wd].reshape((-1,) + ws))
                    nc += wd
                y_MC[:, j, :] = sess.run(
                    reg_0.compute_predictions(
                        X=tf.constant(x_away_data_.astype(np.float32)), network_weights=network_weights))[:, :, 0]
            ess_outpt, rhat_outpt = sess.run(
                [tfp.mcmc.effective_sample_size(y_MC),
                 tfp.mcmc.diagnostic.potential_scale_reduction(y_MC)
                 ])
        return np.sum(ess_outpt, axis=0), rhat_outpt, y_MC

    # Diagnostics on parameters
    with tf.compat.v1.Session() as sess:
        # diagnostics for parameters themselves
        ess, rhat = sess.run(
            [tfp.mcmc.effective_sample_size(states=states_),
             tfp.mcmc.diagnostic.potential_scale_reduction(states_)
             ])
    return np.mean(np.sum(ess, axis=0)), np.mean(rhat)


# HMC transition kernel
#kernel = tfp.mcmc.SimpleStepSizeAdaptation(
#    tfp.mcmc.HamiltonianMonteCarlo(
#        target_log_prob_fn=target_log_prob_fn, step_size=0.001, num_leapfrog_steps=10),
#    num_adaptation_steps=n_burnin, target_accept_prob=0.75)

kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=target_log_prob_fn, step_size=step_size, num_leapfrog_steps=num_leapfrog_steps)
#kernel = tfp.mcmc.RandomWalkMetropolis(target_log_prob_fn=target_log_prob_fn)

t0 = time.time()
# Run the chains
nchains = 5
num_iterations = 2500
current_state = 0.1 * np.random.randn(nchains, reg_0.n_weights).astype(np.float32)
all_states = np.empty((0, nchains, reg_0.n_weights), dtype=np.float32)
all_is_accepted = np.empty((0, nchains), dtype=int)

flag = 0
while flag == 0 and all_states.shape[0] < 20000:
    with tf.compat.v1.Session() as sess:
        [states, results] = sess.run(
            tfp.mcmc.sample_chain(
                num_results=num_iterations,
                num_burnin_steps=n_burnin,
                num_steps_between_results=n_jump,
                kernel=kernel,
                current_state=current_state,
                #trace_fn=lambda _, pkr: pkr.results.is_accepted
                ))
    all_states = np.concatenate([all_states, states], axis=0)
    all_is_accepted = np.concatenate([all_is_accepted, results.is_accepted.astype(int)], axis=0)
    print('Acceptance rate = {}'.format(100 * np.mean(all_is_accepted)))
    ess_outpt, rhat_outpt, _ = compute_diagnostics(all_states, x_away_data)
    if any(np.isnan(ess_) for ess_ in ess_outpt) or any(np.isnan(rhat_) for rhat_ in rhat_outpt):
        print('there is a problem with a chain')
        flag = 2
    elif any(ess_ < 400 for ess_ in ess_outpt) or any(rhat_ > 1.01 for rhat_ in rhat_outpt):
        num_iterations = 500
        n_burnin = 0
        current_state = states[-1, :, :]
        assert current_state.shape == (nchains, reg_0.n_weights)
        print('min ESS = {} and max Rhat = {}'.format(min(ess_outpt), max(rhat_outpt)))
        print('adding 500 samples')
    else:
        flag = 1
t1 = time.time()
print('Elapsed time for training = {}'.format(t1 - t0))
print('Acceptance rate = {}'.format(100 * np.mean(all_is_accepted)))

ess, rhat = compute_diagnostics(all_states)
print('Diagnostics for parameters')
print('av ESS = {}'.format(ess))
print('av Rhat = {}'.format(rhat))

ess_outpt, rhat_outpt, y_MC = compute_diagnostics(all_states, x_away_data)
print('Diagnostics for outputs')
for x_, rhat_, ess_ in zip(x_away_data[:, 0], rhat_outpt, ess_outpt):
    print('ess = {}, Rhat = {} at x = {}'.format(ess_, rhat_, x_))

# Show some chains
fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(15, 9))
for i in range(4):
    ax[0, i].plot(all_states[:, 0, i * 10], color='blue', alpha=0.6)
    ax[0, i].plot(all_states[:, 1, i * 10], color='red', alpha=0.6)
    #ax[0, i].plot([n_burnin, n_burnin], [ax[0, i].get_ylim()[0], ax[0, i].get_ylim()[1]])
    #ax[i].plot(network_weights[i].reshape((network_weights[i].shape[0], -1))[:, 0])
    ax[1, i].acorr(all_states[:, 0, i * 10], maxlags=50)
plt.show()

fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(15, 9))
for i in range(4):
    ax[0, i].plot(y_MC[:, 0, i], color='blue', alpha=0.6)
    ax[0, i].plot(y_MC[:, 1, i], color='red', alpha=0.6)
    #ax[0, i].plot([n_burnin, n_burnin], [ax[0, i].get_ylim()[0], ax[0, i].get_ylim()[1]])
    #ax[i].plot(network_weights[i].reshape((network_weights[i].shape[0], -1))[:, 0])
    ax[1, i].acorr(y_MC[:, 0, i], maxlags=50)
plt.show()

#fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(15, 9))
#for i in range(4):
#    ax[0, i].plot(states_[:, i * 10])
#    #ax[i].plot(network_weights[i].reshape((network_weights[i].shape[0], -1))[:, 0])
#    ax[1, i].acorr(states_[:, i * 10], maxlags=50)
#plt.show()
#with tf.compat.v1.Session() as sess:
#    ess = sess.run(tfp.mcmc.effective_sample_size(states_))
#print('av ESS = {} (min={}, max={})'.format(np.mean(ess), min(ess), max(ess)))


# NN weights after burn-in
network_weights = []
nc = 0
for ws, wd in zip(reg_0.weights_shape, reg_0.weights_dim):
    w_ = all_states[:, 0, nc:nc + wd].reshape((-1,) + ws)
    for j in range(1, nchains):
        w_ = np.concatenate([w_, all_states[:, j, nc:nc + wd].reshape((-1,) + ws)], axis=0)
    network_weights.append(w_)
    nc += wd

# Do prediction
y_mean, y_uq, y_MC = reg_0.predict_uq_from_samples(
    X=x_plot, network_weights=network_weights, return_std=True, return_MC=10, return_percentiles=(),
    aleatoric_in_std_perc=True, aleatoric_in_MC=False)

fig, ax = plt.subplots(figsize=(6, 4.))
plot_mean_uq(x=x_plot, y_uq=y_uq, type_uq='std', ax=ax, y_mean=y_mean, var_aleatoric=var_n)
plt.show()


import pickle
with open("MCMC_runs/hmc_{}_20210516_ter.pkl".format(name), "wb") as file:
    results_dict = {
        'nchains': nchains, 'num_iterations': num_iterations, 'n_burnin': n_burnin, 'n_jump': n_jump,
        'time_training': t1 - t0, 'acceptance_rate': 100 * np.mean(all_is_accepted),
        'diagnostics': {'ESS_params': ess, 'Rhat_params': rhat,
                        'ESS_outpt': [(x_, ess_) for x_, ess_
                                    in zip(x_away_data[:, 0], ess_outpt)],
                        'Rhat_outpt': [(x_, ess_) for x_, ess_ in zip(x_away_data[:, 0], rhat_outpt)]},
        'network_weights': network_weights}
    pickle.dump(results_dict, file)
