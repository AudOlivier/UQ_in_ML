# Audrey Olivier
# 4/16/2019

# This code provides algorithms to estimate uncertainties within neural networks for regression problems
# The aleatoric noise is assumed to be homoscedastic and known, see input var_n to all classes.

from UQ_in_ML.general_utils import *


class Regressor:
    """
    Base Regressor class
    This class does some initial checks of the inputs and defines methods pertaining to the prior.

    **Inputs:**

    :param hidden_units: nb. of units in each hidden layer, list of integers
    :param output_dim: nb. of outputs, int
    :param input_dim: nb. of inputs, int
    :param var_n: (co)variance of aleatoric noise, float or ndarray
    :param activation: activations for NN, str or list of str
    :param prior_means: mean value of gaussian prior, float or list (len 2*n_uq_layers) of floats
    :param prior_stds: std. dev. of gaussian prior, float or list (len 2*n_uq_layers) of floats
    :param random_seed: set the random seed generator in the graph, int or None
    """

    def __init__(self, hidden_units, output_dim=1, input_dim=1, var_n=1e-6, activation=tf.nn.relu, prior_means=0.,
                 prior_stds=1., random_seed=None):

        self._init_attributes(hidden_units, output_dim=output_dim, input_dim=input_dim, var_n=var_n,
                              activation=activation, prior_means=prior_means, prior_stds=prior_stds)
        self.random_seed = random_seed
        if not (self.random_seed is None or isinstance(self.random_seed, int)):
            raise ValueError
        self.training_data = None

        # Initialize the tensorflow graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Set random seed
            if self.random_seed is not None:
                tf.set_random_seed(self.random_seed)

            # Initialize placeholders
            self.X_ = tf.placeholder(dtype=tf.float32, name='X_', shape=(None, self.input_dim))  # input data
            self.y_ = tf.placeholder(dtype=tf.float32, name='y_', shape=(None, self.output_dim))  # output data
            self.ns_ = tf.placeholder(dtype=tf.int32, name='ns_', shape=())  # number of samples in MC approx. of cost
            self.w_ = tf.placeholder(dtype=tf.float32, name='w_', shape=(None, ))  # weights for data points

    def _init_attributes(self, hidden_units, output_dim, input_dim, var_n, activation, prior_means, prior_stds):
        self.hidden_units = hidden_units
        self.n_uq_layers = len(self.hidden_units) + 1
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.activation = activation
        # check input and output dims
        if not (isinstance(self.output_dim, int) and isinstance(self.input_dim, int)):
            raise TypeError('Inputs output_dim and input_dim must be integers.')
        # Compute weights shapes
        kernel_shapes, bias_shapes = compute_weights_shapes(
            hidden_units=self.hidden_units, input_dim=self.input_dim, output_dim=self.output_dim)
        self.weights_shape = []
        [self.weights_shape.extend([k, b]) for (k, b) in zip(kernel_shapes, bias_shapes)]
        self.weights_dim = [int(np.prod(w_shape)) for w_shape in self.weights_shape]
        self.n_weights = int(np.sum(self.weights_dim))

        # check the aleatoric noise term
        if isinstance(var_n, np.ndarray):  # var_n is provided as a full covariance (ndarray)
            if len(var_n.shape) == 1:
                var_n = np.diag(var_n)
            if var_n.shape != (self.output_dim, self.output_dim):
                raise ValueError('The size of the var_n matrix should be (ny, ny).')
            self.var_n = var_n
            self.inv_cov_n = np.linalg.inv(self.var_n + 1e-8 * np.eye(self.output_dim)).astype(np.float32)
            _, self.logdet_cov_n = np.linalg.slogdet(self.var_n)
            self.logdet_cov_n = self.logdet_cov_n.astype(np.float32)
        elif isinstance(var_n, (float, int)):  # var_n is a float
            self.var_n = float(var_n)

        # Check the prior
        if prior_means is None and prior_stds is None:  # Learn the prior
            self.learn_prior = True
        else:  # Prior is provided
            self.learn_prior = False
            if isinstance(prior_means, (float, int)):
                self.prior_means = [float(prior_means)] * (2 * self.n_uq_layers)
            elif isinstance(prior_means, (tuple, list)):
                self.prior_means = [float(pm) for pm in prior_means]
            if isinstance(prior_stds, (float, int)):
                self.prior_stds = [float(prior_stds)] * (2 * self.n_uq_layers)
            elif isinstance(prior_stds, (tuple, list)):
                self.prior_stds = [float(pm) for pm in prior_stds]
            for p in [self.prior_means, self.prior_stds]:
                if not (isinstance(p, (list, tuple)) and len(p) == (2 * self.n_uq_layers)
                        and all(isinstance(f, float) for f in p)):
                    raise ValueError('Error defining prior_means or prior_stds')

    def neg_log_like(self, y_true, y_pred, do_sum=True, axis_sum=None, weights_data=None):
        """
        Computes negative log-likelihood -p(D|w)

        :param y_true: data D
        :param y_pred: output of NN with weight(s) w
        :param do_sum: sum over some axes? If not, the output is of same size as y_true - last dimension
        :param axis_sum: axes to sum over. If None, sum over all axes and returns a scalar.
        :param weights_data: weights for all data points

        :return: negative log likelihood -p(D|w)
        """
        if isinstance(self.var_n, float):
            all_neg_log_like = homoscedastic_negloglike_scalar_variance(y_true=y_true, y_pred=y_pred, var_n=self.var_n)
        else:
            all_neg_log_like = homoscedastic_negloglike_full_covariance(
                y_true=y_true, y_pred=y_pred, logdet_cov_n=self.logdet_cov_n, inv_cov_n=self.inv_cov_n)
        if weights_data is not None:
            all_neg_log_like = tf.multiply(weights_data, all_neg_log_like)
        if do_sum:
            return tf.reduce_sum(all_neg_log_like, axis=axis_sum)
        return all_neg_log_like

    def sample_aleatoric(self, size):
        """
        Sample from the aleatoric noise

        :param size: size must be an integer or a tuple

        :return: ndarray of size (size[0], size[1], ..., ny)
        """
        if isinstance(self.var_n, float):
            return np.sqrt(self.var_n) * np.random.standard_normal(size + (self.output_dim, ))
        else:
            return np.random.multivariate_normal(mean=np.zeros((self.output_dim, )), cov=self.var_n, size=size)

    def generate_seed_layers(self, nfigures=4, previous_seeds=()):
        """
        Generate seeds for random sampling, one per network layer
        :param nfigures:
        :param previous_seeds:
        :return:
        """
        random_seeds = []
        for i in range(2 * self.n_uq_layers):
            new_seed = generate_seed(nfigures=nfigures, previous_seeds=random_seeds + list(previous_seeds))
            random_seeds.append(new_seed)
        return random_seeds

    def compute_predictions(self, X, network_weights):
        """
        Computes predictions for given weights and input X.

        :param X: input tensor of shape (n_data, nx)
        :param network_weights: weights of the neural network, list (length 2*n_layers) of ndarrays

        :return: output tensor y, of size (ns, n_data, ny)
        """
        ndata = tf.shape(X)[-2]  # number of independent data points
        ns_local = tf.shape(network_weights[0])[0]
        if len(X.get_shape().as_list()) == 2:
            x = tf.tile(tf.expand_dims(X, 0), [ns_local, 1, 1])
        else:
            x = X

        # Go through the network layers
        for layer in range(self.n_uq_layers - 1):
            # Compute W X + b for layer l
            x = tf.add(tf.matmul(x, network_weights[2 * layer]),
                       tf.tile(tf.expand_dims(network_weights[2 * layer + 1], 1), [1, ndata, 1]))
            # Apply activation function
            x = self.activation(x)
        # Compute W X + b for last layer (linear)
        x = tf.add(tf.matmul(x, network_weights[-2]),
                   tf.tile(tf.expand_dims(network_weights[-1], 1), [1, ndata, 1]))
        return x

    def log_prior_pdf(self, network_weights, sum_over_ns=True):
        """
        Computes log(p(w)) for NN weights w.

        :param network_weights: list (length 2 * n_layers) of ndarrays of shape (ns, ) + w_shape
        :param sum_over_ns: if True, sum prior over ns independent draws, otherwise return a value for all ns weights.

        :return log(p(w))
        """
        log_q = 0.
        # Loop over all layer kernels/biases
        for layer, (w, w_shape, mean, std) in enumerate(zip(
                network_weights, self.weights_shape, self.prior_means, self.prior_stds)):
            # compute the log_pdf for this layer kernel/bias and add it
            log_q += log_gaussian(
                x=w, mean=mean, std=std, axis_sum=(None if sum_over_ns else list(range(1, 1 + len(w_shape)))),
                )
        return log_q

    def sample_weights_from_prior(self, ns, random_seed=None):
        """
        Samples weights for the NN from the prior.

        :param ns
        :param random_seed

        :return weights w (kernels, biases) sampled from the prior p(w), as a list of length 2 * n_layers
        """
        if self.learn_prior:
            raise ValueError
        weights = []
        for i, (w_shape, mean, std) in enumerate(zip(self.weights_shape, self.prior_means, self.prior_stds)):
            w = mean + std * tf.random_normal(
                shape=((ns,) + w_shape), mean=0., stddev=1., seed=(None if random_seed is None else random_seed[i]))
            weights.append(w)
        return weights

    def predict_uq_from_prior(self, X, ns):
        """
        Predict y for new input X, along with uncertainty.

        :param X: input data shape (n_data, nx)
        :param ns: number of samples to use in sampling

        :return y_MC: samples from posterior, ndarray of shape (ns, n_data, ny)
        """
        if self.learn_prior:
            raise NotImplementedError('Prior should be learnt.')
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            network_weights = sess.run(self.sample_weights_from_prior(ns=ns))
            y_MC = sess.run(
                self.compute_predictions(X=self.X_, network_weights=network_weights), feed_dict={self.X_: X})
        return y_MC

    def predict_uq_from_samples(self, X, network_weights, return_std=True, return_MC=10, return_percentiles=(2.5, 97.5),
                                aleatoric_in_std_perc=True, aleatoric_in_MC=False):
        """
        Predict y for new input X, along with uncertainty.

        :param X: input data shape (n_data, nx)
        :param network_weights: list (length 2*n_layers) of ndarrays
        :param return_std: bool, indicates whether to return std. dev.
        :param return_MC: int, nb of MC samples to be returned
        :param return_percentiles: tuple, if (, ) percentiles are not returned
        :param aleatoric_in_std_perc: bool, if True std and perc outputs account for aleatoric uncertainties
        :param aleatoric_in_MC: bool, if True std and perc outputs account for aleatoric uncertainties

        :return y_mean: mean prediction, ndarray of shape (n_data, ny)
        :return y_std: uq prediction (std. dev.), ndarray of shape (n_data, ny)
        :return y_perc: uq prediction (percentiles), ndarray of shape (n_perc, n_data, ny)
        :return y_MC: samples from posterior, ndarray of shape (return_MC, n_data, ny)
        """
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            y_MC = sess.run(self.compute_predictions(X=self.X_, network_weights=network_weights),
                            feed_dict={self.X_: X})
        outputs = compute_and_return_outputs(
            y_MC=y_MC, var_aleatoric=self.var_n, return_std=return_std, return_percentiles=return_percentiles,
            return_MC=return_MC, aleatoric_in_std_perc=aleatoric_in_std_perc, aleatoric_in_MC=aleatoric_in_MC)
        return outputs

    @staticmethod
    def _sigma(rho, module='tf'):
        """ Compute sigma = log(1+exp(rho)) """
        if module == 'np':
            return np.log(1. + np.exp(rho))
        return tf.log(1. + tf.exp(rho))

    @staticmethod
    def _rho(sigma, module='tf'):
        """ Compute rho = log(exp(sigma)-1) """
        if module == 'np':
            return np.log(np.exp(sigma) - 1.)
        return tf.log(tf.exp(sigma) - 1.)


########################################################################################################################
#                                          Variational Inference Regressors                                            #
########################################################################################################################


class VIRegressor(Regressor):
    """ UQ based on variational inference - child class of Regressor

    **Inputs:**

    See Regressor

    :param weights_to_track: number of weights for which convergence can be visualized at the end of training.
                             None or a list (length 2 * n_uq_layers) of ints
    """

    def __init__(self, hidden_units, input_dim=1, output_dim=1, var_n=1e-6, activation='tanh', prior_means=0.,
                 prior_stds=1., random_seed=None, weights_to_track=None):

        # Do the initial checks and computations for the network
        super().__init__(hidden_units=hidden_units, input_dim=input_dim, output_dim=output_dim, var_n=var_n,
                         activation=activation, prior_means=prior_means, prior_stds=prior_stds,
                         random_seed=random_seed)

        # Check extra inputs
        if weights_to_track is not None and not check_list_input(weights_to_track, 2 * self.n_uq_layers, int):
            raise TypeError('Input weights_to_track should be a list of length 2 * n_uq_layers.')
        self.weights_to_track = weights_to_track

        # Initialize the cost
        self.cost = 0.

        # Initialize the histories and other outputs
        self.loss_history = []
        if self.weights_to_track is not None:
            self.variational_mu_history = [np.array([]).reshape((0, nw)) for nw in self.weights_to_track]
            self.variational_sigma_history = [np.array([]).reshape((0, nw)) for nw in self.weights_to_track]
        self.variational_mu, self.variational_sigma = None, None
        if self.learn_prior:
            self.variational_prior_mu, self.variational_prior_sigma = None, None

    def _initialize_variables_in_graph(self):
        """
        Initialize some variables in VI graph
        """
        self.tf_tracked_means = []
        self.tf_tracked_stds = []
        self.tf_variational_mu = []
        self.tf_variational_rho = []
        self.tf_variational_sigma = []

        # add dense layers, add contributions of each layer to prior and variational posterior costs
        standard_sigmas = compute_standard_sigmas(hidden_units=self.hidden_units, input_dim=self.input_dim,
                                                  output_dim=self.output_dim, scale=1., mode='fan_avg')
        start_sigmas = []
        [start_sigmas.extend([std, 0.01]) for std in standard_sigmas]
        for layer, (start_std, w_shape, w_dim) in enumerate(zip(start_sigmas, self.weights_shape, self.weights_dim)):
            # Define the parameters of the variational distribution to be trained: theta={mu, rho} for kernel and bias
            mu = tf.Variable(
                tf.random_normal(shape=w_shape, mean=0., stddev=start_std), trainable=True, dtype=tf.float32)
            rho = tf.Variable(
                -6.9 * tf.ones(shape=w_shape, dtype=tf.float32), trainable=True, dtype=tf.float32)
            sigma = tf.log(1. + tf.exp(rho))

            self.tf_variational_mu.append(mu)
            self.tf_variational_rho.append(rho)
            self.tf_variational_sigma.append(sigma)

            # Keep track of some of the weights
            if self.weights_to_track is not None:
                self.tf_tracked_means.append(tf.reshape(mu, shape=(w_dim, ))[:self.weights_to_track[layer]])
                self.tf_tracked_stds.append(tf.reshape(sigma, shape=(w_dim, ))[:self.weights_to_track[layer]])

    def _initialize_prior_variables_in_graph(self):
        """
        Initialize prior if it is being learnt.
        """
        self.tf_prior_mu = [tf.Variable(0., trainable=True, dtype=tf.float32)
                            for _ in range(2 * self.n_uq_layers - 1)]
        self.prior_means = self.tf_prior_mu + [tf.constant(0.), ]
        self.tf_prior_rho = [tf.Variable(self._rho(10.), trainable=True, dtype=tf.float32)
                             for _ in range(2 * self.n_uq_layers - 1)]
        self.prior_stds = [self._sigma(rho) for rho in self.tf_prior_rho] + [tf.constant(1.), ]

    def sample_weights_from_variational(self, ns, random_seed=None, evaluate_log_pdf=False, sum_over_ns=False):
        """
        Sample weights w for the NN from the variational density q_{theta}(w) (gaussian).

        :return: weights w, as a list of length 2 * n_layers
        """
        weights = []
        log_q = 0.
        for i, (vi_mu, vi_sigma, w_shape) in enumerate(
                zip(self.tf_variational_mu, self.tf_variational_sigma, self.weights_shape)):
            mu = tf.tile(tf.expand_dims(vi_mu, axis=0), [ns, ] + [1, ] * len(w_shape))
            sigma = tf.tile(tf.expand_dims(vi_sigma, axis=0), [ns, ] + [1, ] * len(w_shape))
            w = tf.add(
                mu, tf.multiply(sigma, tf.random_normal(shape=(ns, ) + w_shape, mean=0., stddev=1.,
                                                        seed=(None if random_seed is None else random_seed[i]))))
            weights.append(w)
            if evaluate_log_pdf:
                log_q += log_gaussian(
                    x=w, mean=mu, std=sigma, axis_sum=(None if sum_over_ns else list(range(-len(w_shape), 0))),)

        if evaluate_log_pdf:
            return weights, log_q
        return weights

    def fit(self, X, y, weights_data=None, ns=10, epochs=100, verbose=0, lr=0.001):
        """
        Fit, i.e., find the variational distribution that minimizes the cost function

        :param X: input data, ndarray of shape (n_data, nx)
        :param y: output data, ndarray of shape (n_data, ny)
        :param weights_data: weight of each data point (if bootstrapping), ndarray of shape (n_data)
        :param epochs: int
        :param ns: nb. of samples used in computing cost (expectation over variational distribution)
        :param lr: learning rate for optimizer
        :param verbose
        """
        # Initilize tensorflow session and required variables
        self.training_data = (X, y)
        if weights_data is None:
            weights_data = np.ones((X.shape[0], ))
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            # Run training loop
            for e in range(epochs):
                _, loss_history_ = sess.run(
                    [self.grad_step, self.cost],
                    feed_dict={self.w_: weights_data, self.X_: X, self.y_: y, self.ns_: ns, self.lr_: lr})
                self.loss_history.append(loss_history_)
                # Save some of the weights
                if self.weights_to_track is not None:
                    mean, std = sess.run([self.tf_tracked_means, self.tf_tracked_stds], feed_dict={self.ns_: ns})
                    self.variational_mu_history = [np.concatenate([m1, m2.reshape((1, -1))], axis=0)
                                                   for m1, m2 in zip(self.variational_mu_history, mean)]
                    self.variational_sigma_history = [np.concatenate([m1, m2.reshape((1, -1))], axis=0)
                                                      for m1, m2 in zip(self.variational_sigma_history, std)]
                # print comments on terminal
                if verbose:
                    print('epoch = {}, loss = {}'.format(e, self.loss_history[-1]))

            # Save the final variational parameters
            self.variational_mu, self.variational_sigma = sess.run(
                [self.tf_variational_mu, self.tf_variational_sigma], feed_dict={self.ns_: ns})
            if self.learn_prior:
                self.variational_prior_mu, self.variational_prior_sigma = map(np.array, sess.run(
                    [self.prior_means, self.prior_stds], feed_dict={self.ns_: ns}))
                #self.variational_prior_mu = np.array(self.variational_prior_mu)
                #self.variational_prior_sigma = np.array(self.variational_prior_sigma)
        return None

    def predict_uq(self, X, ns, return_std=True, return_MC=2, return_percentiles=(2.5, 97.5),
                   aleatoric_in_std_perc=True, aleatoric_in_MC=False):
        """
        Predict y for new input X, along with uncertainty.

        :param X: input data shape (n_data, nx)
        :param ns: number of samples for MC approximation, int
        :param return_std: bool, indicates whether to return std. dev.
        :param return_MC: int, nb of MC samples to be returned
        :param return_percentiles: tuple, if (, ) percentiles are not returned
        :param aleatoric_in_std_perc: bool, if True std and perc outputs account for aleatoric uncertainties
        :param aleatoric_in_MC: bool, if True std and perc outputs account for aleatoric uncertainties

        :return y_mean: mean prediction, ndarray of shape (n_data, ny)
        :return y_std: uq prediction (std. dev.), ndarray of shape (n_data, ny)
        :return y_perc: uq prediction (percentiles), ndarray of shape (n_perc, n_data, ny)
        :return y_MC: samples from posterior, ndarray of shape (return_MC, n_data, ny)
        """
        # Initialize values to be fed to graph (feed_dict), i.e., fitted variational parameters
        feed_dict = dict(zip(self.tf_variational_mu, self.variational_mu))
        feed_dict.update(dict(zip(self.tf_variational_sigma, self.variational_sigma)))
        feed_dict.update({self.X_: X})
        # Set the random seed
        random_seed = None
        if self.random_seed is not None:
            random_seed = self.generate_seed_layers()
        # Run session: sample ns outputs
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            network_weights = sess.run(
                self.sample_weights_from_variational(ns=ns, random_seed=random_seed), feed_dict=feed_dict)
            y_MC = sess.run(self.compute_predictions(X=self.X_, network_weights=network_weights), feed_dict=feed_dict)
        # Compute statistical outputs from MC values
        outputs = compute_and_return_outputs(
            y_MC=y_MC, var_aleatoric=self.var_n, return_std=return_std, return_percentiles=return_percentiles,
            return_MC=return_MC, aleatoric_in_std_perc=aleatoric_in_std_perc, aleatoric_in_MC=aleatoric_in_MC)
        return outputs

    def predict_from_posterior_means(self, X, mask_low_dim=None):
        """
        Predict output in a deterministic fashion, using the mean of the posterior weights learnt. Can also predict
        output of pruned network, where only weights in mask_low_dim are kept

        :param X: input for prediction, ndarray of shape (n_pred, nx)
        :param mask_low_dim: indicates weights to keep in network, list (2 * n_uq_layers) of bool ndarrays
        """
        if mask_low_dim is not None:
            means_pruned = []
            for i, (m, means) in enumerate(zip(mask_low_dim, self.variational_mu)):
                means_pruned.append(np.where(m, means, np.zeros_like(means)))
        else:
            means_pruned = self.variational_mu.copy()
        means_pruned = [np.expand_dims(means, axis=0) for means in means_pruned]
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            y_from_mean = sess.run(self.compute_predictions(X=X, network_weights=means_pruned))[0]
        return y_from_mean

    def rank_network_weights(self, rank_metric='information gain', return_mask=False, threshold_on_number=None,
                             threshold_on_metric_perc=None, threshold_on_metric=None, keep_last_bias=False):
        """
        Rank weights in the network according to results of the VI approximation and some metrics (SNR of info gain)
        :param rank_metric: 'snr' or 'info gain'
        :param return_mask: bool, whether to return the mask of most 'important' weights
        :param threshold_on_number: max number of weights to keep
        :param threshold_on_metric_perc: keep weights so that a given percentage of the metric is achieved
        :param threshold_on_metric: keep weights for which metric > this threshold (try 0.83)

        :return: metric_values: metric value for all weights, list (length 2 * n_uq_layers) of ndarrays
        :return: importance_mask: indicates 'important' weights, list (length 2 * n_uq_layers) of boolean ndarrays
        """
        if rank_metric.lower() == 'snr':
            metric_values = [np.abs(mu) / std for (mu, std) in zip(self.variational_mu, self.variational_sigma)]
        elif rank_metric.lower() == 'information gain':
            if self.learn_prior:
                prior_means = self.variational_prior_mu
                prior_stds = self.variational_prior_sigma
            else:
                prior_means = self.prior_means
                prior_stds = self.prior_stds
            metric_values = [
                kl_div_gaussians(mean_prior=mu_prior, std_prior=std_prior, mean_posterior=mu, std_posterior=std)
                for (mu, std, mu_prior, std_prior)
                in (zip(self.variational_mu, self.variational_sigma, prior_means, prior_stds))]
        elif rank_metric.lower() == 'kl_divergence':
            metric_values = []
            for i, (mu, std, w_shape) in enumerate(zip(
                    self.variational_mu, self.variational_sigma, self.weights_shape)):
                # compute metric as kl divergence between weights and
                metric_val = kl_div_gaussians(
                    mean_prior=np.mean(mu), std_prior=np.sqrt(np.mean(std ** 2)), mean_posterior=mu, std_posterior=std)
                for _ in range(2):
                    # compute mask of outlier weights
                    ranking_bool = extract_mask_from_vi_ranking(
                        rank_metric=rank_metric, metric_values=metric_val.reshape((-1,)),
                        threshold_on_number=(None if threshold_on_number is None else threshold_on_number[i]),
                        threshold_on_metric_perc=(
                            None if threshold_on_metric_perc is None else threshold_on_metric_perc[i]),
                        threshold_on_metric=(None if threshold_on_metric is None else threshold_on_metric[i]))
                    # recompute metric values by removing the outlier gaussians from the mean gaussian
                    m = (1-ranking_bool.reshape(w_shape)).astype(bool)
                    metric_val = kl_div_gaussians(
                        mean_prior=np.mean(mu[m]), std_prior=np.sqrt(np.mean(std[m] ** 2)),
                        mean_posterior=mu, std_posterior=std)
                metric_values.append(metric_val)
        else:
            raise ValueError('rank_distance can be either "snr" or "information gain" or "kl_divergence".')
        if not return_mask:
            return metric_values

        # Also return the mask of importance weights
        importance_mask = []
        for i, (metric_val, w_shape) in enumerate(zip(metric_values, self.weights_shape)):
            ranking_bool = extract_mask_from_vi_ranking(
                rank_metric=rank_metric, metric_values=metric_val.reshape((-1, )),
                threshold_on_number=(None if threshold_on_number is None else threshold_on_number[i]),
                threshold_on_metric_perc=(None if threshold_on_metric_perc is None else threshold_on_metric_perc[i]),
                threshold_on_metric=(None if threshold_on_metric is None else threshold_on_metric[i]))
            importance_mask.append(ranking_bool.reshape(w_shape))
        self.n_weights_after_pruning = sum([np.sum(m) for m in importance_mask])
        if keep_last_bias:
            importance_mask[-1] = np.ones_like(importance_mask[-1]).astype(bool)
        return metric_values, importance_mask

    def return_marginals(self):
        """ Return the mean and std in all dimensions """
        return self.variational_mu.copy(), self.variational_sigma.copy()

    def compute_predictive_density(self, X, y, ns=10000):
        """
        Compute log predictive density p(y|data) at new data points (X, y)

        :return:
        """
        from scipy.special import logsumexp
        feed_dict = {self.X_: X, self.y_: y, self.ns_: ns, self.w_: np.ones((X.shape[0], ))}
        if isinstance(self.variational_mu, list):
            feed_dict.update(dict(zip(self.tf_variational_mu, self.variational_mu)))
            feed_dict.update(dict(zip(self.tf_variational_sigma, self.variational_sigma)))
        else:
            feed_dict.update({self.tf_variational_mu: self.variational_mu})
            feed_dict.update({self.tf_variational_sigma: self.variational_sigma})
        if hasattr(self, 'variational_amat'):
            feed_dict.update({self.tf_variational_amat: self.variational_amat})
        if hasattr(self, 'variational_cholesky'):
            feed_dict.update({self.tf_chol: self.variational_cholesky})

        random_seed = None
        if self.random_seed is not None:
            random_seed = self.generate_seed_layers()
        with tf.Session(graph=self.graph) as sess:
            # log_weight = log(1/N) + logsumexp(log_likelihood_thetais), thetais sampled from posterior
            sess.run(tf.global_variables_initializer())
            sampled_weights = sess.run(
                self.sample_weights_from_variational(ns=ns, random_seed=random_seed), feed_dict=feed_dict)
            all_loglike = -1. * sess.run(
                self.neg_log_like(y_true=tf.tile(tf.expand_dims(y.astype(np.float32), 0), [ns, 1, 1]),
                                  y_pred=self.compute_predictions(X=self.X_, network_weights=sampled_weights),
                                  do_sum=True, axis_sum=-1, weights_data=None),
                feed_dict={self.X_: X, self.ns_: ns})
            value = np.log(1. / ns) + logsumexp(all_loglike, axis=0)
        return value

    def rearrange_weights(self, mask_low_dim):
        """ Rearrange weights so that important ones are all in the same place """
        for layer in range(self.n_uq_layers - 1):
            mask_kernel = mask_low_dim[2 * layer]
            for row, row_mask_kernel in enumerate(mask_kernel):
                c = 0
                ind_imp = np.nonzero(row_mask_kernel)[0]
                if len(ind_imp) > 0:   # there are important weights there
                    for i in ind_imp:
                        # modify current output
                        mask_low_dim[2 * layer][row][[c, i]] = mask_low_dim[2 * layer][row][[i, c]]
                        mask_low_dim[2 * layer + 1][[c, i]] = mask_low_dim[2 * layer + 1][[i, c]]
                        self.variational_mu[2 * layer][row][[c, i]] = self.variational_mu[2 * layer][row][[i, c]]
                        self.variational_mu[2 * layer + 1][[c, i]] = self.variational_mu[2 * layer + 1][[i, c]]
                        self.variational_sigma[2 * layer][row][[c, i]] = self.variational_sigma[2 * layer][row][[i, c]]
                        self.variational_sigma[2 * layer + 1][[c, i]] = self.variational_sigma[2 * layer + 1][[i, c]]
                        # modify next input to match
                        mask_low_dim[2 * layer + 2][[c, i]] = mask_low_dim[2 * layer + 2][[i, c]]
                        self.variational_mu[2 * layer + 2][[c, i]] = self.variational_mu[2 * layer + 2][[i, c]]
                        self.variational_sigma[2 * layer + 2][[c, i]] = self.variational_sigma[2 * layer + 2][[i, c]]
                        c += 1
        return mask_low_dim


class BayesByBackprop(VIRegressor):
    """
    BayesByBackprop algorithm, from 'Weight Uncertainty in Neural Networks', Blundell et al., 2015.

    **Inputs:**

    :param tf_optimizer: optimizer, defaults to Adam optimizer
    """

    def __init__(self, hidden_units, input_dim=1, output_dim=1, var_n=1e-6, activation=tf.nn.relu, prior_means=0.,
                 prior_stds=1., weights_to_track=None, tf_optimizer=tf.train.AdamOptimizer, analytical_grads=False,
                 random_seed=None):

        # Initial checks and computations for the network
        super().__init__(hidden_units=hidden_units, input_dim=input_dim, output_dim=output_dim, var_n=var_n,
                         activation=activation, prior_means=prior_means, prior_stds=prior_stds,
                         random_seed=random_seed, weights_to_track=weights_to_track)
        self.analytical_grads = analytical_grads
        if self.learn_prior and self.analytical_grads:
            raise ValueError

        # Update the graph to compute the cost function
        with self.graph.as_default():
            # Initialize necessary variables
            self._initialize_variables_in_graph()
            if self.learn_prior:
                self._initialize_prior_variables_in_graph()

            if not self.analytical_grads:
                # Sample weights w from variational distribution q_{theta} and compute log(q_{theta}(w))
                tf_network_weights, var_post_term = self.sample_weights_from_variational(
                    ns=self.ns_, evaluate_log_pdf=True, sum_over_ns=True)

                # Compute the cost from the prior term -log(p(w))
                prior_term = self.log_prior_pdf(network_weights=tf_network_weights, sum_over_ns=True)

                # KL divergence KL(q||p)
                kl_q_p = (var_post_term - prior_term) / tf.to_float(self.ns_)
            else:
                # Sample weights w from variational distribution q_{theta}
                tf_network_weights = self.sample_weights_from_variational(ns=self.ns_, evaluate_log_pdf=False)
                kl_q_p = 0.
                for i, (m, std, m0, std0) in enumerate(zip(
                        self.tf_variational_mu, self.tf_variational_sigma, self.prior_means, self.prior_stds)):
                    kl_q_p += 0.5 * tf.reduce_sum((std ** 2 + (m - m0) ** 2) / std0 ** 2 - 2 * tf.log(std))

            # Branch to generate predictions
            self.predictions = self.compute_predictions(X=self.X_, network_weights=tf_network_weights)

            # Compute contribution of likelihood to cost: -log(p(data|w))
            neg_likelihood_term = self.neg_log_like(
                y_true=tf.tile(tf.expand_dims(self.y_, 0), [self.ns_, 1, 1]),
                y_pred=self.predictions, weights_data=self.w_) / tf.to_float(self.ns_)

            # Cost is based on the KL-divergence minimization
            self.cost = neg_likelihood_term + kl_q_p

            # Set-up the training procedure
            self.lr_ = tf.placeholder(tf.float32, name='lr_', shape=())
            self.opt = tf_optimizer(learning_rate=self.lr_)
            var_list = [self.tf_variational_mu, self.tf_variational_rho]
            if self.learn_prior:
                var_list = var_list + [self.tf_prior_mu, self.tf_prior_rho]
            if not self.analytical_grads:
                grads_and_vars = self.opt.compute_gradients(self.cost, var_list)
            else:
                grads_and_vars = self.opt.compute_gradients(neg_likelihood_term, var_list)
                # update gradients with analytical parts
                for i, (m0, std0) in enumerate(zip(self.prior_means, self.prior_stds)):
                    # mean
                    dmi, mi = grads_and_vars[i]
                    grads_and_vars[i] = (dmi + (mi - m0) / std0 ** 2, mi)
                    # std
                    drhoi, rhoi = grads_and_vars[2 * self.n_uq_layers + i]
                    si = self._sigma(rhoi)
                    dsi = tf.exp(rhoi) / (1. + tf.exp(rhoi))
                    grads_and_vars[2 * self.n_uq_layers + i] = (drhoi + (si / std0 ** 2 - 1. / si) * dsi, rhoi)
            self.grad_step = self.opt.apply_gradients(grads_and_vars)


class alphaBB(VIRegressor):
    """
    alpha-BlackBox algorithm, from 'Black-Box α-Divergence Minimization', Hernández-Lobato, 2016

    **Inputs:**

    :param alpha: alpha value between 0 and 1
    """

    def __init__(self, alpha, hidden_units, input_dim=1, output_dim=1, var_n=1e-6, activation=tf.nn.relu,
                 prior_means=0., prior_stds=1., weights_to_track=None, tf_optimizer=tf.train.AdamOptimizer,
                 random_seed=None, analytical_grads=False):

        # Do the initial checks and computations for the network
        super().__init__(hidden_units=hidden_units, input_dim=input_dim, output_dim=output_dim, var_n=var_n,
                         activation=activation, prior_means=prior_means, prior_stds=prior_stds,
                         random_seed=random_seed, weights_to_track=weights_to_track)
        self.analytical_grads = analytical_grads
        if self.learn_prior:
            raise NotImplementedError

        # check alpha value
        self.alpha = alpha
        if self.alpha <= 0. or self.alpha >= 1.:
            raise ValueError('Input alpha must be between 0 and 1')

        # Compute sufficient statistics for the prior, will be used throughout
        self.lmda_1_prior, self.lmda_2_prior = [], []
        for prior_mean, prior_std in zip(self.prior_means, self.prior_stds):
            lmda_1, lmda_2 = self._compute_sufficient_stats_from_moments(mean=prior_mean, sigma=prior_std, module='np')
            self.lmda_1_prior.append(lmda_1)
            self.lmda_2_prior.append(lmda_2)

        # Update the graph to compute the cost
        with self.graph.as_default():
            # Initialize necessary variables
            self._initialize_variables_in_graph()

            # Branch to sample weights from variational distribution
            tf_network_weights = self.sample_weights_from_variational(ns=self.ns_, evaluate_log_pdf=False)

            # Branch to generate predictions
            self.predictions = self.compute_predictions(X=self.X_, network_weights=tf_network_weights)

            ndata = tf.shape(self.X_)[0]  # number of independent data points
            normalization_term = 0.  # contribution of normalization terms to cost: -log(Z(prior))-log(Z(q))
            log_factor = 0.  # log of the site approximations f: log(f(w))=s(w).T lambda_{f}

            # Cost is based on the alpha-divergence minimization
            for weights, w_shape, mu, sigma, lmda_10, lmda_20 in zip(
                    tf_network_weights, self.weights_shape, self.tf_variational_mu, self.tf_variational_sigma,
                    self.lmda_1_prior, self.lmda_2_prior):
                # compute normalization term: log(Z(prior)) / constant if you are not learnign the prior
                # log(Z)=log(sqrt(2*pi)*sig)+mu**2/(2sig**2)
                #normalization_term += np.prod(w_shape) * (
                #        tf.log(prior_std) + 0.5 * tf.square(tf.divide(prior_mean, prior_std)))

                # compute normalization term: -log(Z(q))
                normalization_term -= tf.reduce_sum(tf.log(sigma) + 0.5 * tf.square(tf.divide(mu, sigma)))

                # compute the factor parameters lambda_f, then the log of the site approximations
                # lmda_q,0 is (mu/sig**2, -1/(2sig**2)), also use lmda=(lmda_q-lmda_0)/ndata
                # then f=exp(lmda_1*w + lmda_2*w**2)
                lmda_1_q, lmda_2_q = self._compute_sufficient_stats_from_moments(mean=mu, sigma=sigma)
                lmda_1 = (lmda_1_q - lmda_10) / tf.to_float(ndata)
                lmda_2 = (lmda_2_q - lmda_20) / tf.to_float(ndata)
                #lmda_1, lmda_2 = self.compute_factor_params_gaussian(
                #    mu_prior=prior_mean, var_prior=prior_std ** 2, mu_post=mu, var_post=tf.square(sigma),
                #    ndata=tf.to_float(ndata))
                lw = len(w_shape)
                log_factor += tf.reduce_sum(
                    tf.add(tf.multiply(tf.tile(tf.expand_dims(lmda_1, 0), [self.ns_, ] + [1, ] * lw), weights),
                           tf.multiply(tf.tile(tf.expand_dims(lmda_2, 0), [self.ns_, ] + [1, ] * lw),
                                       tf.square(weights))),
                    axis=list(range(-lw, 0)))

            # tie factors: use same site approximation for all data points
            log_factor = tf.tile(tf.expand_dims(log_factor, 1), [1, ndata])
            # If bootstrapping: need to modify the weights
            log_factor = tf.multiply(self.w_, log_factor)
            assert len(log_factor.get_shape().as_list()) == 2
            # compute likelihood of all data points n separately
            loglike = -1 * self.neg_log_like(
                y_true=tf.tile(tf.expand_dims(self.y_, 0), [self.ns_, 1, 1]), y_pred=self.predictions, do_sum=False,
                weights_data=self.w_)
            assert len(loglike.get_shape().as_list()) == 2

            # compute log(E_{q}[(like_n(w)/f(w)) ** alpha]) for all data points n,
            # expectation is computed by averaging over the ns_ samples
            logexpectation = tf.add(
                tf.log(1. / tf.to_float(self.ns_)),
                tf.reduce_logsumexp(self.alpha * tf.subtract(loglike, log_factor), axis=0)
            )
            # cste_over_ns = tf.reduce_max(self.alpha * (loglike - log_factor), axis=0)
            # in_exp = self.alpha * (loglike - log_factor) - tf.tile(tf.expand_dims(cste_over_ns, 0), [self.ns_, 1])
            # logexpectation = cste_over_ns - tf.log(tf.to_float(self.ns_)) + tf.reduce_logsumexp(in_exp, axis=0)
            # in the cost, sum over all the data points n
            expectation_term = - 1. / self.alpha * tf.reduce_sum(logexpectation)
            self.cost = normalization_term + expectation_term

            # Set-up the training procedure
            self.lr_ = tf.placeholder(tf.float32, name='lr_', shape=())
            opt = tf_optimizer(learning_rate=self.lr_)
            var_list = [self.tf_variational_mu, self.tf_variational_rho]
            if not self.analytical_grads:
                grads_and_vars = opt.compute_gradients(self.cost, var_list)
            else:
                grads_and_vars = opt.compute_gradients(expectation_term, var_list)
                # update gradients with analytical parts
                for i in range(2 * self.n_uq_layers):
                    dmi, mi = grads_and_vars[i]   # mean and its derivative
                    drhoi, rhoi = grads_and_vars[2 * self.n_uq_layers + i]    # rho and its derivative
                    si = self._sigma(rhoi)    # std...
                    dsi = tf.exp(rhoi) / (1. + tf.exp(rhoi))    # ... and its derivative with respect to rhoi
                    # update gradient for mean
                    grads_and_vars[i] = (dmi - mi / si ** 2, mi)
                    # update gradient for rho
                    grads_and_vars[2 * self.n_uq_layers + i] = (drhoi + (1 / si * ((mi / si) ** 2 - 1)) * dsi, rhoi)
            grads_and_vars = [(tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad), val)
                              for grad, val in grads_and_vars]
            self.grad_step = opt.apply_gradients(grads_and_vars)

    @staticmethod
    def _compute_sufficient_stats_from_moments(mean, sigma, module='tf'):
        """ Compute sufficient statistics of a gaussian from its mean and variance """
        if module == 'tf':
            var = tf.square(sigma)
            lmda_1 = tf.divide(mean, var)
        elif module == 'np':
            var = sigma ** 2
            lmda_1 = mean / var
        else:
            raise ValueError
        lmda_2 = - 1. / (2. * var)
        return lmda_1, lmda_2

    @staticmethod
    def _compute_moments_from_sufficient_stats(lmda_1, lmda_2, module='tf'):
        """ Compute sufficient statistics of a gaussian from its mean and variance """
        var = -1. / (2. * lmda_2)
        mean = lmda_1 * var
        if module == 'np':
            return mean, np.sqrt(var)
        return mean, tf.sqrt(var)

    def return_site_approximations(self):
        """ Return the site approximations f_{i}(w) for all training data i, here all the same """
        ndata = self.training_data[0].shape[0]
        self.tie_factors = [list(range(ndata)), ]
        site_approxs_mu, site_approxs_sigma = [], []
        for mu, sigma, lmda1_0, lmda2_0, w_shape in zip(
                self.variational_mu, self.variational_sigma, self.lmda_1_prior, self.lmda_2_prior, self.weights_shape):
            lmda1_q, lmda2_q = self._compute_sufficient_stats_from_moments(mean=mu, sigma=sigma, module='np')
            lmda1_n = (lmda1_q - lmda1_0) / ndata
            lmda2_n = (lmda2_q - lmda2_0) / ndata
            mu_n, sigma_n = self._compute_moments_from_sufficient_stats(lmda_1=lmda1_n, lmda_2=lmda2_n, module='np')
            site_approxs_mu.append(mu_n)
            site_approxs_sigma.append(sigma_n)
        return [site_approxs_mu, ], [site_approxs_sigma, ]

    def get_lso_moments(self, leave_factors):
        # Compute the mean and std of the leave-several-out density
        n_out = len(leave_factors)
        ndata = self.training_data[0].shape[0]
        lso_mu, lso_sigma = [], []
        for mu, sigma, lmda1_0, lmda2_0, w_shape in zip(
                self.variational_mu, self.variational_sigma, self.lmda_1_prior, self.lmda_2_prior, self.weights_shape):
            lmda1_q, lmda2_q = self._compute_sufficient_stats_from_moments(mean=mu, sigma=sigma, module='np')
            lmda1_lso = (lmda1_q - lmda1_0) * (ndata - n_out) / ndata + lmda1_0
            lmda2_lso = (lmda2_q - lmda2_0) * (ndata - n_out) / ndata + lmda2_0
            mu_n, sigma_n = self._compute_moments_from_sufficient_stats(lmda_1=lmda1_lso, lmda_2=lmda2_lso, module='np')
            lso_mu.append(mu_n)
            lso_sigma.append(sigma_n)
        return lso_mu, lso_sigma

    def predict_uq_from_lso(self, X, leave_factors, ns, return_std=True, return_percentiles=(2.5, 97.5),
                            aleatoric_in_std_perc=True, ):
        """ Predict from leave-several-out density. For this case which index it is does not matter """
        lso_mu, lso_sigma = self.get_lso_moments(leave_factors=leave_factors)
        # Run the graph to predict output and associated uncertainty
        feed_dict = dict(zip(self.tf_variational_mu, lso_mu))
        feed_dict.update(dict(zip(self.tf_variational_sigma, lso_sigma)))
        feed_dict.update({self.X_: X})
        # Set the random seed
        random_seed = None
        if self.random_seed is not None:
            random_seed = self.generate_seed_layers()
        # Run session: sample ns outputs
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            network_weights = sess.run(
                self.sample_weights_from_variational(ns=ns, random_seed=random_seed), feed_dict=feed_dict)
            y_MC = sess.run(self.compute_predictions(X=self.X_, network_weights=network_weights), feed_dict=feed_dict)
        # Compute statistical outputs from MC values
        outputs = compute_and_return_outputs(
            y_MC=y_MC, var_aleatoric=self.var_n, return_std=return_std, return_percentiles=return_percentiles,
            return_MC=0, aleatoric_in_std_perc=aleatoric_in_std_perc, aleatoric_in_MC=False)
        return outputs

    def compute_lso_predictive_density(self, leave_factors, ns=10000):
        """
        Compute log predictive density p(y|data) at new data points (X, y)

        :return:
        """
        from scipy.special import logsumexp
        # get data for prediction (left out during training)
        X_pred = np.array([self.training_data[0][i] for i in leave_factors])
        y_pred = np.array([self.training_data[1][i] for i in leave_factors])
        # get lso density
        lso_mu, lso_sigma = self.get_lso_moments(leave_factors=leave_factors)
        # Run the graph to predict output and associated uncertainty
        feed_dict = dict(zip(self.tf_variational_mu, lso_mu))
        feed_dict.update(dict(zip(self.tf_variational_sigma, lso_sigma)))
        # Set the random seed
        random_seed = None
        if self.random_seed is not None:
            random_seed = self.generate_seed_layers()
        # Run session: sample ns outputs
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            sampled_weights = sess.run(
                self.sample_weights_from_variational(ns=ns, random_seed=random_seed), feed_dict=feed_dict)
            # log_weight = log(1/N) + logsumexp(log_likelihood_thetais), thetais sampled from posterior
            all_loglike = -1. * sess.run(
                self.neg_log_like(y_true=tf.tile(tf.expand_dims(y_pred.astype(np.float32), 0), [ns, 1, 1]),
                                  y_pred=self.compute_predictions(X=self.X_, network_weights=sampled_weights),
                                  do_sum=True, axis_sum=-1, weights_data=None),
                feed_dict={self.X_: X_pred, self.ns_: ns})
            value = np.log(1. / ns) + logsumexp(all_loglike, axis=0)
        return value

    def compute_all_loo_predictive_densities(self, ns=10000):
        """
        Compute log predictive density p(y|data) at new data points (X, y)

        :return:
        """
        from scipy.special import logsumexp
        # get data for prediction (left out during training)
        X_pred, y_pred = self.training_data
        # get lso density
        lso_mu, lso_sigma = self.get_lso_moments(leave_factors=[0, ])
        # Run the graph to predict output and associated uncertainty
        feed_dict = dict(zip(self.tf_variational_mu, lso_mu))
        feed_dict.update(dict(zip(self.tf_variational_sigma, lso_sigma)))
        # Set the random seed
        random_seed = None
        if self.random_seed is not None:
            random_seed = self.generate_seed_layers()
        # Run session: sample ns outputs
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            sampled_weights = sess.run(
                self.sample_weights_from_variational(ns=ns, random_seed=random_seed), feed_dict=feed_dict)
            # log_weight = log(1/N) + logsumexp(log_likelihood_thetais), thetais sampled from posterior
            all_loglike = -1. * sess.run(
                self.neg_log_like(y_true=tf.tile(tf.expand_dims(y_pred.astype(np.float32), 0), [ns, 1, 1]),
                                  y_pred=self.compute_predictions(X=self.X_, network_weights=sampled_weights),
                                  do_sum=False, weights_data=None),
                feed_dict={self.X_: X_pred, self.ns_: ns})
            values = np.log(1. / ns) + logsumexp(all_loglike, axis=0)
        return values


########################################################################################################################
#                                          Algorithms with correlations                                                #
########################################################################################################################

class BayesByBackpropLowRank(VIRegressor):
    """
    BayesByBackprop algorithm, from 'Weight Uncertainty in Neural Networks', Blundell et al., 2015.
    Create correlation between dimensions via a low-rank covariance.

    **Inputs:**

    :param tf_optimizer: optimizer, defaults to Adam optimizer
    """

    def __init__(self, hidden_units, input_dim=1, output_dim=1, var_n=1e-6, activation=tf.nn.relu, prior_means=0.,
                 prior_stds=1., weights_to_track=None, tf_optimizer=tf.train.AdamOptimizer,
                 rank=1, analytical_grads=False, keep_diag=True):

        # Initial checks and computations for the network
        super().__init__(hidden_units=hidden_units, input_dim=input_dim, output_dim=output_dim, var_n=var_n,
                         activation=activation, prior_means=prior_means, prior_stds=prior_stds,
                         weights_to_track=weights_to_track)
        self.variational_amat = None
        if self.learn_prior:
            raise NotImplementedError
        self.rank = rank
        self.analytical_grads = analytical_grads
        if self.weights_to_track is not None:
            self.variational_amat_history = [np.array([]).reshape((0, nw, self.rank)) for nw in self.weights_to_track]

        # Update the graph to compute the cost function
        with self.graph.as_default():
            # Initialize necessary variables
            self._initialize_variables_in_graph()

            if not self.analytical_grads:
                # Sample weights w from variational distribution q_{theta} and compute log(q_{theta}(w))
                tf_network_weights, var_post_term = self.sample_weights_from_variational(
                    ns=self.ns_, evaluate_log_pdf=True, sum_over_ns=True)

                # Compute the cost from the prior term -log(p(w))
                prior_term = self.log_prior_pdf(network_weights=tf_network_weights, sum_over_ns=True)

                # KL divergence KL(q||p)
                kl_q_p = (var_post_term - prior_term) / tf.to_float(self.ns_)
            else:
                # Sample weights w from variational distribution q_{theta}
                tf_network_weights = self.sample_weights_from_variational(ns=self.ns_, evaluate_log_pdf=False)
                chol = tf.cholesky(
                    tf.eye(self.rank) + tf.matmul(
                        tf.transpose(self.tf_variational_amat) / self.tf_variational_sigma ** 2,
                        self.tf_variational_amat)
                )
                log_det_term = -0.5 * 2 * tf.reduce_sum(tf.log(tf.diag_part(chol)))
                m0, var0 = [], []
                for m, std, w_dim in zip(self.prior_means, self.prior_stds, self.weights_dim):
                    m0 += [m, ] * w_dim
                    var0 += [std ** 2, ] * w_dim
                m0 = tf.constant(m0)
                var0 = tf.constant(var0)
                kl_q_p = 0.5 * tf.reduce_sum(
                    (self.tf_variational_sigma ** 2 + (self.tf_variational_mu - m0) ** 2 +
                     tf.reduce_sum(self.tf_variational_amat ** 2, axis=-1)) / var0
                    - 2 * tf.log(self.tf_variational_sigma))
                kl_q_p += log_det_term

            # Branch to generate predictions
            self.predictions = self.compute_predictions(X=self.X_, network_weights=tf_network_weights)

            # Compute contribution of likelihood to cost: -log(p(data|w))
            neg_likelihood_term = self.neg_log_like(
                y_true=tf.tile(tf.expand_dims(self.y_, 0), [self.ns_, 1, 1]),
                y_pred=self.predictions, weights_data=self.w_) / tf.to_float(self.ns_)

            # Cost is based on the KL-divergence minimization
            self.cost = neg_likelihood_term + kl_q_p

            # Set-up the training procedure
            self.lr_ = tf.placeholder(tf.float32, name='lr_', shape=())
            self.flag_epoch = tf.placeholder(tf.int32, shape=())
            self.opt = tf_optimizer(learning_rate=self.lr_)
            if keep_diag:
                var_list = [self.tf_variational_mu, self.tf_variational_rho, self.tf_variational_amat]
            else:
                var_list = [self.tf_variational_mu, self.tf_variational_amat]
            if not self.analytical_grads:
                grads_and_vars = self.opt.compute_gradients(self.cost, var_list)
            else:
                grads_and_vars = self.opt.compute_gradients(neg_likelihood_term + log_det_term, var_list)
                # update gradients with analytical parts: mean
                dmi, mi = grads_and_vars[0]
                grads_and_vars[0] = (dmi + (mi - m0) / var0, mi)
                if keep_diag:
                    # diagonal rho
                    drhoi, rhoi = grads_and_vars[1]
                    si = self._sigma(rhoi)
                    dsi = tf.exp(rhoi) / (1. + tf.exp(rhoi))
                    grads_and_vars[1] = (drhoi + (si / var0 - 1. / si) * dsi, rhoi)
                    # amat
                    damat, amat = grads_and_vars[2]
                    grads_and_vars[2] = (damat + tf.transpose(tf.transpose(amat) / var0), amat)
                else:
                    # amat
                    damat, amat = grads_and_vars[1]
                    grads_and_vars[1] = (damat + tf.transpose(tf.transpose(amat) / var0), amat)
            self.grad_step = self.opt.apply_gradients(grads_and_vars)

    def _initialize_variables_in_graph(self):
        """
        Initialize some variables in VI graph
        """
        self.tf_tracked_means = []
        self.tf_tracked_stds = []
        self.tf_tracked_amat = []

        # add dense layers, add contributions of each layer to prior and variational posterior costs
        standard_sigmas = compute_standard_sigmas(hidden_units=self.hidden_units, input_dim=self.input_dim,
                                                  output_dim=self.output_dim, scale=1., mode='fan_avg')
        standard_sigmas_aug = []
        [standard_sigmas_aug.extend([std, 0.01]) for std in standard_sigmas]
        start_sigmas = []
        for std, w_dim in zip(standard_sigmas_aug, self.weights_dim):
            start_sigmas += [std, ] * w_dim
        self.tf_variational_mu = tf.Variable(tf.random_normal(shape=(self.n_weights, ), mean=0., stddev=start_sigmas),
                                             trainable=True, dtype=tf.float32)
        self.tf_variational_rho = tf.Variable(-6.9 * tf.ones(shape=(self.n_weights, ), dtype=tf.float32),
                                              trainable=True, dtype=tf.float32)
        self.tf_variational_sigma = self._sigma(self.tf_variational_rho)
        # self.tf_variational_amat = tf.Variable(tf.random_normal(shape=(self.n_weights, self.rank), mean=0.,
        #                                                         stddev=0.001 / self.rank),
        #                                        trainable=True, dtype=tf.float32)
        self.tf_variational_amat = tf.Variable(tf.zeros(shape=(self.n_weights, self.rank), dtype=tf.float32),
                                               trainable=True, dtype=tf.float32)

        # Keep track of some of the weights
        if self.weights_to_track is not None:
            c = 0
            for i, w_dim in enumerate(self.weights_dim):
                self.tf_tracked_means.append(self.tf_variational_mu[c:c + self.weights_to_track[i]])
                self.tf_tracked_stds.append(self.tf_variational_sigma[c:c + self.weights_to_track[i]])
                self.tf_tracked_amat.append(self.tf_variational_amat[c:c + self.weights_to_track[i], :])
                c += w_dim

    def sample_weights_from_variational(self, ns, random_seed=None, evaluate_log_pdf=False, sum_over_ns=False):
        """
        Samples weights w for the NN from the variational density q_{theta}(w) (a mixture of gaussian)

        :return: weights w, as a list of length 2 * n_uq_layers
        """
        log_q = 0.
        mu = tf.tile(tf.expand_dims(self.tf_variational_mu, axis=0), [ns, 1])
        sigma = tf.tile(tf.expand_dims(self.tf_variational_sigma, axis=0), [ns, 1])
        w = tf.add(
            tf.multiply(sigma, tf.random_normal(shape=(ns, self.n_weights), mean=0., stddev=1.)),
            tf.matmul(tf.random_normal(shape=(ns, self.rank), mean=0., stddev=1.),
                      tf.transpose(self.tf_variational_amat)))
        w = tf.add(w, mu)
        weights = []
        tot_dim = 0
        for w_dim, w_shape in zip(self.weights_dim, self.weights_shape):
            weights.append(tf.reshape(w[:, tot_dim:tot_dim + w_dim], (ns, ) + w_shape))
            tot_dim += w_dim

        if evaluate_log_pdf:
            chol = tf.cholesky(tf.diag(self.tf_variational_sigma ** 2) + tf.matmul(
                self.tf_variational_amat, tf.transpose(self.tf_variational_amat)))
            log_q += log_multiv_gaussian(x=w, mean=self.tf_variational_mu, cov_chol=chol,
                                         axis_sum=(None if sum_over_ns else []))
            return weights, log_q
        return weights

    def fit(self, X, y, weights_data=None, ns=10, epochs=100, verbose=0, lr=0.001):
        """
        Fit, i.e., find the variational distribution that minimizes the cost function

        :param X: input data, ndarray of shape (n_data, nx)
        :param y: output data, ndarray of shape (n_data, ny)
        :param weights_data:
        :param epochs: int
        :param ns: nb. of samples used in computing cost (expectation over variational distribution)
        :param lr: learning rate for optimizer
        :param verbose:
        """
        # Initilize tensorflow session and required variables
        self.training_data = (X, y)
        if weights_data is None:
            weights_data = np.ones((X.shape[0], ))

        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            # Run training loop
            for e in range(epochs):
                # Loop over all minibatches
                _, loss_history_ = sess.run(
                    [self.grad_step, self.cost],
                    feed_dict={self.w_: weights_data, self.X_: X, self.y_: y, self.ns_: ns, self.lr_: lr})
                self.loss_history.append(loss_history_)
                # Save some of the weights
                if self.weights_to_track is not None:
                    mean, std, amat = sess.run(
                        [self.tf_tracked_means, self.tf_tracked_stds, self.tf_tracked_amat], feed_dict={self.ns_: ns})
                    self.variational_mu_history = [np.concatenate([m1, m2.reshape((1, -1))], axis=0)
                                                   for m1, m2 in zip(self.variational_mu_history, mean)]
                    self.variational_sigma_history = [np.concatenate([m1, m2.reshape((1, -1))], axis=0)
                                                      for m1, m2 in zip(self.variational_sigma_history, std)]
                    self.variational_amat_history = [np.concatenate([m1, m2.reshape((1, -1, self.rank))], axis=0)
                                                     for m1, m2 in zip(self.variational_amat_history, amat)]
                # print comments on terminal
                if verbose:
                    print('epoch = {}, loss = {}'.format(e, self.loss_history[-1]))

            # Save the final variational parameters
            self.variational_mu, self.variational_sigma, self.variational_amat = sess.run(
                [self.tf_variational_mu, self.tf_variational_sigma, self.tf_variational_amat], feed_dict={self.ns_: ns})

    def predict_uq(self, X, ns, return_std=True, return_MC=10, return_percentiles=(2.5, 97.5),
                   aleatoric_in_std_perc=True, aleatoric_in_MC=False):
        """
        Predict y for new input X, along with uncertainty.
        """
        feed_dict = {self.tf_variational_mu: self.variational_mu}
        feed_dict.update({self.tf_variational_sigma: self.variational_sigma})
        feed_dict.update({self.tf_variational_amat: self.variational_amat})
        feed_dict.update({self.X_: X, self.ns_: ns})
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            y_MC = sess.run(self.predictions, feed_dict=feed_dict)
        outputs = compute_and_return_outputs(
            y_MC=y_MC, var_aleatoric=self.var_n, return_std=return_std, return_percentiles=return_percentiles,
            return_MC=return_MC, aleatoric_in_std_perc=aleatoric_in_std_perc, aleatoric_in_MC=aleatoric_in_MC)
        return outputs

    def return_marginals(self):
        """ Return the mean and std in all dimensions """
        mean, std = [], []
        c = 0
        var = self.variational_sigma ** 2 + np.diag(np.matmul(self.variational_amat, self.variational_amat.T))
        for w_dim, w_shape in zip(self.weights_dim, self.weights_shape):
            mean.append(self.variational_mu[c:c + w_dim].reshape(w_shape))
            std.append(np.sqrt(var[c:c + w_dim].reshape(w_shape)))
            c += w_dim
        return mean, std


class BayesByBackpropWithCorr(VIRegressor):
    """
    BayesByBackprop algorithm, allows for correlations between RVs

    **Inputs:**

    :param tf_optimizer: optimizer, defaults to Adam optimizer
    """

    def __init__(self, hidden_units, input_dim=1, output_dim=1, var_n=1e-6, activation=tf.nn.relu, prior_means=0.,
                 prior_stds=1., weights_to_track=None, tf_optimizer=tf.train.AdamOptimizer):

        # Initial checks and computations for the network
        super().__init__(hidden_units=hidden_units, input_dim=input_dim, output_dim=output_dim, var_n=var_n,
                         activation=activation, prior_means=prior_means, prior_stds=prior_stds,
                         weights_to_track=weights_to_track)
        if self.learn_prior:
            raise NotImplementedError
        self.mask_correlation = None  # No correlation at the beginning

        # Update the graph to compute the cost function
        with self.graph.as_default():
            # Initialize necessary variables
            self._initialize_variables_in_graph()
            if self.learn_prior:
                self._initialize_prior_variables_in_graph()

            # Branch to sample weights from variational distribution
            # Compute the cost from the variational distribution term log(q_{theta}(w))
            tf_network_weights, var_post_term = self.sample_weights_from_variational(
                ns=self.ns_, evaluate_log_pdf=True, sum_over_ns=True)

            # Branch to generate predictions
            self.predictions = self.compute_predictions(X=self.X_, network_weights=tf_network_weights)

            # Compute contribution of likelihood to cost: -log(p(data|w))
            neg_likelihood_term = self.neg_log_like(
                y_true=tf.tile(tf.expand_dims(self.y_, 0), [self.ns_, 1, 1]), y_pred=self.predictions,
                weights_data=self.w_)

            # Compute the cost from the prior term -log(p(w))
            prior_term = self.log_prior_pdf(network_weights=tf_network_weights, sum_over_ns=True)

            # Cost is based on the KL-divergence minimization
            self.cost = (neg_likelihood_term - prior_term + var_post_term) / tf.to_float(self.ns_)

            # Set-up the training procedure
            self.lr_ = tf.placeholder(tf.float32, name='lr_', shape=())
            self.opt = tf_optimizer(learning_rate=self.lr_)
            var_list = [self.tf_variational_mu, self.tf_variational_rho]
            total_grads_and_vars = self.opt.compute_gradients(self.cost, var_list)
            self.grad_step = self.opt.apply_gradients(total_grads_and_vars)

        # Initialize the histories and other outputs
        self.loss_history = []
        if self.weights_to_track is not None:
            self.variational_mu_history = [np.array([]).reshape((0, nw)) for nw in self.weights_to_track]
            self.variational_sigma_history = [np.array([]).reshape((0, nw)) for nw in self.weights_to_track]
        self.variational_mu, self.variational_sigma = None, None

    def add_correlation(self, mask_correlation):
        """
        Add variables for gaussian correlation
        """
        self.mask_correlation = mask_correlation
        d = int(np.sum([np.sum(m, axis=None) for m in self.mask_correlation]))
        self.variational_small_sigma_history = []
        positions = []
        for i, m in enumerate(self.mask_correlation):
            m_ = np.argwhere(np.reshape(m, (-1,))).reshape((-1,))
            positions.extend([(i, mi) for mi in m_])
        self.positions = positions
        with self.graph.as_default():
            tf_correlations_vars = tf.Variable(tf.zeros((int(d * (d - 1) / 2),)), dtype=tf.float32, trainable=True)
            self.tf_small_sigma = tf.concat([tf.boolean_mask(w, m) for (w, m)
                                             in zip(self.tf_variational_sigma, self.mask_correlation)], axis=0)
            chol_mat = [tf.constant([1., ] + [0.] * (d - 1)), ]
            cnt = 0
            for i in range(1, d):
                chol_mat.append(tf.concat(
                    [tf_correlations_vars[cnt:cnt + i], tf.constant([1., ] + [0.] * (d - i - 1))],
                    axis=0))
                cnt += i
            self.tf_chol = tf.matmul(tf.stack(chol_mat, axis=0), tf.diag(self.tf_small_sigma))

            # Update loss
            #self.tf_network_weights = self.sample_weights_from_variational_v2()
            #self.var_post_term = self.log_variational_pdf_v2(sum_over_ns=True) / tf.to_float(self.ns_)
            tf_network_weights, var_post_term = self.sample_weights_from_variational_v2(
                ns=self.ns_, evaluate_log_pdf=True, sum_over_ns=True)
            self.predictions = self.compute_predictions(X=self.X_, network_weights=tf_network_weights)
            neg_likelihood_term = self.neg_log_like(
                y_true=tf.tile(tf.expand_dims(self.y_, 0), [self.ns_, 1, 1]), y_pred=self.predictions,
                weights_data=self.w_)
            prior_term = self.log_prior_pdf(network_weights=tf_network_weights, sum_over_ns=True)
            self.cost = (neg_likelihood_term - prior_term + var_post_term) / tf.to_float(self.ns_)

            # Update training procedure
            var_list = [self.tf_variational_mu, self.tf_variational_rho, tf_correlations_vars]
            total_grads_and_vars = self.opt.compute_gradients(self.cost, var_list)
            self.grad_step = self.opt.apply_gradients(total_grads_and_vars)

            # Update the histories to store
            m_stored = [m.reshape((-1,))[:n] for m, n in zip(self.mask_correlation, self.weights_to_track)]
            if sum([sum(m_n) for m_n in m_stored]) > 0:
                sigmas = tf.diag_part(tf.matmul(self.tf_chol, tf.transpose(self.tf_chol)))
                self.tf_tracked_stds = [tf.unstack(sig) for sig in self.tf_tracked_stds]
                for i, pos in enumerate(self.positions):
                    if pos[1] < self.weights_to_track[pos[0]]:
                        self.tf_tracked_stds[pos[0]][pos[1]] = sigmas[i]
                self.tf_tracked_stds = [tf.stack(sig) for sig in self.tf_tracked_stds]

    def sample_weights_from_variational_v2(self, ns, evaluate_log_pdf=False, sum_over_ns=False):
        """
        Samples weights w for the NN from the variational density q_{theta}(w) (gaussian), includes correlations

        :return: weights w, as a list of length 2 * n_uq_layers
        """
        log_q = 0.
        weights = []
        for vi_mu, vi_sigma, w_shape, m in zip(
                self.tf_variational_mu, self.tf_variational_sigma, self.weights_shape, self.mask_correlation):
            mu = tf.tile(tf.expand_dims(vi_mu, axis=0), [ns, ] + [1, ] * len(w_shape))
            sigma = tf.tile(tf.expand_dims(vi_sigma, axis=0), [ns, ] + [1, ] * len(w_shape))
            w = tf.add(
                mu, tf.multiply(sigma, tf.random_normal(shape=(ns,) + w_shape, mean=0., stddev=1.)))
            weights.append(w)

            if evaluate_log_pdf:
                m_neg = (1-m).astype(bool)
                log_q += log_gaussian(
                    x=w, mean=mu, std=sigma, axis_sum=(None if sum_over_ns else list(range(1, 1 + len(w_shape)))),
                    mask_to_keep=m_neg)

        d = np.sum([np.sum(m) for m in self.mask_correlation])
        weights = [tf.unstack(tf.reshape(w, (ns, int(np.prod(w_shape)))), axis=-1)
                   for w, w_shape in zip(weights, self.weights_shape)]
        mu_vector = tf.concat([tf.boolean_mask(w, m) for (w, m)
                               in zip(self.tf_variational_mu, self.mask_correlation)], axis=0)
        extra_rvs = tf.tile(tf.reshape(mu_vector, (1, d)), [ns, 1]) + tf.matmul(
            tf.random_normal(shape=(ns, d), mean=0., stddev=1.), tf.transpose(self.tf_chol))
        for i, pos in enumerate(self.positions):
            weights[pos[0]][pos[1]] = tf.reshape(extra_rvs[:, i], (ns, ))
        weights = [tf.reshape(tf.stack(w, axis=-1), (ns, ) + w_shape)
                   for w, w_shape in zip(weights, self.weights_shape)]
        if evaluate_log_pdf:
            log_q += log_multiv_gaussian(
                x=extra_rvs, mean=tf.tile(tf.reshape(mu_vector, (1, d)), [ns, 1]), cov_chol=self.tf_chol,
                axis_sum=(None if sum_over_ns else -1))
            return weights, log_q
        return weights

    def fit(self, X, y, weights_data=None, ns=10, epochs=100, verbose=0, lr=0.001):
        """
        Fit, i.e., find the variational distribution that minimizes the cost function

        :param X: input data, ndarray of shape (n_data, nx)
        :param y: output data, ndarray of shape (n_data, ny)
        :param weights_data
        :param epochs: int
        :param ns: nb. of samples used in computing cost (expectation over variational distribution)
        :param lr: learning rate for optimizer
        :param verbose
        """
        # Initilize tensorflow session (the same will be used later for ) and required variables
        self.training_data = (X, y)
        if weights_data is None:
            weights_data = np.ones((X.shape[0], ))

        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            # If you are re-staring training, start from the previously saved values
            if getattr(self, 'variational_mu', None) is not None:
                sess.run([tf.assign(tf_, val_) for tf_, val_ in zip(self.tf_variational_mu, self.variational_mu)])
                sess.run([tf.assign(tf_, np.log(np.exp(val_) - 1.)) for tf_, val_
                          in zip(self.tf_variational_rho, self.variational_sigma)])

            # Run training loop
            for e in range(epochs):
                # Loop over all minibatches
                _, loss_history_ = sess.run(
                    [self.grad_step, self.cost],
                    feed_dict={self.w_: weights_data, self.X_: X, self.y_: y, self.ns_: ns, self.lr_: lr})
                self.loss_history.append(loss_history_)
                # Save some of the weights
                if self.weights_to_track is not None:
                    mean, std = sess.run([self.tf_tracked_means, self.tf_tracked_stds], feed_dict={self.ns_: ns})
                    self.variational_mu_history = [np.concatenate([m1, m2.reshape((1, -1))], axis=0)
                                                   for m1, m2 in zip(self.variational_mu_history, mean)]
                    self.variational_sigma_history = [np.concatenate([m1, m2.reshape((1, -1))], axis=0)
                                                      for m1, m2 in zip(self.variational_sigma_history, std)]
                # print comments on terminal
                if verbose:
                    print('epoch = {}, loss = {}'.format(e, self.loss_history[-1]))

            # Save the final variational parameters
            self.variational_mu, self.variational_sigma = sess.run(
                [self.tf_variational_mu, self.tf_variational_sigma], feed_dict={self.ns_: ns})
            if self.mask_correlation is not None:
                self.variational_cholesky = sess.run(self.tf_chol, feed_dict={self.ns_: ns})

    def predict_uq(self, X, ns, return_std=True, return_MC=2, return_percentiles=(2.5, 97.5),
                   aleatoric_in_std_perc=True, aleatoric_in_MC=False):
        """
        Predict y for new input X, along with uncertainty.

        :param X: input data shape (n_data, nx)
        :param ns:
        :param return_std: bool, indicates whether to return std. dev.
        :param return_MC: int, nb of MC samples to be returned
        :param return_percentiles: tuple, if (, ) percentiles are not returned
        :param aleatoric_in_std_perc: bool, if True std and perc outputs account for aleatoric uncertainties
        :param aleatoric_in_MC: bool, if True std and perc outputs account for aleatoric uncertainties

        :return y_mean: mean prediction, ndarray of shape (n_data, ny)
        :return y_std: uq prediction (std. dev.), ndarray of shape (n_data, ny)
        :return y_perc: uq prediction (percentiles), ndarray of shape (n_perc, n_data, ny)
        :return y_MC: samples from posterior, ndarray of shape (return_MC, n_data, ny)
        """
        feed_dict = dict(zip(self.tf_variational_mu, self.variational_mu))
        feed_dict.update(dict(zip(self.tf_variational_sigma, self.variational_sigma)))
        if self.mask_correlation is not None:
            feed_dict.update({self.tf_chol: self.variational_cholesky})
        feed_dict.update({self.X_: X, self.ns_: ns})
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            y_MC = sess.run(self.predictions, feed_dict=feed_dict)
        outputs = compute_and_return_outputs(
            y_MC=y_MC, var_aleatoric=self.var_n, return_std=return_std, return_percentiles=return_percentiles,
            return_MC=return_MC, aleatoric_in_std_perc=aleatoric_in_std_perc, aleatoric_in_MC=aleatoric_in_MC)
        return outputs

    def return_marginals(self):
        """ Return the mean and std in all dimensions """
        mean, std = self.variational_mu.copy(), self.variational_sigma.copy()
        if self.mask_correlation is not None:
            # for the ones that have been modified by the cholesky
            std_tmp = [std_i.reshape((-1,)) for std_i in std]
            cov = np.sum(self.variational_cholesky ** 2, axis=1)
            for i, pos in enumerate(self.positions):
                std_tmp[pos[0]][pos[1]] = np.sqrt(cov[i])
            std = [std_i.reshape(w_shape) for w_shape, std_i in zip(self.weights_shape, std_tmp)]
        return mean, std


########################################################################################################################
#                                              alphaBB with correlations                                               #
########################################################################################################################

class alphaBBLowRank(VIRegressor):
    """
    BayesByBackprop algorithm, from 'Weight Uncertainty in Neural Networks', Blundell et al., 2015.
    Create correlation between dimensions via a low-rank covariance.

    **Inputs:**

    :param tf_optimizer: optimizer, defaults to Adam optimizer
    """

    def __init__(self, alpha, hidden_units, input_dim=1, output_dim=1, var_n=1e-6, activation=tf.nn.relu,
                 prior_means=0., prior_stds=1., weights_to_track=None, tf_optimizer=tf.train.AdamOptimizer,
                 rank=1):

        # Initial checks and computations for the network
        super().__init__(hidden_units=hidden_units, input_dim=input_dim, output_dim=output_dim, var_n=var_n,
                         activation=activation, prior_means=prior_means, prior_stds=prior_stds,
                         weights_to_track=weights_to_track)
        self.variational_amat = None
        if self.learn_prior:
            raise NotImplementedError
        self.rank = rank
        if self.weights_to_track is not None:
            self.variational_amat_history = [np.array([]).reshape((0, nw, self.rank)) for nw in self.weights_to_track]
        # check alpha value
        self.alpha = alpha
        if self.alpha <= 0. or self.alpha >= 1.:
            raise ValueError('Input alpha must be between 0 and 1')

        # Update the graph to compute the cost
        with self.graph.as_default():
            # Initialize necessary variables
            self._initialize_variables_in_graph()
            inv_cov, log_det_cov = self.process_covariance(
                rank=self.rank, diag_sigmas=self.tf_variational_sigma, amat=self.tf_variational_amat)

            # Branch to sample weights from variational distribution
            tf_network_weights = self.sample_weights_from_variational(ns=self.ns_, evaluate_log_pdf=False)

            # Branch to generate predictions
            self.predictions = self.compute_predictions(X=self.X_, network_weights=tf_network_weights)

            ndata = tf.shape(self.X_)[0]  # number of independent data points
            normalization_term = 0.  # contribution of normalization terms to cost: -log(Z(prior))-log(Z(q))
            # Cost is based on the alpha-divergence minimization
            m0, var0 = [], []
            for w_shape, w_dim, prior_mean, prior_std in zip(
                    self.weights_shape, self.weights_dim, self.prior_means, self.prior_stds):
                # compute normalization term: log(Z(prior))
                # log(Z)=log(sqrt(2*pi)*sig)+mu**2/(2sig**2)
                normalization_term += np.prod(w_shape) * (
                        tf.log(prior_std) + 0.5 * tf.square(tf.divide(prior_mean, prior_std)))
                m0 += [prior_mean, ] * w_dim
                var0 += [prior_std ** 2, ] * w_dim
            m0 = tf.constant(m0)
            var0 = tf.constant(var0)

            # compute normalization term: -log(Z(q))=-0.5*(mu.T C-1 mu + log det C)
            normalization_term -= 0.5 * (tf.matmul(tf.matmul(
                tf.reshape(self.tf_variational_mu, (1, -1)),
                inv_cov),
                tf.reshape(self.tf_variational_mu, (-1, 1)))[0] + log_det_cov)

            # log of the site approximations f: log(f(w))=s(w).T lambda_{f}
            log_factor = self.compute_log_site_approximation(
                network_weights=tf_network_weights, mu_prior_vec=m0, var_prior_vec=var0,
                mu_post_vec=self.tf_variational_mu, inv_cov_post=inv_cov, ndata=ndata)

            # tie factors: use same site approximation for all data points
            # log_factor = tf.tile(tf.expand_dims(log_factor, 1), [1, ndata])
            log_factor = tf.tile(log_factor, [1, ndata])
            assert len(log_factor.get_shape().as_list()) == 2
            # compute likelihood of all data points n separately
            loglike = -1 * self.neg_log_like(
                y_true=tf.tile(tf.expand_dims(self.y_, 0), [self.ns_, 1, 1]), y_pred=self.predictions, do_sum=False,
                weights_data=self.w_)
            assert len(loglike.get_shape().as_list()) == 2
            # compute log(E_{q}[(like_n(w)/f(w)) ** alpha]) for all data points n,
            # expectation is computed by averaging over the ns_ samples
            logexpectation = tf.add(
                tf.log(1. / tf.to_float(self.ns_)),
                tf.reduce_logsumexp(self.alpha * tf.subtract(loglike, log_factor), axis=0)
            )
            # cste_over_ns = tf.reduce_max(self.alpha * (loglike - log_factor), axis=0)
            # in_exp = self.alpha * (loglike - log_factor) - tf.tile(tf.expand_dims(cste_over_ns, 0),
            #                                                        [self.ns_, 1])
            # logexpectation = cste_over_ns - tf.log(tf.to_float(self.ns_)) + tf.reduce_logsumexp(in_exp, axis=0)
            # in the cost, sum over all the data points n
            self.cost = normalization_term - 1. / self.alpha * tf.reduce_sum(logexpectation)

            # Set-up the training procedure
            self.lr_ = tf.placeholder(tf.float32, name='lr_', shape=())
            opt = tf_optimizer(learning_rate=self.lr_)
            var_list = [self.tf_variational_mu, self.tf_variational_rho, self.tf_variational_amat]
            total_grads_and_vars = opt.compute_gradients(self.cost, var_list)
            total_grads_and_vars = [(tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad), val)
                                    for grad, val in total_grads_and_vars]
            self.grad_step = opt.apply_gradients(total_grads_and_vars)

    @staticmethod
    def process_covariance(rank, diag_sigmas, amat):
        """ Compute inverse of covariance and log det covariance, for Cov=D+AA.T """
        inv_D = tf.diag(1. / diag_sigmas ** 2)
        chol_temp = tf.cholesky(
            tf.eye(rank) + tf.matmul(tf.transpose(amat) / diag_sigmas ** 2, amat)
        )
        Kmat = tf.matmul(tf.matmul(inv_D, amat), tf.matrix_inverse(tf.transpose(chol_temp)))
        inv_cov = inv_D - tf.matmul(Kmat, tf.transpose(Kmat))
        log_det_cov = 2 * (tf.reduce_sum(tf.log(tf.diag_part(chol_temp))) + tf.reduce_sum(tf.log(diag_sigmas)))
        return inv_cov, log_det_cov

    @staticmethod
    def compute_log_site_approximation(network_weights, mu_prior_vec, var_prior_vec, mu_post_vec, inv_cov_post, ndata):
        """
        Compute factors for alpha BB algorithm
        """
        # Compute lambdas for prior and var_post
        lmda_1_prior = tf.matmul(tf.diag(1. / var_prior_vec), tf.reshape(mu_prior_vec, (-1, 1)))
        lmda_1_post = tf.matmul(inv_cov_post, tf.reshape(mu_post_vec, (-1, 1)))
        lmda_1 = tf.cast(1 / ndata, tf.float32) * (lmda_1_post - lmda_1_prior)
        lmda_2 = tf.cast(1 / (2 * ndata), tf.float32) * (tf.diag(1. / var_prior_vec) - inv_cov_post)
        # Compute log site approx
        ns_local = tf.shape(network_weights[0])[0]
        w = []
        for w_l in network_weights:
            w.append(tf.reshape(w_l, [ns_local, -1]))
        w = tf.concat(w, axis=1)
        log_factor = tf.reshape(tf.reduce_sum(tf.matmul(w, lmda_2) * w, axis=-1), [ns_local, 1]) + tf.matmul(w, lmda_1)
        return log_factor

    def _initialize_variables_in_graph(self):
        """
        Initialize some variables in VI graph
        """
        self.tf_tracked_means = []
        self.tf_tracked_stds = []
        self.tf_tracked_amat = []

        # add dense layers, add contributions of each layer to prior and variational posterior costs
        standard_sigmas = compute_standard_sigmas(hidden_units=self.hidden_units, input_dim=self.input_dim,
                                                  output_dim=self.output_dim, scale=1., mode='fan_avg')
        standard_sigmas_aug = []
        [standard_sigmas_aug.extend([std, 0.01]) for std in standard_sigmas]
        start_sigmas = []
        for std, w_dim in zip(standard_sigmas_aug, self.weights_dim):
            start_sigmas += [std, ] * w_dim
        self.tf_variational_mu = tf.Variable(tf.random_normal(shape=(self.n_weights, ), mean=0., stddev=start_sigmas),
                                             trainable=True, dtype=tf.float32)
        self.tf_variational_rho = tf.Variable(-6.9 * tf.ones(shape=(self.n_weights, ), dtype=tf.float32),
                                              trainable=True, dtype=tf.float32)
        self.tf_variational_sigma = self._sigma(self.tf_variational_rho)
        # self.tf_variational_amat = tf.Variable(tf.random_normal(shape=(self.n_weights, self.rank), mean=0.,
        #                                                         stddev=0.001 / self.rank),
        #                                        trainable=True, dtype=tf.float32)
        self.tf_variational_amat = tf.Variable(tf.zeros(shape=(self.n_weights, self.rank), dtype=tf.float32),
                                               trainable=True, dtype=tf.float32)

        # Keep track of some of the weights
        if self.weights_to_track is not None:
            c = 0
            for i, w_dim in enumerate(self.weights_dim):
                self.tf_tracked_means.append(self.tf_variational_mu[c:c + self.weights_to_track[i]])
                self.tf_tracked_stds.append(self.tf_variational_sigma[c:c + self.weights_to_track[i]])
                self.tf_tracked_amat.append(self.tf_variational_amat[c:c + self.weights_to_track[i], :])
                c += w_dim

    def sample_weights_from_variational(self, ns, random_seed=None, evaluate_log_pdf=False, sum_over_ns=False):
        """
        Samples weights w for the NN from the variational density q_{theta}(w) (a mixture of gaussian)

        :return: weights w, as a list of length 2 * n_uq_layers
        """
        log_q = 0.
        mu = tf.tile(tf.expand_dims(self.tf_variational_mu, axis=0), [ns, 1])
        sigma = tf.tile(tf.expand_dims(self.tf_variational_sigma, axis=0), [ns, 1])
        w = tf.add(
            tf.multiply(sigma, tf.random_normal(shape=(ns, self.n_weights), mean=0., stddev=1.)),
            tf.matmul(tf.random_normal(shape=(ns, self.rank), mean=0., stddev=1.),
                      tf.transpose(self.tf_variational_amat)))
        w = tf.add(w, mu)
        weights = []
        tot_dim = 0
        for w_dim, w_shape in zip(self.weights_dim, self.weights_shape):
            weights.append(tf.reshape(w[:, tot_dim:tot_dim + w_dim], (ns, ) + w_shape))
            tot_dim += w_dim

        if evaluate_log_pdf:
            chol = tf.cholesky(tf.diag(self.tf_variational_sigma ** 2) + tf.matmul(
                self.tf_variational_amat, tf.transpose(self.tf_variational_amat)))
            log_q += log_multiv_gaussian(x=w, mean=self.tf_variational_mu, cov_chol=chol,
                                         axis_sum=(None if sum_over_ns else []))
            return weights, log_q
        return weights

    def fit(self, X, y, weights_data=None, ns=10, epochs=100, verbose=0, lr=0.001):
        """
        Fit, i.e., find the variational distribution that minimizes the cost function
        """
        # Initialize tensorflow session and required variables
        self.training_data = (X, y)
        if weights_data is None:
            weights_data = np.ones((X.shape[0], ))

        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            # Run training loop
            for e in range(epochs):
                # Loop over all minibatches
                _, loss_history_ = sess.run(
                    [self.grad_step, self.cost],
                    feed_dict={self.w_: weights_data, self.X_: X, self.y_: y, self.ns_: ns, self.lr_: lr})
                self.loss_history.append(loss_history_)
                # Save some of the weights
                if self.weights_to_track is not None:
                    mean, std, amat = sess.run(
                        [self.tf_tracked_means, self.tf_tracked_stds, self.tf_tracked_amat], feed_dict={self.ns_: ns})
                    self.variational_mu_history = [np.concatenate([m1, m2.reshape((1, -1))], axis=0)
                                                   for m1, m2 in zip(self.variational_mu_history, mean)]
                    self.variational_sigma_history = [np.concatenate([m1, m2.reshape((1, -1))], axis=0)
                                                      for m1, m2 in zip(self.variational_sigma_history, std)]
                    self.variational_amat_history = [np.concatenate([m1, m2.reshape((1, -1, self.rank))], axis=0)
                                                     for m1, m2 in zip(self.variational_amat_history, amat)]
                # print comments on terminal
                if verbose:
                    print('epoch = {}, loss = {}'.format(e, self.loss_history[-1]))

            # Save the final variational parameters
            self.variational_mu, self.variational_sigma, self.variational_amat = sess.run(
                [self.tf_variational_mu, self.tf_variational_sigma, self.tf_variational_amat], feed_dict={self.ns_: ns})

    def predict_uq(self, X, ns, return_std=True, return_MC=10, return_percentiles=(2.5, 97.5),
                   aleatoric_in_std_perc=True, aleatoric_in_MC=False):
        """
        Predict y for new input X, along with uncertainty.
        """
        feed_dict = {self.tf_variational_mu: self.variational_mu}
        feed_dict.update({self.tf_variational_sigma: self.variational_sigma})
        feed_dict.update({self.tf_variational_amat: self.variational_amat})
        feed_dict.update({self.X_: X, self.ns_: ns})
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer(), feed_dict={self.ns_: ns})
            y_MC = sess.run(self.predictions, feed_dict=feed_dict)
        outputs = compute_and_return_outputs(
            y_MC=y_MC, var_aleatoric=self.var_n, return_std=return_std, return_percentiles=return_percentiles,
            return_MC=return_MC, aleatoric_in_std_perc=aleatoric_in_std_perc, aleatoric_in_MC=aleatoric_in_MC)
        return outputs

    def return_marginals(self):
        """ Return the mean and std in all dimensions """
        mean, std = [], []
        c = 0
        var = self.variational_sigma ** 2 + np.diag(np.matmul(self.variational_amat, self.variational_amat.T))
        for w_dim, w_shape in zip(self.weights_dim, self.weights_shape):
            mean.append(self.variational_mu[c:c + w_dim].reshape(w_shape))
            std.append(np.sqrt(var[c:c + w_dim].reshape(w_shape)))
            c += w_dim
        return mean, std


class BayesByBackpropMixture(VIRegressor):
    """
    BayesByBackprop algorithm, from 'Weight Uncertainty in Neural Networks', Blundell et al., 2015.
    Use a mixture of Gaussians as an approximate posterior.

    **Inputs:**

    :param tf_optimizer: optimizer, defaults to Adam optimizer
    """

    def __init__(self, hidden_units, input_dim=1, output_dim=1, var_n=1e-6, activation=tf.nn.relu, prior_means=0.,
                 prior_stds=1., weights_to_track=None, tf_optimizer=tf.train.AdamOptimizer,
                 ncomp=1, lower_bound=False):

        # Initial checks and computations for the network
        super().__init__(hidden_units=hidden_units, input_dim=input_dim, output_dim=output_dim, var_n=var_n,
                         activation=activation, prior_means=prior_means, prior_stds=prior_stds,
                         weights_to_track=weights_to_track, random_seed=None)
        # if self.learn_prior:
        #     raise NotImplementedError
        self.ncomp = ncomp
        self.lower_bound = lower_bound

        # Update the graph to compute the cost function
        with self.graph.as_default():
            # Initialize necessary variables
            self._initialize_variables_in_graph()
            if self.learn_prior:
                self._initialize_prior_variables_in_graph()

            if not self.lower_bound:
                # Sample weights w from variational distribution q_{theta} and compute log(q_{theta}(w))
                tf_network_weights, var_post_term = self.sample_weights_from_variational(
                    ns=self.ns_, evaluate_log_pdf=True, sum_over_ns=True)
                var_post_term /= tf.to_float(self.ns_ * self.ncomp)

                # Compute the cost from the prior term -log(p(w))
                prior_term = self.log_prior_pdf(
                    network_weights=tf_network_weights, sum_over_ns=True) / tf.to_float(self.ns_ * self.ncomp)
            else:
                # Sample weights w from variational distribution q_{theta}
                tf_network_weights = self.sample_weights_from_variational(
                    ns=self.ns_, evaluate_log_pdf=False, sum_over_ns=True)

                var_post_term = 0.
                tf_m = tf.concat([tf.reshape(m, (self.ncomp, w_dim))
                                 for m, w_dim in zip(self.tf_variational_mu, self.weights_dim)], axis=-1)
                tf_std = tf.concat([tf.reshape(std, (self.ncomp, w_dim))
                                   for std, w_dim in zip(self.tf_variational_sigma, self.weights_dim)], axis=-1)
                for k in range(self.ncomp):
                    log_zks = log_gaussian(x=tf_m, mean=tf_m[k, :], std=tf.sqrt(tf_std ** 2 + tf_std[k, :] ** 2),
                                           axis_sum=-1)
                    # log_zks = []
                    # for j in range(self.ncomp):
                    #     log_z_kj = 0
                    #     for m, std in zip(self.tf_variational_mu, self.tf_variational_sigma):
                    #         log_z_kj += log_gaussian(
                    #             x=m[j, ...], mean=m[k, ...], std=tf.sqrt(std[j, ...] ** 2 + std[k, ...] ** 2))
                    #     log_zks.append(log_z_kj)
                    # log_zks = tf.stack(log_zks)
                    var_post_term += tf.reduce_logsumexp(tf.log(1. / self.ncomp) + log_zks)
                var_post_term /= self.ncomp

                # Compute the cost from the prior term -log(p(w))
                prior_term = 0.
                for m, std, m0, std0 in zip(
                        self.tf_variational_mu, self.tf_variational_sigma, self.prior_means, self.prior_stds):
                    prior_term += tf.reduce_sum(
                        - 0.5 * tf.log(2 * np.pi * std0 ** 2)
                        - 0.5 * (std ** 2 + (m - m0) ** 2) / std0 ** 2)
                prior_term /= self.ncomp

            # Branch to generate predictions
            predictions = self.compute_predictions(network_weights=tf_network_weights, X=self.X_)

            # Compute contribution of likelihood to cost: -log(p(data|w))
            neg_likelihood_term = self.neg_log_like(
                y_true=tf.tile(tf.expand_dims(self.y_, 0), [self.ns_ * self.ncomp, 1, 1]),
                y_pred=predictions) / tf.to_float(self.ns_ * self.ncomp)

            # Cost is based on the KL-divergence minimization
            self.cost = neg_likelihood_term - prior_term + var_post_term

            # Set-up the training procedure
            self.lr_ = tf.placeholder(tf.float32, name='lr_', shape=())
            self.opt = tf_optimizer(learning_rate=self.lr_)
            var_list = [self.tf_variational_mu, self.tf_variational_rho]
            if self.learn_prior:
                var_list = var_list + [self.tf_prior_mu, self.tf_prior_rho]
            grads_and_vars = self.opt.compute_gradients(self.cost, var_list)
            self.grad_step = self.opt.apply_gradients(grads_and_vars)

    def _initialize_variables_in_graph(self):
        """
        Initialize some variables in VI graph
        """
        self.tf_tracked_means = []
        self.tf_tracked_stds = []
        self.tf_variational_mu = []
        self.tf_variational_rho = []
        self.tf_variational_sigma = []

        # add dense layers, add contributions of each layer to prior and variational posterior costs
        standard_sigmas = compute_standard_sigmas(hidden_units=self.hidden_units, input_dim=self.input_dim,
                                                  output_dim=self.output_dim, scale=1., mode='fan_avg')
        start_sigmas = []
        [start_sigmas.extend([std, 0.01]) for std in standard_sigmas]
        for l, (start_std, w_shape, w_dim) in enumerate(zip(start_sigmas, self.weights_shape, self.weights_dim)):
            # Define the parameters of the variational distribution to be trained: theta={mu, rho} for kernel and bias
            mu = tf.Variable(tf.random_normal(shape=(self.ncomp, ) + w_shape, mean=0., stddev=start_std),
                             trainable=True, dtype=tf.float32)
            rho = tf.Variable(-6.9 * tf.ones(shape=(self.ncomp, ) + w_shape, dtype=tf.float32),
                              trainable=True, dtype=tf.float32)
            sigma = tf.log(1. + tf.exp(rho))

            self.tf_variational_mu.append(mu)
            self.tf_variational_rho.append(rho)
            self.tf_variational_sigma.append(sigma)

            # Keep track of some of the weights
            if self.weights_to_track is not None:
                self.tf_tracked_means.append(tf.reshape(mu[0, ...], shape=(w_dim, ))[:self.weights_to_track[l]])
                self.tf_tracked_stds.append(tf.reshape(sigma[0, ...], shape=(w_dim, ))[:self.weights_to_track[l]])

    def sample_weights_from_variational(self, ns, random_seed=None, evaluate_log_pdf=False, sum_over_ns=False):
        """
        Samples weights w for the NN from the variational density q_{theta}(w) (a mixture of gaussian)

        :return: weights w, as a list of length 2 * n_uq_layers
        """
        weights = []
        log_q = 0.
        ns_local = ns * self.ncomp
        for vi_mu, vi_sigma, w_shape in zip(self.tf_variational_mu, self.tf_variational_sigma, self.weights_shape):
            lenp1 = len(w_shape) + 1
            mu = tf.tile(tf.expand_dims(vi_mu, axis=0), [ns, ] + [1, ] * lenp1)
            sigma = tf.tile(tf.expand_dims(vi_sigma, axis=0), [ns, ] + [1, ] * lenp1)
            w = tf.add(
                mu, tf.multiply(sigma, tf.random_normal(shape=(ns, self.ncomp) + w_shape, mean=0., stddev=1.)))
            w = tf.reshape(w, (ns_local, ) + w_shape)
            weights.append(w)
            if evaluate_log_pdf:
                log_q += log_gaussian(x=tf.tile(tf.expand_dims(w, axis=1), [1, self.ncomp] + [1, ] * len(w_shape)),
                                      mean=tf.tile(tf.expand_dims(vi_mu, axis=0), [ns_local, ] + [1, ] * lenp1),
                                      std=tf.tile(tf.expand_dims(vi_sigma, axis=0), [ns_local, ] + [1, ] * lenp1),
                                      axis_sum=list(range(-len(w_shape), 0)))
        if evaluate_log_pdf:
            log_q += tf.log(1. / tf.cast(self.ncomp, tf.float32))
            if sum_over_ns:
                log_q = tf.reduce_sum(tf.reduce_logsumexp(log_q, axis=1), axis=None)
            else:
                log_q = tf.reduce_logsumexp(log_q, axis=1)
            return weights, log_q
        return weights

    def return_marginals(self, index_component=0):
        """ Return the mean and std in all dimensions """
        #mean = [np.mean(m, axis=0) for m in self.variational_mu]
        #std = [np.sqrt(np.mean(std**2, axis=0) + np.mean((m - mtot) ** 2, axis=0)) for std, m, mtot
        #       in zip(self.variational_sigma, self.variational_mu, mean)]
        return [mu[index_component, ...] for mu in self.variational_mu], \
               [sigma[index_component, ...] for sigma in self.variational_sigma]

    def rank_network_weights(self, rank_metric='information gain', return_mask=False, threshold_on_number=None,
                             keep_last_bias=False):
        """
        Rank weights in the network according to results of the VI approximation and some metrics (SNR of info gain)
        :param rank_metric: 'snr' or 'info gain'
        :param return_mask: bool, whether to return the mask of most 'important' weights
        :param threshold_on_number: max number of weights to keep
        :param threshold_on_metric_perc: keep weights so that a given percentage of the metric is achieved
        :param threshold_on_metric: keep weights for which metric > this threshold (try 0.83)

        :return: metric_values: metric value for all weights, list (length 2 * n_uq_layers) of ndarrays
        :return: importance_mask: indicates 'important' weights, list (length 2 * n_uq_layers) of boolean ndarrays
        """
        if rank_metric.lower() == 'snr':
            metric_values = [np.abs(mu) / std for (mu, std) in zip(self.variational_mu, self.variational_sigma)]
        elif rank_metric.lower() == 'information gain':
            if self.learn_prior:
                prior_means = self.variational_prior_mu
                prior_stds = self.variational_prior_sigma
            else:
                prior_means = self.prior_means
                prior_stds = self.prior_stds
            metric_values = [
                kl_div_gaussians(mean_prior=mu_prior, std_prior=std_prior, mean_posterior=mu, std_posterior=std)
                for (mu, std, mu_prior, std_prior)
                in (zip(self.variational_mu, self.variational_sigma, prior_means, prior_stds))]
        else:
            raise ValueError('rank_distance can be either "snr" or "information gain".')
        if not return_mask:
            return metric_values

        # Also return the mask of importance weights
        importance_mask = []
        for i, (metric_val, w_shape) in enumerate(zip(metric_values, self.weights_shape)):
            ranking_bool_comps = []
            for j in range(self.ncomp):
                ranking_bool = extract_mask_from_vi_ranking(
                    rank_metric=rank_metric, metric_values=metric_val[j].reshape((-1, )),
                    threshold_on_number=(None if threshold_on_number is None else threshold_on_number[i]))
                ranking_bool_comps.append(ranking_bool.reshape(w_shape))
            importance_mask.append(np.stack(ranking_bool_comps, axis=0))
        if keep_last_bias:
            importance_mask[-1] = np.ones_like(importance_mask[-1]).astype(bool)
        self.n_weights_after_pruning = [sum([np.sum(m[j]) for m in importance_mask]) for j in range(self.ncomp)]
        return metric_values, importance_mask

    def rearrange_weights(self, mask_low_dim):
        """ Rearrange weights so that important ones are all in the same place """
        for layer in range(self.n_uq_layers - 1):
            for comp in range(self.ncomp):
                mask_kernel = mask_low_dim[2 * layer][comp]
                for row, row_mask_kernel in enumerate(mask_kernel):
                    c = 0
                    ind_imp = np.nonzero(row_mask_kernel)[0]
                    if len(ind_imp) > 0:   # there are important weights there
                        for i in ind_imp:
                            # modify current output (kernel and bias)
                            mask_low_dim[2 * layer][comp][row][[c, i]] = mask_low_dim[2 * layer][comp][row][[i, c]]
                            mask_low_dim[2 * layer + 1][comp][[c, i]] = mask_low_dim[2 * layer + 1][comp][[i, c]]
                            self.variational_mu[2 * layer][comp][row][[c, i]] = \
                                self.variational_mu[2 * layer][comp][row][[i, c]]
                            self.variational_mu[2 * layer + 1][comp][[c, i]] = \
                                self.variational_mu[2 * layer + 1][comp][[i, c]]
                            self.variational_sigma[2 * layer][comp][row][[c, i]] = \
                                self.variational_sigma[2 * layer][comp][row][[i, c]]
                            self.variational_sigma[2 * layer + 1][comp][[c, i]] = \
                                self.variational_sigma[2 * layer + 1][comp][[i, c]]
                            # modify next input to match
                            mask_low_dim[2 * layer + 2][comp][[c, i]] = mask_low_dim[2 * layer + 2][comp][[i, c]]
                            self.variational_mu[2 * layer + 2][comp][[c, i]] = \
                                self.variational_mu[2 * layer + 2][comp][[i, c]]
                            self.variational_sigma[2 * layer + 2][comp][[c, i]] = \
                                self.variational_sigma[2 * layer + 2][comp][[i, c]]
                            c += 1
        #return mask_low_dim
        return [m[0] for m in mask_low_dim]


class ModelAveraging:
    def __init__(self, nn_dict, training_data, n_bootstrap=100):
        self.nn = nn_dict
        self.training_data = training_data
        self.n_bootstrap = n_bootstrap
        self.deltas_bb = [np.random.dirichlet([1.] * self.training_data[0].shape[0]) for _ in range(n_bootstrap)]

        self.alpha_list = []
        self.random_seed_list = []

        self.regressors = []
        self.all_lpd = []
        self.elpd = []
        self.modified_elpd = []
        self.elpd_bb_sample = []

        self.weights_elpd = None
        self.weights_modified_elpd = None
        self.weights_elpd_bb = None

    def _compute_elpd_and_modifs(self, loo_logpd):
        """
        Compute various versions of the model weights
        :param all_logpd:
        :return: weights from elpd, modified version (with standard error) and BB
        """
        elpd = np.sum(loo_logpd)
        # Modified version with standard error
        error_elpd = np.sqrt(np.sum((loo_logpd - np.mean(loo_logpd)) ** 2))
        modified_elpd = elpd - 0.5 * error_elpd
        # Bayesian bootstrap version
        elpd_bb_sample = np.empty((self.n_bootstrap, ))
        for b, delta in enumerate(self.deltas_bb):
            assert delta.shape == loo_logpd.shape
            elpd_bb_sample[b] = len(loo_logpd) * np.sum(delta * loo_logpd)
        return elpd, modified_elpd, elpd_bb_sample

    def _compute_weights(self):
        from scipy.special import logsumexp
        self.weights_elpd = np.exp(np.array(self.elpd) - logsumexp(self.elpd))
        self.weights_modified_elpd = np.exp(np.array(self.modified_elpd) - logsumexp(self.modified_elpd))
        elpd_bb_sample_list = np.array(self.elpd_bb_sample)
        assert(elpd_bb_sample_list.shape[1] == self.n_bootstrap)
        weights_bb_sample = np.exp(
            elpd_bb_sample_list - np.tile(logsumexp(elpd_bb_sample_list, axis=0), [len(self.elpd), 1]))
        self.weights_elpd_bb = np.mean(weights_bb_sample, axis=1)

    def predict_uq(self, X, ns, return_std=True, return_percentiles=(2.5, 97.5),
                   aleatoric_in_std_perc=True, aleatoric_in_MC=False, return_MC=0,
                   weights_attribute='weights_modified_elpd'):
        """
        Predict y for new input X, along with uncertainty
        """
        perc_ns = [int(np.round(ns * w_)) for w_ in getattr(self, weights_attribute)]
        mc_ns = [int(np.round(return_MC * w_)) for w_ in getattr(self, weights_attribute)]
        y_mean, y_std, y_MC_perc, y_MC_MC = [], [], [], []
        for ns1, ns2, reg_props in zip(perc_ns, mc_ns, self.regressors):
            reg = set_properties_vi(*reg_props)
            outs = reg.predict_uq(
                X=X, ns=ns, return_MC=0, return_std=return_std, return_percentiles=(), aleatoric_in_MC=False,
                aleatoric_in_std_perc=aleatoric_in_std_perc)
            if return_std:
                y_mean.append(outs[0])
                y_std.append(outs[1])
            else:
                y_mean.append(outs)
            if return_percentiles and ns1 > 0:
                _, y_MC = reg.predict_uq(
                    X=X, ns=ns1, return_MC=ns1, return_std=False, return_percentiles=(), aleatoric_in_MC=False)
                y_MC_perc.append(y_MC)
            if return_MC > 0 and ns2 > 0:
                _, y_MC = reg.predict_uq(X=X, ns=ns2, return_MC=ns2, return_std=False, return_percentiles=(),
                                         aleatoric_in_MC=aleatoric_in_MC)
                y_MC_MC.append(y_MC)
            del reg
        y_mean = np.array(y_mean)
        y_mean_out = np.average(y_mean, axis=0, weights=getattr(self, weights_attribute))
        outputs = (y_mean_out, )
        if len(y_std) > 0:
            y_var = np.average(np.array(y_std) ** 2 + (y_mean - y_mean_out) ** 2, axis=0,
                               weights=getattr(self, weights_attribute))
            y_std = np.sqrt(y_var)
            outputs = outputs + (y_std, )
        if len(y_MC_perc) > 0:
            y_MC_perc = np.concatenate(y_MC_perc, axis=0)
            # Compute statistical outputs from MC values
            _, outputs_perc = compute_and_return_outputs(
                y_MC=y_MC_perc, var_aleatoric=self.nn['var_n'], return_std=False,
                return_percentiles=return_percentiles, return_MC=0, aleatoric_in_std_perc=aleatoric_in_std_perc)
            outputs = outputs + (outputs_perc, )
        if len(y_MC_MC) > 0:
            outputs = outputs + (np.concatenate(y_MC_MC, axis=0), )
        return outputs


class ModelAveragingLOO(ModelAveraging):
    def __init__(self, nn_dict, training_data, alpha_list=None, random_seed_list=None, training_dict=None,
                 n_bootstrap=100):
        super().__init__(nn_dict, training_data, n_bootstrap=n_bootstrap)
        if alpha_list is not None and random_seed_list is not None:
            for alpha, random_seed in zip(alpha_list, random_seed_list):
                self.add_one_model(alpha=alpha, random_seed=random_seed, training_dict=training_dict)

    def add_one_model(self, alpha, random_seed, training_dict):
        print('Adding model with alpha={}'.format(alpha))
        X_train, y_train = self.training_data
        # for ndata regressors
        log_pis = []
        for ind_data in range(X_train.shape[0]):
            # fit to data-i
            indices_data = [i for i in range(X_train.shape[0]) if i != ind_data]
            if alpha == 0:
                reg = BayesByBackprop(analytical_grads=True, random_seed=random_seed, **self.nn)
            else:
                reg = alphaBB(alpha=alpha, random_seed=random_seed, **self.nn)
            reg.fit(X=X_train[indices_data, :], y=y_train[indices_data, :], **training_dict)
            # compute log p(yi|data-i)
            log_pi = reg.compute_predictive_density(
                X=X_train[np.newaxis, ind_data, :], y=y_train[np.newaxis, ind_data, :], ns=10000)
            log_pis.append(log_pi)
            del reg
        # compute unnormalized weight and fit to all data
        if alpha == 0:
            reg = BayesByBackprop(analytical_grads=True, random_seed=random_seed, **self.nn)
        else:
            reg = alphaBB(alpha=alpha, random_seed=random_seed, **self.nn)
        reg.fit(X=X_train, y=y_train, **training_dict)

        self.alpha_list.append(alpha)
        self.random_seed_list.append(random_seed)
        self.regressors.append(save_properties_vi(reg))
        del reg
        self.all_lpd.append(np.array(log_pis))
        elpd, modified_elpd, elpd_bb_sample = self._compute_elpd_and_modifs(loo_logpd=np.array(log_pis))
        self.elpd.append(elpd)
        self.modified_elpd.append(modified_elpd)
        self.elpd_bb_sample.append(elpd_bb_sample)
        # self.error_elpd.append(np.sqrt(np.sum((log_pis - np.mean(log_pis)) ** 2)))

        self._compute_weights()


class ModelAveragingLOOalphaBB(ModelAveraging):
    def __init__(self, nn_dict, training_data, alpha_list=None, random_seed_list=None, training_dict=None,
                 n_bootstrap=100):
        super().__init__(nn_dict, training_data, n_bootstrap=n_bootstrap)
        if alpha_list is not None and random_seed_list is not None:
            for alpha, random_seed in zip(alpha_list, random_seed_list):
                if float(alpha) == 0.:
                    alpha = 0.0001
                self.add_one_model(alpha=alpha, random_seed=random_seed, training_dict=training_dict)

    def add_one_model(self, alpha, random_seed, training_dict):
        print('Adding model with alpha={}'.format(alpha))
        X_train, y_train = self.training_data
        reg = alphaBB(alpha=alpha, random_seed=random_seed, **self.nn)
        reg.fit(X=X_train, y=y_train, **training_dict)
        log_pis = reg.compute_all_loo_predictive_densities(ns=10000)
        #reg_props = save_properties_vi(reg)
        #log_pis = []
        #for ind_data in range(X_train.shape[0]):
        #    del reg
        #    reg = set_properties_vi(*reg_props)
        #    # compute log p(yi|data-i)
        #    log_pi = reg.compute_lso_predictive_density(leave_factors=[ind_data, ], ns=10000)
        #    log_pis.append(log_pi)
        #elpd, se, log_pis = reg.compute_elpd(ns=10000, return_log_pis=True)

        self.alpha_list.append(alpha)
        self.random_seed_list.append(random_seed)
        self.regressors.append(save_properties_vi(reg))
        del reg
        self.all_lpd.append(log_pis) # should be a matrix
        # self.elpd.append(np.sum(log_pis))
        # self.error_elpd.append(np.sqrt(np.sum((log_pis - np.mean(log_pis)) ** 2)))
        # self.elpd.append(elpd)
        # self.error_elpd.append(se)

        elpd, modified_elpd, elpd_bb_sample = self._compute_elpd_and_modifs(loo_logpd=log_pis)
        self.elpd.append(elpd)
        self.modified_elpd.append(modified_elpd)
        self.elpd_bb_sample.append(elpd_bb_sample)
        # self.error_elpd.append(np.sqrt(np.sum((log_pis - np.mean(log_pis)) ** 2)))

        self._compute_weights()