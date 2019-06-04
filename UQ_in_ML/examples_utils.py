import numpy as np
from functools import partial
from .nn_utils import build_scaling_layers

# Define benchmark problems
def f_cubic(x, var_n=9., noisy=False):
    y = x**3 / 100
    if noisy:
        y += np.random.normal(loc=0, scale=np.sqrt(var_n), size=x.shape)
    return y


def f_sin_cos(x, var_n=0.05**2, noisy=False):
    y = 0.4*np.sin(x)+0.5*np.cos(3*x)
    if noisy:
        y += np.random.normal(loc=0, scale=np.sqrt(var_n), size=x.shape)
    return y


def g_homoscedastic(x, var_n=0.02**2, noisy=False):
    y = x + 0.3 * np.sin(2 * np.pi * x) + 0.3 * np.sin(4 * np.pi * x)
    if noisy:
        y += np.random.normal(loc=0, scale=np.sqrt(var_n), size=x.shape)
    return y


def g_heteroscedastic(x, var_n=0.02**2, noisy=False):
    if noisy:
        eps = np.random.normal(loc=0, scale=np.sqrt(var_n), size=x.shape)
    else:
        eps = np.zeros_like(x)
    y = x + 0.3 * np.sin(2 * np.pi * (x + eps)) + 0.3 * np.sin(4 * np.pi * (x + eps)) + eps
    return y


def linear_regression(x, var_n=1 ** 2, noisy=False):
    # in this example x is a 2-d feature vector
    y = -1.5 + 1. * x[:, 0] + 2. * x[:, 1]
    if noisy:
        y += np.random.normal(loc=0, scale=np.sqrt(var_n), size=y.shape)
    return y


# Define some
def set_problem(problem, n_data, var_n=None, as_dict=False):
    if problem == 'cubic':
        if var_n is None:
            var_n = 9.
        f = f_cubic
        xn = np.random.uniform(low=-4.5, high=4.5, size=(n_data, 1))
        bounds = (-6, 6)
    elif problem == 'cos_sin':
        if var_n is None:
            var_n = 0.05 ** 2
        f = partial(f_sin_cos, var_n=var_n)
        bounds = (-5, 5)
        xn = np.random.uniform(low=bounds[0], high=bounds[1], size=(n_data, 1))
    elif problem == 'linear_sin':
        if var_n is None:
            var_n = 0.02 ** 2
        f = partial(g_homoscedastic, var_n=var_n)
        xn = np.random.uniform(low=0., high=0.5, size=(n_data, 1))
        bounds = (-0.2, 1.2)
    elif problem == 'linear_regression':
        if var_n is None:
            var_n = 1 ** 2
        f = partial(linear_regression, var_n=var_n)
        xn = np.random.uniform(low=0., high=10, size=(n_data, 1))
        xn = np.concatenate([xn, xn ** 2], axis=1)
        bounds = (0, 10)
    else:
        raise ValueError('Problem is not defined')
    yn = f(xn, noisy=True)
    if as_dict:
        return {'f': f, 'var_n': var_n, 'bounds': bounds, 'xn': xn, 'yn': yn}
    return f, var_n, bounds, xn, yn


def set_default_network(problem, X_train, as_dict=False):
    if problem == 'cubic':
        units_per_layer = (14, 14, 14, 14,)
        prior = {'type': 'gaussian', 'variance': [2.]*len(units_per_layer)+[0.02]}
        pre_model = None
    elif problem == 'cos_sin':
        units_per_layer = (14, 14, 14, 14,)
        prior = {'type': 'gaussian', 'variance': [2.]*len(units_per_layer)+[0.02]}
        #pre_model = None
        pre_model = build_scaling_layers(X_train=X_train)
    elif problem == 'linear_sin':
        units_per_layer = (14, 14, 14, 14,)
        prior = {'type': 'gaussian', 'variance': [1.]*len(units_per_layer)+[0.05]}
        pre_model = None
    if as_dict:
        return {'units_per_layer': units_per_layer, 'prior': prior, 'pre_model': pre_model}
    return units_per_layer, prior, pre_model
