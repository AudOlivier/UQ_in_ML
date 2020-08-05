# UQ_in_ML

Algorithms for probabilistic training of neural networks.

The folder UQ_in_ML contains the core code implementing the various algorithms, which aim at approximating the posterior pdf of network weights based on training data, thus allowing for quantification of epistemic uncertainties within the data. 

Examples are provided in various jupuyter notebooks:
- Linear_Gaussian_Example.ipynb is a check of the algorithms on a linear problem for which the posterior pdf of the network weights can be computed analytically.
_ Cubic_VI.ipynb and Sinusoid_VI.ipynb illustrate the use of various algorithms on a 1D cubic function and 1D sinusoid function respectively.
