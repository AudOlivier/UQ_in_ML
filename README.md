# UQ_in_ML

Algorithms for probabilistic training of neural networks.

The folder UQ_in_ML contains the core code implementing the various algorithms, which aim at approximating the posterior pdf of network weights based on training data, thus allowing for quantification of epistemic uncertainties within the data. 

Examples are provided in various jupuyter notebooks:
- Introduction_VI_and_algorithms.ipynb is an introduction to variational inference and a check of the algorithms on a linear problem for which the posterior pdf of the network weights can be computed analytically.
- Cubic_VI.ipynb, Cubic_VI_extra.ipynb and Sinusoid_VI.ipynb illustrate the use of various algorithms on a 1D cubic function and 1D sinusoid function.
- Materials_data_Abaqus_v3_partA/B.ipynb illustrate the use of those probabilistic networks for surrogate modeling in a materials example problem.