import numpy as np
import csv
import os
from pyDOE import lhs
from utils_v3 import generate_circular_fibers_random

n_data_ = [100, ]
n_same_ = [1, ]
n_inputs_ = [4, ]
type_distrib_ = ['beta', ]
for n_data, n_same, n_inputs, type_distrib in zip(n_data_, n_same_, n_inputs_, type_distrib_):

    extension_file = '_v3_inputs[{}]_ndata[{}]_nsame[{}]_'.format(n_inputs, n_data, n_same) + type_distrib.lower() + '_2'
    path_IO = 'IOfolder'+extension_file
    if not os.path.exists(path_IO):
        os.mkdir(path_IO)

    areaRVE = 1

    #########################   FIXED INPUTS  ###############################
    propFixedDict = {}
    # mesh properties
    propFixedDict['mesh_type'] = 'quad'
    propFixedDict['n_nodes_per_el'] = 4
    n_inclusions = 30
    propFixedDict['n_inclusions'] = n_inclusions
    # load
    propFixedDict['strain_x'] = 0.05
    propFixedDict['dt'] = 0.01
    # materials properties
    propFixedDict['matrix_type_material'] = 'plasticity'
    propFixedDict['fibers_type_material'] = 'elasticity'
    with open(path_IO+'/input_fixed_dict'+extension_file+'.csv', 'w') as csv_file:
        writer = csv.DictWriter(csv_file, propFixedDict.keys(), lineterminator = '\n')
        writer.writeheader()
        writer.writerow(propFixedDict)

    #########################   VARIABLE INPUTS  ###############################
    propVariableDict = {}
    propVariableDict['n_per_cluster'] = None
    propVariableDict['fibers_center'] = None
    propVariableDict['fibers_radius'] = None
    propVariableDict['point_in_matrix'] = None
    propVariableDict['vf'] = None
    propVariableDict['matrix_properties'] = None
    propVariableDict['fibers_properties'] = None
    propVariableDict['mesh_size'] = None
    with open(path_IO+'/input_variable_dict'+extension_file+'.csv', 'w') as csv_file:
        writer = csv.DictWriter(csv_file, propVariableDict.keys(), lineterminator = '\n')
        writer.writeheader()

    # Random properties
    if 'bis' in extension_file:
        bounds_list = [(0.05, 0.3), (200, 600), (300, 500), (0.2, 0.55)]
    else:
        bounds_list = [(0.05, 0.4), (200, 600), (300, 500), (0.2, 0.55)]
    #params_beta_list = [(2, 5), (3, 1.5), (5, 5), (3, 3)]
    params_beta_list = [(2., 4.), (3., 2.), (3., 3.), (3., 3.)]
    probs_clustering = [0.45, 0.3, 0.25]

    if type_distrib.lower() == 'beta':
        from numpy.random import beta
        random_design = np.zeros((n_data, n_inputs))
        for i, (bounds, params_beta) in enumerate(zip(bounds_list, params_beta_list)):
            random_design[:, i] = bounds[0] + (bounds[1]-bounds[0]) * beta(a=params_beta[0], b=params_beta[1], size=(n_data, ))
        if n_inputs == 5:
            random_design[:, -1] = np.random.choice([1, 3, 5], size=(n_data, ), replace=True, p=probs_clustering)
    elif type_distrib.lower() == 'random':
        random_design = np.zeros((n_data, n_inputs))
        for i, (bounds, params_beta) in enumerate(zip(bounds_list, params_beta_list)):
            random_design[:, i] = bounds[0] + (bounds[1]-bounds[0]) * np.random.rand(n_data)
        if n_inputs == 5:
            random_design[:, -1] = np.random.choice([1, 3, 5], size=(n_data, ), replace=True, p=probs_clustering)
    elif type_distrib.lower() == 'lhs':
        random_design = lhs(n=n_inputs, samples=n_data)
        for i, bounds in enumerate(bounds_list):
            random_design[:, i] = bounds[0] + (bounds[1]-bounds[0]) * random_design[:, i]
        if n_inputs == 5:
            mask_1 = random_design[:, -1] > 0.33333
            mask_2 = random_design[:, -1] > 0.66666
            random_design[:, -1] = 1
            random_design[mask_1, -1] = 3
            random_design[mask_2, -1] = 5


    for j in range(n_data):
        vf = random_design[j, 0]
        radius_inclusion = np.sqrt(areaRVE * vf / n_inclusions / np.pi)
        mesh_size = round(0.32*radius_inclusion / 2.8, 5) # slightly smaller than safe/2
        # microstructure properties
        propVariableDict['vf'] = vf
        propVariableDict['n_inclusions'] = n_inclusions
        propVariableDict['mesh_size'] = mesh_size
        # material properties
        Ef = random_design[j, 1]
        b = random_design[j, 2]
        c = random_design[j, 3]
        propVariableDict['matrix_properties'] = (100*10**9, 0.3, 400*10**6, b*10**6, c) #E_linear, nu_linear, plastic law parameters
        propVariableDict['fibers_properties'] = (Ef*10**9, 0.25)
        if n_inputs == 5:
            n_per_cluster = random_design[j, 4]
        else:
            n_per_cluster = 1
        propVariableDict['n_per_cluster'] = n_per_cluster

        for i in range(n_same):
            radius_inclusion, centers, point_in_matrix = generate_circular_fibers_random(GVf=vf, n_inclusions = n_inclusions, 
                n_per_cluster = int(n_per_cluster))
            propVariableDict['fibers_center'] = tuple([(center[0], center[1]) for center in centers])
            propVariableDict['fibers_radius'] = radius_inclusion
            propVariableDict['point_in_matrix'] = tuple(point_in_matrix)

            with open(path_IO+'/input_variable_dict'+extension_file+'.csv', 'a') as csv_file:
                writer = csv.DictWriter(csv_file, propVariableDict.keys(), lineterminator = '\n')
                writer.writerow(propVariableDict)
