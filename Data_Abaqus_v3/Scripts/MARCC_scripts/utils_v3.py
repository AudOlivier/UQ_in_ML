import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.tri as mtri
#import matplotlib.cm as cm
#from matplotlib.colors import Normalize
#from matplotlib.collections import PolyCollection
#from matplotlib.patches import Ellipse, Rectangle
from sklearn.metrics.pairwise import euclidean_distances
import csv
import pickle
from scipy.interpolate import interp1d
from scipy.linalg import block_diag

perc_radius_safe = 0.3333
corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

#############################        GENERATE MICROSTRUCTURES          ######################################

def generate_circular_fibers_square_packed(GVf, n_inclusions):

    """ generate square packed circular inclusions in a 2D [0 1] square domain """

    n_circles_per_row = np.sqrt(n_inclusions)
    if (n_circles_per_row - np.floor(n_circles_per_row)) != 0:
        raise ValueError('n_inclusions must be a perfect square for square packed')
    n_circles_per_row = int(n_circles_per_row)
    areaRVE = 1 * 1
    radius_inclusion = np.sqrt(areaRVE * GVf / n_inclusions / np.pi)
    length_per_inclusion = 1 / n_circles_per_row
    centers = np.zeros((n_inclusions, 2))
    count_inclusions = 0
    for i in range(n_circles_per_row):
        for j in range(n_circles_per_row):
            centers[count_inclusions, :] = [j * length_per_inclusion + length_per_inclusion / 2,
                                            i * length_per_inclusion + length_per_inclusion / 2]
            count_inclusions += 1
    return radius_inclusion, centers


def generate_circular_fibers_random(GVf, n_inclusions, n_per_cluster=1):
    """ generate random circular inclusions in a 2D [0 1] square domain """
    if n_inclusions % n_per_cluster != 0:
        raise ValueError('Audrey :( n_inclusions must be a multiple of n_per_cluster')
    areaRVE = 1
    radius_inclusion = np.sqrt(areaRVE * GVf / n_inclusions / np.pi)
    n_spheres = int(n_inclusions/n_per_cluster)
    # first sample the spheres in which the fibers will lie
    if n_per_cluster == 1:
        radius_spheres = 1 * radius_inclusion
    elif n_per_cluster == 3:
        radius_spheres = np.sqrt(n_per_cluster/0.25) * radius_inclusion #volume fraction of 25% in the sphere
    elif n_per_cluster == 5:
        radius_spheres = np.sqrt(n_per_cluster/0.35) * radius_inclusion
    else:
        raise ValueError('Audrey :( only n_per_cluster = 1, 3, 5 supported for now')

    npm = 0
    point_in_matrix_not_found = True
    while point_in_matrix_not_found:
        # first iteration: ensure that this one is strictly inside the domain
        n_iter = 0
        count_inclusions = n_per_cluster
        center_sphere = np.random.uniform(low=1.1 * radius_spheres, high=1 - 1.1 * radius_spheres, size=(1, 2))
        centers = create_centers_from_spheres(center_sphere, n_per_cluster, radius_spheres, radius_inclusion)

        while count_inclusions < n_inclusions:
            n_iter += 1
            if n_iter >= n_spheres * 490:
                print('n_iter = {}'.format(n_iter))
            if n_iter >= n_spheres * 500:
                n_iter = 0
                count_inclusions = n_per_cluster
                center_sphere = np.random.uniform(low=1.1 * radius_spheres, high=1 - 1.1 * radius_spheres, size=(1, 2))
                centers = create_centers_from_spheres(center_sphere, n_per_cluster, radius_spheres, radius_inclusion)
                continue

            center_sphere = np.random.uniform(low=0, high=1, size=(1, 2))
            new_centers = create_centers_from_spheres(center_sphere, n_per_cluster, radius_spheres, radius_inclusion)
            mirrors = mirror_fiber_v2(new_centers, centers, radius_inclusion)
            new_centers = np.vstack((new_centers, mirrors))
            if check_fiber_to_others(new_centers, centers, radius_inclusion) and check_fiber_to_sides(new_centers, radius_inclusion):
                count_inclusions += n_per_cluster
                centers = np.concatenate((centers, new_centers, mirrors), axis=0)
            #if check_fiber_to_sides(new_centers, radius_inclusion):
            #    if check_fiber_to_others(new_centers, centers, radius_inclusion):
            #        b, mirrors = mirror_fiber(new_centers, centers, radius_inclusion)
            #        if b:
            #            count_inclusions += n_per_cluster
            #            centers = np.concatenate((centers, new_centers, mirrors), axis=0)
        
            # at the end, remove all those for which the center is more than a radius away from a border
            id_to_remove = []
            for id_center in range(centers.shape[0]):
                if (centers[id_center,0] > 1+radius_inclusion) or (centers[id_center,0] < (-1)*radius_inclusion) or (
                    centers[id_center,1] > 1+radius_inclusion) or (centers[id_center,1] < (-1)*radius_inclusion):
                    id_to_remove.append(id_center)
            centers = np.delete(centers, id_to_remove, axis=0)
            # find point in matrix
            domain = np.linspace(0.05, 0.95, 181)
            xv, yv = np.meshgrid(domain, domain, sparse=False, indexing='ij')
            X = np.concatenate((np.reshape(xv, (-1,1)), np.reshape(yv, (-1,1))), axis=1)
            pairwise_distances = euclidean_distances(X, centers)
            min_d_to_fiber = np.min(pairwise_distances, axis=1)
            ind_max = np.argmax(min_d_to_fiber)
            if min_d_to_fiber[ind_max] >= 2*radius_inclusion:
                point_in_matrix = X[ind_max,:]
                point_in_matrix_not_found = False
            else:
                npm += 1
                if npm > 490:
                    print('npm = {}'.format(npm))
                continue
            if npm > 500:
                radius_inclusion, centers, point_in_matrix = None, None, None
                point_in_matrix_not_found = False
    return radius_inclusion, centers, point_in_matrix


def create_centers_from_spheres(center_sphere, n_per_cluster, R, r):
    if n_per_cluster == 1:
        cs = center_sphere
    else:
        n_tmp = 0
        phis = 2 * np.pi * np.random.rand(n_per_cluster, 1)
        rs = (R-r) * np.random.rand(n_per_cluster, 1)
        xs, ys = np.cos(phis) * rs, np.sin(phis) * rs
        while not check_fiber_to_self(np.concatenate([xs, ys], axis=1), r):
            if n_tmp >= 490:
                print('n_temp = {}'.format(n_tmp))
            if n_tmp >= 500:
                n_tmp = 0
                phis = 2 * np.pi * np.random.rand(n_per_cluster, 1)
                rs = (R-r) * np.random.rand(n_per_cluster, 1)
                xs, ys = np.cos(phis) * rs, np.sin(phis) * rs
                continue
            phis = 2 * np.pi * np.random.rand(n_per_cluster, 1)
            rs = (R-r) * np.random.rand(n_per_cluster, 1)
            xs, ys = np.cos(phis) * rs, np.sin(phis) * rs
            n_tmp += 1
        cs = np.concatenate([center_sphere[:,0]+xs, center_sphere[:,1]+ys], axis=1)
    return cs


def check_fiber_to_sides(new_centers, r):
    """ the distance between the fibers and the sides should not be less than dist_min = perc_radius_safe * r"""
    mask = [True if (c[0]<1 and c[0]>0 and c[1]<1 and c[1]>0) else False for c in new_centers]
    new_centers_in = new_centers[mask,:]
    dist_to_sides = np.vstack([np.minimum(new_centers_in[:, 0], 1 - new_centers_in[:, 0]).reshape((-1,1)),
                               np.minimum(new_centers_in[:, 1], 1 - new_centers_in[:, 1]).reshape((-1,1))])
    if any(((d >= r) and (d < (1 + perc_radius_safe) * r)) for d in dist_to_sides):
        return False
    if any(((d <= r) and (d >= 0.5 * r)) for d in dist_to_sides):
        return False
    #if any((d <= r) and (d <= 0.4 * r) for d in dist_to_sides):
    #    return False
    return True


def check_fiber_to_others(center_proposed, centers_exist, r):
    """ the distance between fibers should not be less than dist_min"""
    dist_to_others = euclidean_distances(center_proposed, centers_exist).reshape((-1,1))
    if any(d <= (2 + perc_radius_safe) * r for d in dist_to_others):
        return False
    return True


def check_fiber_to_self(center_proposed, r):
    """ the distance between fibers should not be less than dist_min"""
    dist_to_others = euclidean_distances(center_proposed, center_proposed)
    iu1 = np.triu_indices(center_proposed.shape[0], k=1)
    dist_to_others = dist_to_others[iu1]
    if any(d <= (2 + perc_radius_safe) * r for d in dist_to_others):
        return False
    return True


def mirror_fiber_v2(center_proposed, centers_exist, r):
    """ mirror the fiber to the right/left/bottom/top/all corners if possible"""
    mirrored = np.empty((0, 2), float)
    # left side
    if any(c <= r for c in center_proposed[:,0]):
        mirr = np.concatenate(((1 + center_proposed[:, 0]).reshape((-1,1)),
                               center_proposed[:, 1].reshape((-1,1))), axis=1)
        mirrored = np.concatenate((mirrored, mirr), axis=0)
    # right side
    if any(c >= 1-r for c in center_proposed[:,0]):
        mirr = np.concatenate(((center_proposed[:, 0] - 1).reshape((-1,1)),
                              center_proposed[:, 1].reshape((-1,1))), axis=1)
        mirrored = np.concatenate((mirrored, mirr), axis=0)
    # bottom side
    if any(c <= r for c in center_proposed[:,1]):
        mirr = np.concatenate((center_proposed[:, 0].reshape((-1,1)),
                               (1 + center_proposed[:, 1]).reshape((-1,1))), axis=1)
        mirrored = np.concatenate((mirrored, mirr), axis=0)
    # top side
    if any(c >= 1-r for c in center_proposed[:,1]):
        mirr = np.concatenate((center_proposed[:, 0].reshape((-1,1)),
                               (center_proposed[:, 1] - 1).reshape((-1,1))), axis=1)
        mirrored = np.concatenate((mirrored, mirr), axis=0)
    # special case: the corners
    dist_to_corners = euclidean_distances(center_proposed, corners).reshape((-1,1))
    if any(d <= r for d in dist_to_corners):
        mirr = np.concatenate(((center_proposed[:, 0] - 1).reshape((-1,1)),
                               (center_proposed[:, 1] - 1).reshape((-1,1))), axis=1)
        for j in range(mirr.shape[0]):
            mirr[j, center_proposed[j, :] < 0.5] = 1 + center_proposed[j, center_proposed[j, :] < 0.5]
        mirrored = np.concatenate((mirrored, mirr), axis=0)
    return mirrored


def mirror_fiber(center_proposed, centers_exist, r):
    """ mirror the fiber to the right/left/bottom/top/all corners if possible"""
    mirrored = np.empty((0, 2), float)
    # left side
    if any(c <= r for c in center_proposed[:,0]):
        mirr = np.concatenate(((1 + center_proposed[:, 0]).reshape((-1,1)),
                               center_proposed[:, 1].reshape((-1,1))), axis=1)
        if check_fiber_to_others(mirr, centers_exist, r) and check_fiber_to_sides(mirr, r):
            mirrored = np.concatenate((mirrored, mirr), axis=0)
        else:
            return False, None
    # right side
    if any(c >= 1-r for c in center_proposed[:,0]):
        mirr = np.concatenate(((center_proposed[:, 0] - 1).reshape((-1,1)),
                              center_proposed[:, 1].reshape((-1,1))), axis=1)
        if check_fiber_to_others(mirr, centers_exist, r) and check_fiber_to_sides(mirr, r):
            mirrored = np.concatenate((mirrored, mirr), axis=0)
        else:
            return False, None
    # bottom side
    if any(c <= r for c in center_proposed[:,1]):
        mirr = np.concatenate((center_proposed[:, 0].reshape((-1,1)),
                               (1 + center_proposed[:, 1]).reshape((-1,1))), axis=1)
        if check_fiber_to_others(mirr, centers_exist, r) and check_fiber_to_sides(mirr, r):
            mirrored = np.concatenate((mirrored, mirr), axis=0)
        else:
            return False, None
    # top side
    if any(c >= 1-r for c in center_proposed[:,1]):
        mirr = np.concatenate((center_proposed[:, 0].reshape((-1,1)),
                               (center_proposed[:, 1] - 1).reshape((-1,1))), axis=1)
        if check_fiber_to_others(mirr, centers_exist, r) and check_fiber_to_sides(mirr, r):
            mirrored = np.concatenate((mirrored, mirr), axis=0)
        else:
            return False, None
    # special case: the corners
    dist_to_corners = euclidean_distances(center_proposed, corners).reshape((-1,1))
    if any(d <= r for d in dist_to_corners):
        mirr = np.concatenate(((center_proposed[:, 0] - 1).reshape((-1,1)),
                               (center_proposed[:, 1] - 1).reshape((-1,1))), axis=1)
        for j in range(mirr.shape[0]):
            mirr[j, center_proposed[j, :] < 0.5] = 1 + center_proposed[j, center_proposed[j, :] < 0.5]
        if check_fiber_to_others(mirr, centers_exist, r) and check_fiber_to_sides(mirr, r):
            mirrored = np.concatenate((mirrored, mirr), axis=0)
        else:
            return False, None
    return True, mirrored


###########################   POST PROCESSING   ###########################


def read_output_data(job_nb, path_IO, ext=''):
    """ Read output data (.pkl files) created by the Abaqus/python runs
    Possible extensions are: '', '_average', 'centroid' """
    file_name = path_IO+'JOB-'+str(job_nb)+ext
    with open(file_name + '.pkl', 'rb') as f:
        outputsDict = pickle.load(f, encoding='latin1')
    if 'el_connectivity' in outputsDict.keys():
        n_elements, n_nodes_per_el = outputsDict['el_connectivity'].shape
        n_nodes, n_frames, _ = outputsDict['node_U'].shape
        outputsDict.update({'n_el': n_elements, 'n_nodes_per_el': n_nodes_per_el, 'n_nodes': n_nodes, 'n_frames': n_frames})
        return outputsDict


def read_input_data(job_nb, path_IO, extension_file):
    inputsDict = {}
    with open(path_IO+'input_fixed_dict'+extension_file+'.csv') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            inputsDict.update(dict(row))
    with open(path_IO+'input_variable_dict'+extension_file+'.csv') as csv_file:
        reader = csv.DictReader(csv_file)
        for row_number, row in enumerate(reader):
            if row_number == job_nb:
                inputsDict.update(dict(row))
                break
    fibers_center = inputsDict['fibers_center'][2:-2].split('), (')
    fibers_center = [tuple(map(float, stuff.split(', '))) for stuff in fibers_center]
    inputsDict['fibers_center'] = np.array(fibers_center)
    inputsDict['fibers_radius'] = float(inputsDict['fibers_radius'])
    point_in_matrix = inputsDict['point_in_matrix'][1:-1].split(', ')
    inputsDict['point_in_matrix'] = (float(point_in_matrix[0]), float(point_in_matrix[1]))
    return inputsDict


def compute_VonMises(fieldX, fieldY, fieldZ, fieldXY):
    fieldX_dev = fieldX - 1/3*(fieldX+fieldY+fieldZ)
    fieldY_dev = fieldY - 1/3*(fieldX+fieldY+fieldZ)
    fieldZ_dev = fieldZ - 1/3*(fieldX+fieldY+fieldZ)
    tmp = fieldX_dev**2+fieldY_dev**2+fieldZ_dev**2+2*(fieldXY**2)
    field_VonMises = np.sqrt(3/2*tmp)
    #tmp = (fieldX-fieldY)**2+(fieldX-fieldZ)**2+(fieldZ-fieldY)**2+6*fieldXY**2
    #field_VonMises = np.sqrt(1/2*tmp)
    return field_VonMises

    
def compute_PEEQ(fieldX, fieldY, fieldZ, fieldXY):
    fieldX_dev = fieldX - 1/3*(fieldX+fieldY+fieldZ)
    fieldY_dev = fieldY - 1/3*(fieldX+fieldY+fieldZ)
    fieldZ_dev = fieldZ - 1/3*(fieldX+fieldY+fieldZ)
    tmp = fieldX_dev**2+fieldY_dev**2+fieldZ_dev**2+2*(fieldXY**2)
    field_PEEQ = np.sqrt(2/3*tmp)
    return field_PEEQ


def compute_E_nu_plane_strain(epsX, epsY, epsXY, sigX, sigY, sigXY):
    print(epsY)
    C1 = epsXY / sigXY  # C1 = (1+nu)/E
    nu = (C1 * sigX - epsX) / (C1 * (sigX + sigY))
    E = (1+nu) / C1
    return E, nu

def power_law(x, a, b, c):
    return (a + b * (x ** c))*10**6

def plot_avXY(fieldX, fieldY, field_av, ax, label='', color=None, return_av_fields=False, linestyle='-',
              fit_power_law = False):
    avX = compute_av_field(fieldX, field_av)
    avY = compute_av_field(fieldY, field_av)
    if color is None:
        ax.plot(avX, avY, label=label, linestyle=linestyle)
    else:
        ax.plot(avX, avY, label=label, color=color, linestyle=linestyle)
    if fit_power_law:
        from scipy.optimize import curve_fit
        pfit, pcov = curve_fit(power_law, avX, avY)
        return avX, avY, pfit, pcov
    else:
        return avX, avY

def compute_av_field(fieldX, field_av):
    nFrames = fieldX.shape[1]
    # save the average strains and stresses, for all frames
    av_field = np.zeros((nFrames, ))
    #el_mises = np.zeros((n_frames, numElements))
    for fr in range(nFrames):
        vol_sum = np.sum(field_av[:,fr])
        vol_sum_fieldX = [vol_el*field for (vol_el, field) in zip(field_av[:,fr], fieldX[:,fr])]
        av_field[fr] = sum(vol_sum_fieldX)/vol_sum
    return av_field

def compute_av_std_X(fieldX, field_av):
    vol_sum = np.sum(field_av)
    vol_av_field = np.sum(fieldX*field_av)/vol_sum
    vol_std_field = np.sqrt(np.sum(field_av*(fieldX-vol_av_field)**2)/vol_sum)
    return vol_av_field, vol_std_field

def compute_vol_fraction_yield(peeq, field_av):
    v_yield = [v for (i, v) in enumerate(field_av) if peeq[i]>10**(-3)]
    return sum(v_yield)/sum(field_av)

def plot_evolution_qty(qty, fieldX, field_av, ax, time_steps=None,
                       color='', linestyle='-', title='', label=''):
    nFrames = fieldX.shape[1]
    qty_to_plot = np.zeros((nFrames,))
    if qty.lower()=='yield_volume_ratio':
        for fr in range(nFrames):
            qty_to_plot[fr] = compute_vol_fraction_yield(fieldX[:,fr], field_av[:,fr])
    elif qty.lower()=='std_over_domain':
        for fr in range(nFrames):
            _, qty_to_plot[fr] = compute_av_std_X(fieldX[:,fr], field_av[:,fr])
    if time_steps is None:
        time_steps = np.arange(1, nFrames+1)
        time_steps = time_steps/nFrames
    ax.plot(time_steps, qty_to_plot, color=color, linestyle=linestyle, label=label)
    ax.set_title(title)
    ax.set_xlabel('frames')             


def extract_input_vector(list_jobs, **kwargs):
    # extract the feature vector for a set of jobs
    rvs_key = ['vf', 'n_inclusions', 'matrix_properties']
    rvs = []
    for job_nb in list_jobs:
        inputsDict, _, _, _ = read_input_data(job_nb, kwargs['path_IO'], kwargs['extension_file'])
        rvi = [inputsDict[key] for key in rvs_key]
        rvi[2] = rvi[2][2:-2].split(',')
        rvi = [float(rvi[0]), float(rvi[1]), float(rvi[2][2]), float(rvi[2][3]), float(rvi[2][4])]
        rvs.append(rvi)
    return np.array(rvs)
    

from scipy.optimize import curve_fit
def extract_output_vector(list_jobs, **kwargs):    
    # extract the outputs for a given list of jobs
    params_constitutive_law = []
    params_constitutive_law_cov = []
    perc_above_1 = []
    perc_above_1p2 = []
    for job_nb in list_jobs:
        try:
            vol_av = read_output_data(job_nb, kwargs['path_IO'], ext='_outputs_volume_averages')

            # Fit parameters to Mises vs LE/PEEQ laws
            array_von_mises = compute_VonMises(vol_av['av_S'][:,0], vol_av['av_S'][:,1], vol_av['av_S'][:,2], 
                                               vol_av['av_S'][:,3])
            array_peeq = compute_PEEQ(vol_av['av_PE'][:,0], vol_av['av_PE'][:,1], vol_av['av_PE'][:,2], 
                                      vol_av['av_PE'][:,3])
            array_le = compute_PEEQ(vol_av['av_LE'][:,0], vol_av['av_LE'][:,1], vol_av['av_LE'][:,2], 
                                    vol_av['av_LE'][:,3])
            mask = array_peeq > 1e-3
            pfit, pcov = curve_fit(power_law, array_peeq[mask], array_von_mises[mask], bounds=(0, np.inf))
            #pfit, pcov = curve_fit(lambda x, c: 0.4*10**9+0.5*10**9*x**c, 
            #                       array_peeq[mask], array_von_mises[mask], bounds=(0, np.inf))
            
            # E by looking at the le_strain_equivalent vs mises curve
            mask = array_peeq <= 1e-3
            pfit_2, pcov_2 = curve_fit(lambda x, a: a*x*10**9, array_le[mask], array_von_mises[mask], bounds=(0, np.inf))
            
            # E by looking at the ee_1 vs s_1 curve
            mask = array_peeq <= 1e-3
            ee_1 = vol_av['av_EE'][mask,0]
            s_1 = vol_av['av_S'][mask,0]
            pfit_3, pcov_3 = curve_fit(lambda x, a: a*x*10**9, ee_1, s_1, bounds=(0,np.inf))
            
            params_constitutive_law.append(np.concatenate([pfit, pfit_2, pfit_3]))
            params_constitutive_law_cov.append(block_diag(pcov, pcov_2, pcov_3))

            # S1, E1 -> nope, does not make sense here
            #mask = (vol_av['av_PE'][:,0] > 1e-3)
            #pfit, pcov = curve_fit(power_law, vol_av['av_PE'][mask,0], vol_av['av_S'][mask,0], bounds=(0, np.inf))
            #pfit_s1_e1.append(pfit)
            #pcov_s1_e1.append(pcov)

            # Mises > 2 thresholds
            outputsDict, nNodes, nElements, nNodesPerElement, nFrames, _ = read_output_data(job_nb, kwargs['path_IO'], 
                                                                                        ext='_outputs_mesh')
            at_centroids = read_output_data(job_nb, kwargs['path_IO'], ext='_outputs_centroid')
            list_fibers = np.setdiff1d(list(range(nElements)),outputsDict['el_in_matrix'], assume_unique=True)
            mises_in_fibers = at_centroids['el_Mises'][list_fibers,-1]
            vol_fibers = np.sum(at_centroids['el_Vol'][list_fibers,-1])
            mises_above_1 = [ind for (i, ind) in enumerate(list_fibers) if mises_in_fibers[i]>1*10**9 ]
            mises_above_1p2 = [ind for (i, ind) in enumerate(list_fibers) if mises_in_fibers[i]>1.2*10**9] 
            perc_above_1.append(np.sum(at_centroids['el_Vol'][mises_above_1,-1])/vol_fibers)
            perc_above_1p2.append(np.sum(at_centroids['el_Vol'][mises_above_1p2,-1])/vol_fibers)
            
            # Toughness: 50% of the matrix has 
            # le1_in_matrix = at_centroids['el_LE'][outputsDict['el_in_matrix'], -1, 0]
            
        except:
            print('Pbm with job nb {}'.format(job_nb))
    return np.array(params_constitutive_law), np.array(params_constitutive_law_cov), np.array(perc_above_1), np.array(perc_above_1p2)

def extract_input_vector_EMI(list_jobs, **kwargs):
    # extract the feature vector for a set of jobs
    rvs_key = ['vf', 'matrix_properties']
    rvs = []
    for job_nb in list_jobs:
        inputsDict, _, _, _ = read_input_data(job_nb, kwargs['path_IO'], kwargs['extension_file'])
        rvi = [inputsDict[key] for key in rvs_key]
        rvi[1] = rvi[1][2:-2].split(',')
        rvi = [float(rvi[0]), float(rvi[1][4])]
        rvs.append(rvi)
    # return only vf and matrix_c
    return np.array(rvs)

from scipy.optimize import curve_fit
def extract_output_vector_EMI(list_jobs, **kwargs):    
    # extract the outputs for a given list of jobs
    outputs = []
    params_constitutive_law_cov = []
    for job_nb in list_jobs:
        try:
            vol_av = read_output_data(job_nb, kwargs['path_IO'], ext='_outputs_volume_averages')

            # Fit parameters to Mises vs LE/PEEQ laws
            array_von_mises = compute_VonMises(vol_av['av_S'][:,0], vol_av['av_S'][:,1], vol_av['av_S'][:,2], 
                                               vol_av['av_S'][:,3])
            array_peeq = compute_PEEQ(vol_av['av_PE'][:,0], vol_av['av_PE'][:,1], vol_av['av_PE'][:,2], 
                                      vol_av['av_PE'][:,3])
            mask = array_peeq > 1e-3
            pfit, pcov = curve_fit(lambda x, a, b, c: a*10**6+b*10**6*x**c,
                                   array_peeq[mask], array_von_mises[mask], bounds=(0, np.inf))
            pfit, pcov = pfit[2], pcov[2, 2]
            #pfit, pcov = curve_fit(lambda x, c: 400*10**6+500*10**6*x**c, 
            #                       array_peeq[mask], array_von_mises[mask], bounds=(0, np.inf))
            
            # E by looking at the ee_1 vs s_1 curve
            mask = array_peeq <= 1e-3
            ee_1 = vol_av['av_EE'][mask,0]
            s_1 = vol_av['av_S'][mask,0]
            pfit_3, pcov_3 = curve_fit(lambda x, a: a*x*10**11, ee_1, s_1, bounds=(0,np.inf))

            # Mises > threshold
            outputsDict, nNodes, nElements, nNodesPerElement, nFrames, _ = read_output_data(job_nb, kwargs['path_IO'], 
                                                                                        ext='_outputs_mesh')
            at_centroids = read_output_data(job_nb, kwargs['path_IO'], ext='_outputs_centroid')
            list_fibers = np.setdiff1d(list(range(nElements)),outputsDict['el_in_matrix'], assume_unique=True)
            mises_in_fibers = at_centroids['el_Mises'][list_fibers,-1]
            vol_fibers = np.sum(at_centroids['el_Vol'][list_fibers,-1])
            mises_above_1 = [ind for (i, ind) in enumerate(list_fibers) if mises_in_fibers[i]>1*10**9]
            perc_above_1 = np.sum(at_centroids['el_Vol'][mises_above_1,-1])/vol_fibers
            
            outputs.append([pfit_3, pfit, perc_above_1])
            params_constitutive_law_cov.append(block_diag(pcov_3, pcov))
        except:
            print('Pbm with job nb {}'.format(job_nb))
    return np.array(outputs), np.array(params_constitutive_law_cov)                         


##############################          PLOTS          ############################## 

def plot_section(radius_inclusion, centers, ax, point_matrix=None):
    #an = np.linspace(0, 2 * np.pi, 100)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color='black', linewidth=0.5)
    for i in range(centers.shape[0]):
        ellipse = Ellipse(centers[i, :], 2 * radius_inclusion, 2 * radius_inclusion, angle=0)
        ax.add_patch(ellipse)
        # ax.plot(centers[i,0]+radius_inclusion*np.cos(an), centers[i,1]+radius_inclusion*np.sin(an),color='r')
    ax.axis('equal')
    #ax.axis([0, 1, 0, 1])
    plt.gca().set_adjustable("box")
    if point_matrix is not None:
        ax.plot(point_matrix[0], point_matrix[1], color='red', marker='+')
    return ax

def plot_section_v2(radius_inclusion, centers, ax, point_matrix=None, colors=None, alphas=None):
    #an = np.linspace(0, 2 * np.pi, 100)
    if colors is None:
        colors = ['orange', 'blue']
    if alphas is None:
        alphas = [0.5, 1.]
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color='black', linewidth=0.5)
    ax.add_patch(Rectangle(xy=[0., 0.], width=1., height=1., color=colors[0], alpha=alphas[0]))
    for i in range(centers.shape[0]):
        ellipse = Ellipse(centers[i, :], 2 * radius_inclusion, 2 * radius_inclusion, angle=0, color=colors[1], alpha=alphas[1])
        ax.add_patch(ellipse)
        # ax.plot(centers[i,0]+radius_inclusion*np.cos(an), centers[i,1]+radius_inclusion*np.sin(an),color='r')
    ax.axis('equal')
    ax.axis([0, 1, 0, 1])
    #plt.gca().set_adjustable("box")
    if point_matrix is not None:
        ax.plot(point_matrix[0], point_matrix[1], color='red', marker='+')
    return ax

def plot_section_repeated(radius_inclusion, centers):
    an = np.linspace(0, 2 * np.pi, 100)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color='black')
    ax.plot([1, 2, 2, 1, 1], [0, 0, 1, 1, 0], color='black')
    ax.plot([1, 2, 2, 1, 1], [1, 1, 2, 2, 1], color='black')
    ax.plot([0, 1, 1, 0, 0], [1, 1, 2, 2, 1], color='black')
    centers_right = np.array(centers, copy=True)
    centers_right[:, 0] = centers_right[:, 0] + 1
    centers_up = np.array(centers, copy=True)
    centers_up[:, 1] = centers_up[:, 1] + 1
    centers_right_up = np.array(centers_right, copy=True)
    centers_right_up[:, 1] = centers_right_up[:, 1] + 1
    centers_4 = np.concatenate((centers, centers_right, centers_up, centers_right_up), axis=0)
    for i in range(centers_4.shape[0]):
        ellipse = Ellipse(centers_4[i, :], 2 * radius_inclusion, 2 * radius_inclusion, angle=0)
        ax.add_patch(ellipse)
        # ax.plot(centers[i,0]+radius_inclusion*np.cos(an), centers[i,1]+radius_inclusion*np.sin(an),color='r')
    ax.axis('equal')
    ax.axis([0, 2, 0, 2])
    plt.gca().set_adjustable("box")
    return fig, ax

def plot_mesh(coord, connectivity, ax = None):
    # compute triangulation
    if connectivity.shape[1] in [3, 6]:
        triangulation = mtri.Triangulation(x=coord[:,0], y=coord[:,1], triangles=connectivity[:,0:3])
        # plot
        if ax is not None:
            ax.triplot(triangulation, 'b-')
            ax.axis('equal')
        return None
    elif connectivity.shape[1] in [4, 8]:
        #triangles1 = np.array([connect_e[[0, 1, 2]] for connect_e in connectivity])
        #triangles2 = np.array([connect_e[[2, 3, 0]] for connect_e in connectivity])
        #triangles = np.concatenate([triangles1, triangles2], axis=0)
        #triangulation = mtri.Triangulation(x=coord[:,0], y=coord[:,1], triangles=triangles)
        # plot
        if ax is not None:
            pc = PolyCollection(verts=coord[connectivity[:,0:4],:], facecolors='white', edgecolors='black')
            ax.add_collection(pc)
        return None

def plot_field(field, coord, ax, connectivity=None, cbarlabel=" ", cmap='Reds', cbar_range=None,
               plot_mesh = True, centers=None, radius=None):
    edgecolors = 'grey'
    if not plot_mesh:
        edgecolors = 'none'
        if centers is not None:
            for i in range(centers.shape[0]):
                ellipse = Ellipse(centers[i, :], 2 * radius, 2 * radius, angle=0, edgecolor='black', facecolor='none',
                                  alpha=0.5)
                ax.add_patch(ellipse)
    # 
    if cbar_range is None:
        cbar_range = (field.min()-0.001, field.max()+0.001)
    # plot only those elements that are inside the range
    mask_el = [(f > cbar_range[0] and f < cbar_range[1]) for f in field]
    not_mask_el = [not m for m in mask_el]
    # if field is given for all elements
    if (connectivity is not None) and (field.shape[0] == connectivity.shape[0]):
        # plot a field
        if connectivity.shape[1] in [3, 6]:
            triangulation = mtri.Triangulation(x=coord[:,0], y=coord[:,1], triangles=connectivity[:,0:3])
            t = ax.tripcolor(triangulation[mask_el], facecolors=field[mask_el], edgecolors=edgecolors, cmap=cmap)
        elif connectivity.shape[1] in [4, 8]:
            norm = Normalize(vmin=cbar_range[0], vmax=cbar_range[1])
            t = cm.ScalarMappable(cmap=cmap, norm=norm)
            pc = PolyCollection(verts=coord[connectivity[mask_el,0:4],:], edgecolors=edgecolors, 
                                facecolors=t.to_rgba(field[mask_el]))
            ax.add_collection(pc)
            pc2 = PolyCollection(verts=coord[connectivity[not_mask_el,0:4],:], edgecolors=edgecolors, 
                                 facecolor='black')
            ax.add_collection(pc2)
            pc.set_clim(cbar_range[0], cbar_range[1])
            t.set_array(field[mask_el])
    # if field is given at nodal points
    elif field.shape[0] == coord.shape[0]:
        t = ax.tripcolor(coord[:,0], coord[:,1], field, 20, cmap=cmap)
    else:
        raise ValueError('field should have n_elements or n_nodes components')
    print(cbar_range)
    cbar = plt.colorbar(t, ax=ax, ticks=np.linspace(cbar_range[0], cbar_range[1], 10), cmap=cmap)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.axis('equal')
    return None
