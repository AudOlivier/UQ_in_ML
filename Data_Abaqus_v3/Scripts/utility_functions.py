import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.collections import PolyCollection
from matplotlib.patches import Ellipse, Rectangle
import csv
import pickle



###########################   READ DATA   ###########################


import warnings
def read_output_data(job_nb, path_IO, ext='_outputs_mesh'):
    """ 
    Read output data (.pkl files) created by the Abaqus/python runs
    Possible extensions are: '_outputs_mesh', '_outputs_volume_averages', 'outputs_centroid'
    """
    file_name = path_IO+'JOB-'+str(job_nb)+ext
    with open(file_name + '.pkl', 'rb') as f:
        outputsDict = pickle.load(f, encoding='latin1')
    if ext == '_outputs_mesh':
        # check the warnings
        if outputsDict['warning'][0] != 'Okay' or outputsDict['warning'][1] != 'Okay':
            warnings.warn('The PBC seem wrong for job nb {} - check results carefully'.format(job_nb))
        n_elements, n_nodes_per_el = outputsDict['el_connectivity'].shape
        n_nodes, n_frames, _ = outputsDict['node_U'].shape
        outputsDict.update({'n_el': n_elements, 'n_nodes_per_el': n_nodes_per_el, 'n_nodes': n_nodes, 'n_frames': n_frames})
    return outputsDict


def read_input_data(job_nb, path_IO, extension_file):
    """ 
    Read input data from two csv files
    """
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
    # Process from strings to floats or ints
    for key in ['n_inclusions', 'n_per_cluster']:
        inputsDict[key] = int(inputsDict[key])
    for key in ['vf', 'mesh_size', 'dt']:
        inputsDict[key] = float(inputsDict[key])
    for strain_ in ['strain_x', 'strain_y']:
        if strain_ in inputsDict.keys() and inputsDict[strain_] != 'None':
            inputsDict[strain_] = float(inputsDict[strain_])
        else:
            inputsDict[strain_] = None
    fibers_center = inputsDict['fibers_center'][2:-2].split('), (')
    inputsDict['fibers_center'] = [tuple(map(float, stuff.split(', '))) for stuff in fibers_center]
    inputsDict['fibers_radius'] = float(inputsDict['fibers_radius'])
    point_matrix = inputsDict['point_in_matrix'][1:-1].split(', ')
    inputsDict['point_in_matrix'] = (float(point_matrix[0]), float(point_matrix[1]))
    for domain in ['MATRIX', 'FIBERS']:
        inputsDict[domain.lower()+'_properties'] = tuple(map(float, inputsDict[domain.lower()+'_properties'][1:-1].split(',')))
    return inputsDict

############################     CORRELATION AND COVARIANCE     ###########################

def cov_to_corr(cov):
    Dinv = np.diag(1/np.sqrt(np.diag(cov)))
    return np.matmul(np.matmul(Dinv, cov), Dinv)


def corr_to_cov(corr, sigmas):
    return np.matmul(np.matmul(np.diag(sigmas), corr), np.diag(sigmas))


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


###########################        DATA PROCESSING          ############################## 


def extract_input_vars(job_nb, path_IO, extension_file, n_inputs):
    """ Given a job nb, read the input files and return the vector of inputs for learning """
    inputsDict = read_input_data(job_nb, path_IO, extension_file)
    vf = inputsDict['vf']
    Ef = inputsDict['fibers_properties'][0] / 1e9
    b, c = inputsDict['matrix_properties'][3] / 1e6, inputsDict['matrix_properties'][4]
    if n_inputs == 5:
        clustering = inputsDict['n_per_cluster']
        return [vf, Ef, b, c, clustering]
    return [vf, Ef, b, c]


def extract_output_vars(job_nb, path_IO, return_nu=False, threshold_perc_ac_yield=0.01, return_pcov=False, threshold_mises=1e3):
    """ Given a job nb, read the raw output data and return the vector of outputs for learning """
    dict_mesh = read_output_data(job_nb, path_IO, ext='_outputs_mesh')
    dict_2 = read_output_data(job_nb, path_IO, ext='_outputs_volume_averages')
    E, nu, a, b, c = fit_behavior_law_to_field_data(
        dict_mesh=dict_mesh, dict_av=dict_2, threshold_perc_ac_yield=threshold_perc_ac_yield, return_pcov=return_pcov)
    outputs_list = [a, b, c, E]
    if return_nu:
        outputs_list.append(nu)
    dict_2 = read_output_data(job_nb, path_IO, ext='_outputs_centroid')
    perc_above = compute_perc_Mises_above_threshold(dict_mesh=dict_mesh, dict_centroids=dict_2, threshold_mises=threshold_mises)
    if isinstance(perc_above, list):
        [outputs_list.append(p) for p in perc_above]
    else:
        outputs_list.append(perc_above)
    return outputs_list


def extract_output_vars2(job_nb, path_IO, return_nu=False, a_fixed=400., return_pcov=False, threshold_mises=1e3):
    """ Given a job nb, read the raw output data and return the vector of outputs for learning """
    dict_mesh = read_output_data(job_nb, path_IO, ext='_outputs_mesh')
    dict_2 = read_output_data(job_nb, path_IO, ext='_outputs_volume_averages')
    E, nu, b, c = fit_behavior_law_to_field_data2(
        dict_mesh=dict_mesh, dict_av=dict_2, a_fixed=a_fixed, return_pcov=return_pcov)
    outputs_list = [b, c, E]
    if return_nu:
        outputs_list.append(nu)
    dict_2 = read_output_data(job_nb, path_IO, ext='_outputs_centroid')
    perc_above = compute_perc_Mises_above_threshold(dict_mesh=dict_mesh, dict_centroids=dict_2, threshold_mises=threshold_mises)
    if isinstance(perc_above, list):
        [outputs_list.append(p) for p in perc_above]
    else:
        outputs_list.append(perc_above)
    return outputs_list


def fit_behavior_law_to_field_data(dict_mesh, dict_av, threshold_perc_ac_yield=0.01, return_pcov=False):
    """ Fit parameters E, nu, a, b and c to data (field data, needs to do volume averaging)
    a is being learnt along with parameters b and c in plasticity law """
    ind_sep = separate_el_pl(dict_mesh, threshold_perc_ac_yield=threshold_perc_ac_yield)
    # Linear elastic part: compute E, nu using Hooke Law in plane strain
    E, nu = compute_E_nu_plane_strain(epsX=dict_av['av_LE'][1:ind_sep, 0], epsY=dict_av['av_LE'][1:ind_sep, 1], 
                                      sigX=dict_av['av_S'][1:ind_sep, 0], sigZ=dict_av['av_S'][1:ind_sep, 2])
    # Plastic part: fit power law (a, b, c) to 
    array_mises, array_peeq = compute_mises_peeq_from_av(dict_av)
    a = compute_a_parameter(array_mises[ind_sep:], dict_mesh['perc_ac_yield'][ind_sep:], 
                            threshold_perc_ac_yield=threshold_perc_ac_yield)
    a = float(a)
    from scipy.optimize import curve_fit
    pfit, pcov = curve_fit(lambda x, b, c: a+b*10**2*x**(c/10),
                           array_peeq[ind_sep:], array_mises[ind_sep:], bounds=(0, np.inf))
    b, c = pfit[0]*1e2, pfit[1]/10
    if return_pcov:
        return E/1e3, nu, a, b, c, pcov
    return E/1e3, nu, a, b, c


def fit_behavior_law_to_field_data2(dict_mesh, dict_av, a_fixed=400., return_pcov=False):
    """ Fit parameters E, nu, a, b and c to data (field data, needs to do volume averaging)
    a is fixed here """
    array_mises, array_peeq = compute_mises_peeq_from_av(dict_av)
    ind_sep_linear = separate_el_pl(dict_mesh, threshold_perc_ac_yield=0.01)
    ind_sep = int(sum(array_mises < a_fixed))
    # Linear elastic part: compute E, nu using Hooke Law in plane strain
    #E, nu = compute_E_nu_plane_strain(epsX=dict_av['av_LE'][1:ind_sep, 0], epsY=dict_av['av_LE'][1:ind_sep, 1], 
    #                                  sigX=dict_av['av_S'][1:ind_sep, 0], sigZ=dict_av['av_S'][1:ind_sep, 2])
    E, nu = compute_E_nu_plane_strain(epsX=dict_av['av_LE'][1:ind_sep_linear, 0], epsY=dict_av['av_LE'][1:ind_sep_linear, 1], 
                                      sigX=dict_av['av_S'][1:ind_sep_linear, 0], sigZ=dict_av['av_S'][1:ind_sep_linear, 2])
    # Plastic part: fit power law (a, b, c) to 
    from scipy.optimize import curve_fit
    pfit, pcov = curve_fit(lambda x, b, c: a_fixed+b*10**2*x**(c/10),
                           array_peeq[ind_sep:], array_mises[ind_sep:], bounds=(0, np.inf))
    b, c = pfit[0]*1e2, pfit[1]/10
    if return_pcov:
        return E/1e3, nu, b, c, pcov
    return E/1e3, nu, b, c


# The following functions are helper functions for the functions above
def compute_E_nu_plane_strain(epsX, epsY, sigX, sigZ=None, sigY=None):
    if sigZ is not None and sigY is None:
        s = epsX + epsY
        r = epsX / (sigX-sigZ)
        E = (3. * sigZ + s / r) / (s + 2. * r * sigZ)
        nu = E * epsX / (sigX - sigZ) - 1.
    elif sigY is not None and sigZ is None:
        r1 = (sigX+sigY) / (epsX+epsY)
        r2 = (sigX-sigY) / (epsX-epsY)
        nu = 0.5 * (1. - r2 / r1)
        E = (1. + nu) * r2
    else:
        raise ValueError('sigZ or sigY should be non None')
    return np.mean(E), np.mean(nu)


def compute_mises_peeq_from_av(dict_av):
    array_mises = compute_VonMises(sigX=dict_av['av_S'][:, 0], sigY=dict_av['av_S'][:, 1], sigZ=dict_av['av_S'][:, 2], 
                                   sigXY=dict_av['av_S'][:, 3])
    array_peeq = compute_PEEQ(epsX=dict_av['av_PE'][:, 0], epsY=dict_av['av_PE'][:, 1], epsZ=dict_av['av_PE'][:, 2], 
                              epsXY=dict_av['av_PE'][:, 3])
    return array_mises, array_peeq


def compute_VonMises(sigX, sigY, sigZ, sigXY):
    # See http://www.continuummechanics.org/vonmisesstress.html
    #fieldX_dev = sigX - 1/3*(sigX+sigY+sigZ)
    #fieldY_dev = sigY - 1/3*(sigX+sigY+sigZ)
    #fieldZ_dev = sigZ - 1/3*(sigX+sigY+sigZ)
    #tmp = fieldX_dev**2+fieldY_dev**2+fieldZ_dev**2+2*(sigXY**2)
    #field_VonMises = np.sqrt(3/2*tmp)
    tmp = (sigX-sigY)**2 + (sigX-sigZ)**2 + (sigZ-sigY)**2 + 6*sigXY**2
    field_VonMises = np.sqrt(1/2*tmp)
    return field_VonMises

    
def compute_PEEQ(epsX, epsY, epsZ, epsXY):
    fieldX_dev = epsX - 1/3*(epsX+epsY+epsZ)
    fieldY_dev = epsY - 1/3*(epsX+epsY+epsZ)
    fieldZ_dev = epsZ - 1/3*(epsX+epsY+epsZ)
    tmp = fieldX_dev**2+fieldY_dev**2+fieldZ_dev**2+2*(epsXY**2)
    field_PEEQ = np.sqrt(2/3*tmp)
    return field_PEEQ



def separate_el_pl(dict_mesh, threshold_perc_ac_yield=0.01):
    """ Returns the index that separates elastic from plastic behavior 
    (do a[:ind_sep] for get elastic part of array a and a[ind_sep:] to get plastic part) """
    mask_elasticity = (dict_mesh['perc_ac_yield'] < threshold_perc_ac_yield)
    ind_sep = int(sum(mask_elasticity))
    return ind_sep


def compute_a_parameter(stress_pl, perc_ac_yield_pl, threshold_perc_ac_yield=0.01):
    """ Compute a parameter as the point where perc_ac_yield=threshold. 
    Note: better to extrapolate from values above plasticity threshold """
    from scipy.interpolate import interp1d
    f = interp1d(perc_ac_yield_pl, stress_pl, kind='linear', fill_value='extrapolate')
    return f(threshold_perc_ac_yield)


def compute_perc_Mises_above_threshold(dict_mesh, dict_centroids, threshold_mises):
    """ This function computes the volume percentage of fibers for which S_mises is above the given threshold (MPa) """
    flag = False
    if not isinstance(threshold_mises, list):
        flag = True
        threshold_mises = [threshold_mises]
    # elements that belong to a fiber
    list_fibers = np.setdiff1d(list(range(dict_mesh['n_el'])), dict_mesh['el_in_matrix'], assume_unique=True)
    # their mises stress
    mises_in_fibers = dict_centroids['el_Mises'][list_fibers, -1]
    # total volume of fibers
    vol_fibers = np.sum(dict_centroids['el_Vol'][list_fibers, -1])
    
    perc_above = [0] * len(threshold_mises)
    for i, threshold in enumerate(threshold_mises):
        # the ones that are above the threshold
        mises_above = [ind for (i, ind) in enumerate(list_fibers) if mises_in_fibers[i] > threshold]
        # perc of fiber volume with stress above threshold
        perc_above[i] = np.sum(dict_centroids['el_Vol'][mises_above, -1]) / vol_fibers
    if flag:
        return perc_above[0]
    return perc_above


#Functions to normalize the data
def normalize_from_bounds(X, bounds, type_norm='[0,1]'):
    if type_norm == '[0,1]':
        return (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    elif type_norm == '[-1,1]':
        return (X - (bounds[:, 1]+bounds[:, 0])/2.) / ((bounds[:, 1]-bounds[:, 0])/2.)
    
def unnormalize_from_bounds(X, bounds, type_norm='[0,1]'):
    if type_norm == '[0,1]':
        return X * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    elif type_norm == '[-1,1]':
        return X * (bounds[:, 1]-bounds[:, 0])/2. + (bounds[:, 1]+bounds[:, 0])/2.
    
def normalize_covariance_from_bounds(covX, bounds, type_norm='[0,1]'):
    if type_norm == '[0,1]':
        D = 1. / (bounds[:, 1] - bounds[:, 0])
    elif type_norm == '[-1,1]':
        D = 2. / (bounds[:, 1] - bounds[:, 0])
    return np.matmul(np.matmul(np.diag(D), covX), np.diag(D))


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
            ax.axis('equal')
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