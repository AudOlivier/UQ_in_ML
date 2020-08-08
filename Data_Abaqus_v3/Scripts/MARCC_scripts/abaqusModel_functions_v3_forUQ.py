# Audrey Olivier - updated in July 2019
# This file defines several utility functions used to run the Abaqus model for a composite microstructure 
# subjected to uniaxial strain (periodic boundary conditions). The properties of the microstructure are defined 
# in an inputsDict dictionary passed as an argument to many of those functions.

import numpy as np
import pickle
import functools

from abaqusConstants import *
from mesh import *
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from job import *
from sketch import *
import regionToolset

# Element type: quad, CPE4 for plane strain, 4 nodes per element, 4 integration points per element
meshShape, el_type, el_code, n_nodes_per_el, n_int_pts = QUAD, CPE4, CPE4R, 4, 4

def generate_points_for_abaqus(radius_inclusion, centers):
    """ This function generates points that are used to define the fibers in abaqus (3 points per fiber) """
    fiber_point1 = [(ctr[0]+radius_inclusion, ctr[1]) for idx, ctr in enumerate(centers)]  # point 1 somewhere on the fiber edge
    fiber_point2 = [(ctr[0], ctr[1]+radius_inclusion) for idx, ctr in enumerate(centers)]  # point 2 somewhere on the fiber edge
    point_inside = np.array(centers)
    point_inside[point_inside[:,0]<0,0] = 0.001
    point_inside[point_inside[:,0]>1,0] = 0.999
    point_inside[point_inside[:,1]<0,1] = 0.001
    point_inside[point_inside[:,1]>1,1] = 0.999
    fiber_point_in = [(ctr[0], ctr[1]) for idx, ctr in enumerate(point_inside)]  # point 3 somewhere inside the fiber
    return fiber_point1, fiber_point2, fiber_point_in
    
def createModelGeometry(model_name, inputsDict):
    """ Create part, materials, sections, mesh and assembly of Abaqus model"""
    
    # Read some geometry inputs from the input dictionary
    fibers_center = inputsDict['fibers_center']
    fibers_radius = inputsDict['fibers_radius']
    point_in_matrix = inputsDict['point_in_matrix']
    print(len(fibers_center))
    print(point_in_matrix)
    if len(fibers_center) != 0:
        fiber_point1, fiber_point2, fiber_point_in = generate_points_for_abaqus(fibers_radius, fibers_center)
    
    modelRVE = mdb.Model(name=model_name)
    ##################################################################################
    #CREATE PART
    ##################################################################################
    modelRVE.ConstrainedSketch(name='__profile__', sheetSize=10.0)
    modelRVE.sketches['__profile__'].rectangle(point1=(0.0, 0.0), point2=(1.0, 1.0))
    partRVE = modelRVE.Part(dimensionality=TWO_D_PLANAR, name='Part-1', type=DEFORMABLE_BODY)
    partRVE.BaseShell(sketch=modelRVE.sketches['__profile__'])
    del modelRVE.sketches['__profile__']

    # Define the location of the fibers
    modelRVE.ConstrainedSketch(gridSpacing=0.0001, name='__profile__', 
        sheetSize=10.0, transform=partRVE.MakeSketchTransform(
        sketchPlane=partRVE.faces.findAt((point_in_matrix[0], point_in_matrix[1], 0.0), (0.0, 0.0, 1.0)), 
        sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.0, 0.0, 0.0)))
    partRVE.projectReferencesOntoSketch(filter=COPLANAR_EDGES, sketch=modelRVE.sketches['__profile__'])
    for i in range(len(fibers_center)):
        modelRVE.sketches['__profile__'].EllipseByCenterPerimeter(
            axisPoint1=fiber_point1[i], axisPoint2=fiber_point2[i], center=fibers_center[i])
    if len(fibers_center) != 0:
        partRVE.PartitionFaceBySketch(faces=
            partRVE.faces.findAt(((point_in_matrix[0], point_in_matrix[1], 
            0.0), )), sketch=modelRVE.sketches['__profile__'])
    del modelRVE.sketches['__profile__']

    ##################################################################################
    #DEFINE MATERIAL
    ##################################################################################
    modelRVE.Material(name='MATERIAL-MATRIX')
    modelRVE.Material(name='MATERIAL-FIBERS')
    for domain in ['MATRIX', 'FIBERS']:
        # Read material properties for matrix and fibers from input dictionary
        prop_ = inputsDict[domain.lower()+'_properties']
        type_ = inputsDict[domain.lower()+'_type_material']
        if type_ == 'plasticity':
            n_pts = 700
            plastic_strain = np.concatenate([np.zeros((1,)), 
                np.exp(np.linspace(np.log(10**(-6)),np.log(2.0),n_pts))])
            stress_ = prop_[2]+prop_[3]*plastic_strain**prop_[4]
            table_ = tuple([(stress_[i], plastic_strain[i]) for i in range(n_pts)])
            #table_ = ((prop_[2], 0.0), )
            modelRVE.materials['MATERIAL-'+domain].Elastic(table=((prop_[0], prop_[1]), ))
            modelRVE.materials['MATERIAL-'+domain].Plastic(table=table_)
        elif type_ == 'elasticity':
            modelRVE.materials['MATERIAL-'+domain].Elastic(table=(prop_, ))    
        elif type_ == 'hyperelasticity_ArrudaBoyce':
            modelRVE.materials['MATERIAL-'+domain].Hyperelastic(
                materialType=ISOTROPIC,table=(prop_,),
                testData=OFF, type=ARRUDA_BOYCE, volumetricResponse=VOLUMETRIC_DATA)
        elif type_ == 'hyperelasticity_NeoHooke':
            modelRVE.materials['MATERIAL-'+domain].Hyperelastic(materialType=
                ISOTROPIC, table=(prop_, ), testData=OFF, type=NEO_HOOKE, 
                volumetricResponse=VOLUMETRIC_DATA)
        else:
            print('Material type non-recognized')
        
    ##################################################################################
    #DEFINE SECTION
    ##################################################################################
    modelRVE.HomogeneousSolidSection(material='MATERIAL-MATRIX', name=
        'SECTION-MATRIX', thickness=None)
    modelRVE.HomogeneousSolidSection(material='MATERIAL-FIBERS', name=
        'SECTION-FIBERS', thickness=None)
      
    partRVE.Set(faces=partRVE.faces.findAt(((point_in_matrix[0], point_in_matrix[1], 0.0), )), 
        name='SET-MATRIX')
    partRVE.SectionAssignment(offset=0.0, offsetField='', offsetType=MIDDLE_SURFACE, 
                              region=partRVE.sets['SET-MATRIX'], sectionName='SECTION-MATRIX', 
                              thicknessAssignment=FROM_SECTION)
    for i in range(len(fibers_center)):
        partRVE.Set(faces=partRVE.faces.findAt(((fiber_point_in[i][0], fiber_point_in[i][1], 0.0), )), 
                    name='SET-FIBER-'+str(i+1))
        partRVE.SectionAssignment(offset=0.0, offsetField='', offsetType=MIDDLE_SURFACE, region=
                                  partRVE.sets['SET-FIBER-'+str(i+1)], sectionName='SECTION-FIBERS', 
                                  thicknessAssignment=FROM_SECTION)

    ##################################################################################
    # MESH
    ##################################################################################
    mesh_size = inputsDict['mesh_size']  # read from input dictionary

    partRVE.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=mesh_size)

    partRVE.setMeshControls(elemShape=meshShape, 
        regions=partRVE.faces.findAt(((point_in_matrix[0], point_in_matrix[1], 0.0), )))
    for i in range(len(fibers_center)):
        partRVE.setMeshControls(elemShape=meshShape,
            regions=partRVE.faces.findAt(((fiber_point_in[i][0], fiber_point_in[i][1], 0.0), )))
    
    partRVE.setElementType(elemTypes=(ElemType(elemCode=el_code, elemLibrary=STANDARD), ElemType(elemCode=el_type, 
                           elemLibrary=STANDARD, secondOrderAccuracy=OFF, distortionControl=DEFAULT)), 
                           regions=(partRVE.faces.findAt(((point_in_matrix[0], point_in_matrix[1], 0.0), )), ))
    for i in range(len(fibers_center)):
        partRVE.setElementType(elemTypes=(ElemType(elemCode=el_code, elemLibrary=STANDARD), ElemType(elemCode=el_type, 
                               elemLibrary=STANDARD, secondOrderAccuracy=OFF, distortionControl=DEFAULT)), 
                               regions=(partRVE.faces.findAt(((fiber_point_in[i][0], fiber_point_in[i][1], 0.0), )), ))
    partRVE.generateMesh()

    ##################################################################################
    #ASSEMBLE
    ##################################################################################
    assemblyRVE = modelRVE.rootAssembly
    assemblyRVE.DatumCsysByDefault(CARTESIAN)
    assemblyRVE.Instance(dependent=ON, name='INSTANCE-1', part=partRVE)
    assemblyRVE.Set(name = 'all_nodes', nodes = assemblyRVE.instances['INSTANCE-1'].nodes[:])

    return modelRVE, partRVE, assemblyRVE
        
def createStepPBC(modelRVE, partRVE, assemblyRVE, inputsDict):
    """ Create a Step for the analysis: periodic boundary conditions are applied as constraints 
    between the two matching edges (left/right and top/bottom) and a dummy node. 
    Displacement is applied to the 2 dummy nodes."""
    
    DefMat = inputsDict['PBC']  # Read the applied strains for the dummy nodes
    
    ##################################################################################
    #CREATE PERIODIC BOUNDARY CONDITIONS
    ##################################################################################
    CoorFixNode, NameRef1, NameRef2, warning_ = PBC_2d(modelRVE, assemblyRVE)
    inputsDict['warning'] = warning_  # This warning says wether the PBC were applied correctly or not

    ##################################################################################
    #CREATE STEP AND APPLY BC
    ##################################################################################
    dt = inputsDict['dt']
    modelRVE.StaticStep(initialInc=dt, maxInc=dt, minInc=0.00001, timePeriod=1.0, 
                        name='STEP-1', nlgeom=ON, previous='Initial', maxNumInc=1000)
    #region_microstructure = regionToolset.Region(nodes=assemblyRVE.instances['INSTANCE-1'].nodes[:])
    modelRVE.fieldOutputRequests['F-Output-1'].setValues(region=Region(nodes=assemblyRVE.instances['INSTANCE-1'].nodes[:], 
        elements=assemblyRVE.instances['INSTANCE-1'].elements[:]), 
        variables=('U', 'S', 'LE', 'COORD', 'EVOL', 'IVOL', 'PEEQ', 'PE', 'EE')) #U, SENER, IVOL, SVOL
    for inst_dummy in [NameRef1, NameRef2]:
        modelRVE.FieldOutputRequest(name='output_'+inst_dummy, createStepName='STEP-1', region=Region(referencePoints=(
            assemblyRVE.instances[inst_dummy].referencePoints[1],)), variables=('RF', )) #U, SENER, IVOL, SVOL
    #Apply boundary conditions on reference nodes
    modelRVE.DisplacementBC(amplitude=UNSET, createStepName='STEP-1', 
        distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=
        'BC-REF-1', region=Region(referencePoints=(
        assemblyRVE.instances[NameRef1].referencePoints[1], 
        )), u1=DefMat[0][0], u2=DefMat[0][1], ur3=0.)
    modelRVE.DisplacementBC(amplitude=UNSET, createStepName='STEP-1', 
        distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=
        'BC-REF-2', region=Region(referencePoints=(
        assemblyRVE.instances[NameRef2].referencePoints[1], 
        )), u1=DefMat[1][0], u2=DefMat[1][1], ur3=0.)
    modelRVE.DisplacementBC(amplitude=UNSET, createStepName='STEP-1', 
        distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=
        'BC-FIXNODE', region=Region(
        nodes=assemblyRVE.instances['INSTANCE-1'].nodes.getByBoundingSphere(center=CoorFixNode, 
        radius=0.00001)), u1=0.0, u2=0.0, ur3=UNSET)
    #mdb.models['Model--'].steps['Step-1'].setValues(initialInc=0.1, noStop=OFF, 
    #    timeIncrementationMethod=FIXED)

def createJob(model_name, job_name):
    ##################################################################################
    #JOB AND RUN
    ##################################################################################
    mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF, 
        explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF, 
        memory=90, memoryUnits=PERCENTAGE, model=model_name, modelPrint=OFF, 
        multiprocessingMode=DEFAULT, name=job_name, nodalOutputPrecision=SINGLE, 
        numCpus=1, queue=None, scratch='', type=ANALYSIS, userSubroutine='', 
        waitHours=0, waitMinutes=0)
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    mdb.jobs[job_name].waitForCompletion()

def saveOutputDict(job_name, IOdir_name, inputsDict):
    o1 = session.openOdb(name=job_name+'.odb') 

    # save plots
    # save_plots(o1, IOdir_name, job_name)

    # Isolate the instance, get the number of nodes, elements and frames
    myInstance = o1.rootAssembly.instances['INSTANCE-1']
    numNodes = len(myInstance.nodes)
    numElements = len(myInstance.elements)
    frameRepository = o1.steps['STEP-1'].frames
    n_frames = len(frameRepository)-1
    list_frames = [fr.frameValue for fr in frameRepository]
    ind_frames = list(range(len(frameRepository)))
    ind_frames_centroid = [ind_frames[-1]] #[4, ind_frames[len(frameRepository)//2], ind_frames[-1]]

    # save mesh ouputs and displacements
    node_coord = np.zeros((numNodes, 2))
    mask = [n.label-1 for n in myInstance.nodes]
    node_coord[mask,:] = [n.coordinates[0:2] for n in myInstance.nodes]

    el_connectivity = np.zeros((numElements, n_nodes_per_el))
    mask = [e.label-1 for e in myInstance.elements]
    tmp_connectivity = np.array([e.connectivity for e in myInstance.elements])
    el_connectivity[mask,:] = tmp_connectivity-np.ones_like(tmp_connectivity)

    el_in_matrix = [el.label-1 for el in myInstance.elementSets['SET-MATRIX'].elements]

    node_U = np.zeros((numNodes, len(ind_frames_centroid), 2))
    # Save values of displacement at nodes
    for fr_ind, fr in enumerate(ind_frames_centroid):
        Disp = frameRepository[fr].fieldOutputs['U'].values
        mask = [v.nodeLabel-1 for v in Disp]
        for i in range(2):
            node_U[mask,fr_ind,i] = [v.data[i] for v in Disp]

    # check yielding of elements
    perc_ac_yield = np.zeros(len(ind_frames))
    tmp_vol, tmp_ac_yield = np.zeros((numElements,)), np.zeros((numElements,))
    for fr_ind, fr in enumerate(ind_frames):
        # volume of each element
        vol_field = frameRepository[fr].fieldOutputs['EVOL'].values
        mask = [v.elementLabel-1 for v in vol_field]
        tmp_vol[mask] = [v.data for v in vol_field]
        ac_yield_field = frameRepository[fr].fieldOutputs['AC YIELD'].values
        mask = [v.elementLabel-1 for v in ac_yield_field]
        tmp_ac_yield[mask] = [v.data for v in ac_yield_field]
        perc_ac_yield[fr_ind] = sum([v * ac for (v, ac) in zip(tmp_vol, tmp_ac_yield)]) / sum(tmp_vol)

    if inputsDict['strain_x'] is not None and inputsDict['strain_y'] is None:
        strain_applied = np.array(list_frames) * inputsDict['strain_x']
    elif inputsDict['strain_x'] is None and inputsDict['strain_y'] is not None:
        strain_applied = np.array(list_frames) * inputsDict['strain_y']
    outputs_mesh = {'node_U': node_U, 'node_coord': node_coord, 'el_connectivity': el_connectivity, 
        'el_in_matrix':el_in_matrix, 'perc_ac_yield': perc_ac_yield,
        'strain_applied': strain_applied, 'value_frames': list_frames, 
        'warning': inputsDict['warning']}
        
    #dumnode_RF, length_right, length_top, strain_x, strain_y = extract_reaction_forces(
    #    myInstance, frameRepository, ind_frames, node_coord)
    #outputs_mesh.update({'dumnode_RF': dumnode_RF, 'strain_x': strain_x, 'strain_y': strain_y,
    #    'length_right_side': length_right, 'length_top_side': length_top})
    
    with open(IOdir_name+'/'+job_name + '_outputs_mesh.pkl', 'wb') as f:
        pickle.dump(outputs_mesh, f)
    
    # Save value at centroids and averages
    save_at_centroid_and_average(myInstance, frameRepository, el_type, IOdir_name, job_name, 
        ind_frames, ind_frames_centroid, Mises_bool=True, PEEQ_bool=True,
        quantities_to_save = (('S', ['S11', 'S22', 'S33', 'S12'], []),
        ('LE', ['LE11', 'LE22', 'LE33', 'LE12'], []),
        ('PE', ['PE11', 'PE22', 'PE33', 'PE12'], []),
        ('EE', ['EE11', 'EE22', 'EE33', 'EE12'], []),))
    #save_at_centroid_and_average(myInstance, frameRepository, el_type, IOdir_name, job_name, 
    #    ind_frames, [], Mises_bool=False, PEEQ_bool=False,
    #    quantities_to_save = (('S', ['S11', 'S22', 'S33', 'S12'], []),
    #    ('LE', ['LE11', 'LE22', 'LE33', 'LE12'], []),
    #    ('PE', ['PE11', 'PE22', 'PE33', 'PE12'], []),
    #    ('EE', ['EE11', 'EE22', 'EE33', 'EE12'], []),))

    # Save value at centroids and averages
    #save_Mises_at_centroids(myInstance, frameRepository, el_type, IOdir_name, job_name, 
    #    ind_frames_centroid)
    # Save value at integration
    #save_at_integration(o1, el_type, n_int_pts, IOdir_name, job_name, ind_frames=[-1],
    #    S_list = ['S11'], 
    #    LE_list = ['LE11'], 
    #    PE_list = ['PE11'])
    # Save fields at pixels for later learning
    #save_field_pixels(IOdir_name, job_name, n_pixels_side = 224, which_step=-1)

#from scipy.spatial.distance import cdist
def extract_boundary_nodes(myInstance):
    nodes_left = []; nodes_left_coord = []
    nodes_right = []; nodes_right_coord = []
    nodes_top = []; nodes_top_coord = []
    nodes_bottom = []; nodes_bottom_coord = []
    one_free_node = []
    boundary_nodes_label = {'left': [], 'right': [], 'top': [], 'bottom': []}
    for ind_n, n in enumerate(myInstance.nodes):
        if n.coordinates[0] < 1e-6:
            nodes_left.append(ind_n)
            nodes_left_coord.append(n.coordinates[0:2])
            boundary_nodes_label['left'].append(n.label-1)
        elif n.coordinates[0] > 1. - 1e-6:
            nodes_right.append(ind_n)
            nodes_right_coord.append(n.coordinates[0:2])
            boundary_nodes_label['right'].append(n.label-1)
        elif n.coordinates[1] < 1e-6:
            nodes_bottom.append(ind_n)
            nodes_bottom_coord.append(n.coordinates[0:2])
            boundary_nodes_label['bottom'].append(n.label-1)
        elif n.coordinates[1] > 1. - 1e-6:
            nodes_top.append(ind_n)
            nodes_top_coord.append(n.coordinates[0:2])
            boundary_nodes_label['top'].append(n.label-1)
        else:
            if len(one_free_node) == 0:
                one_free_node.append(n)
    # Left to right: compute distance between point on the left and all points on the right,
    # Pair the left node with the point on the closest point on the right
    warning_nodes = ['Okay', 'Okay']
    nodes_left_coord = np.array(nodes_left_coord)
    nodes_right_coord = np.array(nodes_right_coord)
    dist_LR = np.zeros((len(nodes_left), len(nodes_right)))
    for i, node_left_coord in enumerate(nodes_left_coord):
        dist_LR[i, :] = np.apply_along_axis(np.linalg.norm, 1, np.tile(node_left_coord.reshape(1, 2), [len(nodes_right_coord), 1])-nodes_right_coord)
    if dist_LR.shape[0] <= dist_LR.shape[1]:  # less or equal number of nodes on left than right
        ind_right = np.argmin(dist_LR, axis=-1)
        nodes_right = [nodes_right[i] for i in ind_right]
        boundary_nodes_label['right'] = [boundary_nodes_label['right'][i] for i in ind_right]
    else: 
        ind_left = np.argmin(dist_LR, axis=0)
        nodes_left = [nodes_left[i] for i in ind_left]
        boundary_nodes_label['left'] = [boundary_nodes_label['left'][i] for i in ind_left]
    if dist_LR.shape[0] != dist_LR.shape[1]:  # raise a warning if not same number of nodes
        warning_nodes[0] = 'Not same number of nodes on left and right !'

    nodes_bottom_coord = np.array(nodes_bottom_coord)
    nodes_top_coord = np.array(nodes_top_coord)
    dist_BT = np.zeros((len(nodes_bottom), len(nodes_top)))
    for i, node_bottom_coord in enumerate(nodes_bottom_coord):
        dist_BT[i, :] = np.apply_along_axis(np.linalg.norm, 1, np.tile(node_bottom_coord.reshape(1, 2), [len(nodes_top_coord), 1])-nodes_top_coord)
    if dist_BT.shape[0] <= dist_BT.shape[1]:  # less or equal number of nodes on bottom than up
        ind_top = np.argmin(dist_BT, axis=-1)
        nodes_top = [nodes_top[i] for i in ind_top]
        boundary_nodes_label['top'] = [boundary_nodes_label['top'][i] for i in ind_top]
    else: 
        ind_bottom = np.argmin(dist_BT, axis=0)
        nodes_bottom = [nodes_bottom[i] for i in ind_bottom]
        boundary_nodes_label['bottom'] = [boundary_nodes_label['bottom'][i] for i in ind_bottom]
    if dist_BT.shape[0] != dist_BT.shape[1]:  # raise a warning if not same number of nodes
        warning_nodes[1] = 'Not same number of nodes on bottom and top !'

    return nodes_left, nodes_right, nodes_bottom, nodes_top, one_free_node[0], warning_nodes, boundary_nodes_label
    
    
#Created by J.T.B. Overvelde
#2015/07/07
#http://www.overvelde.com
#CREATED IN ABAQUS VERSION 6.11-1.

#FUNCTION TO APPLY PERIODIC BOUNDARY CONDITIONS IN 2D
#mdb: model database
#NameModel: 	A string with the name of your model
#NameSet: 	A string with the name of your set (for a faster script, this set 
#		should only contain those nodes that will have periodic boundary conditions applied to them)
#LatticeVec:	An array with the lattice vectors, for example [(1.0, 0.0), (0.0, 1.0)] for a square lattice
def PBC_2d(modelRVE, assemblyRVE):
    """ See codes by JTB Overvelde, 2015, or paper 'Applying Periodic Boundary Conditions in Finite Element Analysis', Wu et al., 2014"""

    # Get the points on the boundaries
    nodes_left, nodes_right, nodes_bottom, nodes_top, one_node_free, warning_nodes, _ = extract_boundary_nodes(
        assemblyRVE.instances['INSTANCE-1'])
    #Left to right PBC
    #Create reference parts and assemble
    NameRef_LR='RefPoint-LR'
    modelRVE.Part(dimensionality=TWO_D_PLANAR, name=NameRef_LR, type=DEFORMABLE_BODY)
    modelRVE.parts[NameRef_LR].ReferencePoint(point=(0.0, 0.0, 0.0))
    modelRVE.rootAssembly.Instance(dependent=ON, name=NameRef_LR, part=modelRVE.parts[NameRef_LR])
    #Create set of reference points
    assemblyRVE.Set(name=NameRef_LR, referencePoints=(
        assemblyRVE.instances[NameRef_LR].referencePoints[1],))

    NameRef_TB='RefPoint-TB'
    modelRVE.Part(dimensionality=TWO_D_PLANAR, name=NameRef_TB, type=DEFORMABLE_BODY)
    modelRVE.parts[NameRef_TB].ReferencePoint(point=(0.0, 0.0, 0.0))
    assemblyRVE.Instance(dependent=ON, name=NameRef_TB, part=modelRVE.parts[NameRef_TB])
    assemblyRVE.Set(name=NameRef_TB, referencePoints=(
        assemblyRVE.instances[NameRef_TB].referencePoints[1],))

    #Create sets for use in equations constraints
    num_constraint = 0
    for n_left, n_right in zip(nodes_left, nodes_right):
        assemblyRVE.Set(name='Node-1-'+str(num_constraint), nodes=assemblyRVE.instances['INSTANCE-1'].nodes[n_left: n_left+1])
        assemblyRVE.Set(name='Node-2-'+str(num_constraint), nodes=assemblyRVE.instances['INSTANCE-1'].nodes[n_right: n_right+1])
        #Create equations constraints for each dof
        for dim in [1, 2]:
            modelRVE.Equation(name='PerConst'+str(dim)+'-'+str(num_constraint),
            terms=((1.0,'Node-1-'+str(num_constraint), dim),(-1.0, 'Node-2-'+str(num_constraint), dim) ,
                (1.0, 'RefPoint-LR', dim)))
        num_constraint += 1

    for n_bottom, n_top in zip(nodes_bottom, nodes_top):
        assemblyRVE.Set(name='Node-1-'+str(num_constraint), nodes=assemblyRVE.instances['INSTANCE-1'].nodes[n_bottom: n_bottom+1])
        assemblyRVE.Set(name='Node-2-'+str(num_constraint), nodes=assemblyRVE.instances['INSTANCE-1'].nodes[n_top: n_top+1])
        #Create equations constraints for each dof
        for dim in [1,2]:
            modelRVE.Equation(name='PerConst'+str(dim)+'-'+str(num_constraint),
            terms=((1.0,'Node-1-'+str(num_constraint), dim),(-1.0, 'Node-2-'+str(num_constraint), dim) ,
                (1.0, 'RefPoint-TB', dim)))
        num_constraint += 1
    return one_node_free.coordinates, NameRef_LR, NameRef_TB, warning_nodes
    

def save_at_centroid_and_average(myInstance, frameRepository, el_type, IOdir_name, 
    job_name, ind_frames, ind_frames_centroid, Mises_bool=True, PEEQ_bool=True,
    quantities_to_save = (('S', ['S11', 'S22', 'S33', 'S12'], ['S11']),
                          ('LE', ['LE11', 'LE22', 'LE33', 'LE12'], ['LE11']),
                          ('PE', ['PE11', 'PE22', 'PE33', 'PE12'], ['PE11']),
                          ('EE', ['EE11', 'EE22', 'EE33', 'EE12'], ['EE11']),)):
    """ Save certain quanities at centroid and volume averages
    quantities_to_save is a tuple of tuples (name, to_save_as_average, to_save_at_centroid)""" 
    
    numNodes = len(myInstance.nodes)
    numElements = len(myInstance.elements)
    n_frames = len(ind_frames)
    n_frames_centroid = len(ind_frames_centroid)

    # Initialize matrices: volume element, mises, peeq
    el_Vol = np.zeros((numElements, n_frames_centroid))
    el_Mises = np.zeros((numElements, n_frames_centroid))
    el_PEEQ = np.zeros((numElements, n_frames_centroid))
    #el_COORD = np.zeros((numElements, n_frames, 2))

    # Then other quantities and their components
    n_quantities = len(quantities_to_save)
    qoi_avs = [[] for _ in  range(n_quantities)]
    qoi_centroids = [[] for _ in  range(n_quantities)]
    qoi_names = [[] for _ in  range(n_quantities)]
    qoi_lists = [[] for _ in  range(n_quantities)]
    for j, qoi in enumerate(quantities_to_save):
        qoi_names[j], qoi_list_av, qoi_list_centroid = qoi[0], qoi[1], qoi[2]
        qoi_lists[j] = list(set(qoi_list_av).union(qoi_list_centroid))

        if len(qoi_list_av) != 0: 
            qoi_avs[j] = np.zeros((n_frames, len(qoi_list_av)))
        else: 
            qoi_avs[j] = None
        if len(qoi_list_centroid) != 0: 
            qoi_centroids[j] = np.zeros((numElements, n_frames_centroid, len(qoi_list_centroid)))
        else: 
            qoi_centroids[j] = None

    # Loop over all the frames and save the required fields and components
    for fr_ind, fr in enumerate(ind_frames):
        # First save the volume of each element, will be used to compute the volume averages
        scalar_field = frameRepository[fr].fieldOutputs['EVOL'].values
        mask = [v.elementLabel-1 for v in scalar_field]
        tmp_vol = [v.data for v in scalar_field]
        vol_sum = np.sum(tmp_vol)
        ind_sorted_mask = np.argsort(mask)
        tmp_vol_by_elementLabel = [tmp_vol[i] for i in ind_sorted_mask]
        if fr in ind_frames_centroid:
            el_Vol[mask, ind_frames_centroid.index(fr)] = tmp_vol
            # Save Mises
            if Mises_bool:
                scalar_field = frameRepository[fr].fieldOutputs['S'].getScalarField(invariant=MISES).getSubset(
                    position = CENTROID, elementType=str(el_type)).values
                mask = [v.elementLabel-1 for v in scalar_field]
                list_tmp = [v.data/1e6 for v in scalar_field]
                el_Mises[mask, ind_frames_centroid.index(fr)] = list_tmp
            # Save PEEQ
            if PEEQ_bool:
                scalar_field = frameRepository[fr].fieldOutputs['PEEQ'].getSubset(position=CENTROID, 
                    elementType=str(el_type)).values
                mask = [v.elementLabel-1 for v in scalar_field]
                list_tmp = [v.data for v in scalar_field]
                el_PEEQ[mask, ind_frames_centroid.index(fr)] = list_tmp
        # Then look at other variables and compoenents
        for j, qoi in enumerate(quantities_to_save):
            qoi_name, qoi_list_av, qoi_list_centroid = qoi[0], qoi[1], qoi[2]
            if len(qoi_lists[j])!= 0:
                for _, comp_label in enumerate(qoi_lists[j]):
                    scalar_field = frameRepository[fr].fieldOutputs[qoi_name].getScalarField(
                        componentLabel=comp_label).getSubset(
                        position=CENTROID, elementType=str(el_type)).values
                    mask = [v.elementLabel-1 for v in scalar_field]
                    if qoi_name == 'S':
                        list_tmp = [v.data/1e6 for v in scalar_field]
                    else:
                        list_tmp = [v.data for v in scalar_field]
                    list_tmp_by_elementLabel = [list_tmp[i] for i in np.argsort(mask)]
                    if comp_label in qoi_list_av:
                        vol_sum_fieldX = [vol_el*field for (vol_el, field) in zip(tmp_vol_by_elementLabel, list_tmp_by_elementLabel)]
                        qoi_avs[j][fr_ind, qoi_list_av.index(comp_label)] = sum(vol_sum_fieldX)/vol_sum
                    if (comp_label in qoi_list_centroid) and (fr in ind_frames_centroid):
                        qoi_centroids[j][mask, ind_frames_centroid.index(fr), qoi_list_centroid.index(comp_label)] = list_tmp
            
    len_qois_centroid = sum([len(qoi[2]) for qoi in quantities_to_save])
    if Mises_bool or PEEQ_bool or len_qois_centroid >= 1:
        outputs_centroid = dict(zip(['el_'+qoi_name for qoi_name in qoi_names], qoi_centroids))
        outputs_centroid.update({'el_Mises':el_Mises, 'el_PEEQ': el_PEEQ, 'el_Vol':el_Vol})#, 'el_COORD':el_COORD}
        with open(IOdir_name+'/'+job_name + '_outputs_centroid.pkl', 'wb') as f:
            pickle.dump(outputs_centroid, f)
    
    len_qois_av = sum([len(qoi[1]) for qoi in quantities_to_save])
    if len_qois_av >=1:
        outputs_volume_averages = dict(zip(['av_'+qoi_name for qoi_name in qoi_names], qoi_avs))
        with open(IOdir_name+'/'+job_name + '_outputs_volume_averages.pkl', 'wb') as f:
            pickle.dump(outputs_volume_averages, f)
        
def extract_reaction_forces(myInstance, frameRepository, ind_frames, node_coord):
    numNodes = len(myInstance.nodes)
    n_frames = len(ind_frames)
    
    # get reaction forces at dummy nodes
    dumnode_RF = np.zeros((2, len(ind_frames), 2))
    for fr_ind, fr in enumerate(ind_frames):
        reac_forces = frameRepository[fr].fieldOutputs['RF'].values
        for i, v in enumerate(reac_forces):
            if v.instance.name == 'REFPOINT-LR':
                dumnode_RF[0, fr_ind, :] = [rf/1e6 for rf in v.data]
            if v.instance.name == 'REFPOINT-TB':
                dumnode_RF[1, fr_ind, :] = [rf/1e6 for rf in v.data]
                
    # Compute the modified length on the right and top sides
    length_right = np.zeros((len(ind_frames), ))
    length_top = np.zeros((len(ind_frames), ))
    for i in [0, 1]:
        mask_right = node_coord[:, i] > 1.-1e-6
        nodes_right_coord = node_coord[mask_right, :]
        mask_sort = np.argsort(nodes_right_coord[:, int(1-i)])
        new_coord = np.zeros((numNodes, 2))
        for fr_ind, fr in enumerate(ind_frames):
            coord = frameRepository[fr].fieldOutputs['COORD'].values
            mask = [v.nodeLabel-1 for v in coord]
            new_coord[mask, :] = np.array([v.data[:2] for v in coord])
            new_coord_right = new_coord[mask_right]
            new_coord_sort = new_coord_right[mask_sort]
            all_dist = np.apply_along_axis(np.linalg.norm, 1, new_coord_sort[1:]-new_coord_sort[:-1])
            if i == 0:
                length_right[fr_ind] = np.sum(all_dist)
            if i == 1:
                length_top[fr_ind] = np.sum(all_dist)
        
    # Compute the strains in x and y directions
    _, _, _, _, _, _, boundary_nodes_label = extract_boundary_nodes(myInstance)
    strain_x = np.zeros((len(ind_frames), ))
    strain_y = np.zeros((len(ind_frames), ))
    for i, (side1, side2) in enumerate(zip(['right', 'top'], ['left', 'bottom'])):
        labels_side1 = boundary_nodes_label[side1]
        new_coord = np.zeros((numNodes, ))
        for fr_ind, fr in enumerate(ind_frames):
            coord = frameRepository[fr].fieldOutputs['COORD'].values
            mask = [v.nodeLabel-1 for v in coord]
            new_coord[mask] = np.array([v.data[i] for v in coord])
            new_coord_side1 = [new_coord[n] for n in boundary_nodes_label[side1]]
            new_coord_side2 = [new_coord[n] for n in boundary_nodes_label[side2]]
            if i == 0:
                strain_x[fr_ind] = np.mean(np.array(new_coord_side1) - np.array(new_coord_side2)) - 1.
            if i == 1:
                strain_y[fr_ind] = np.mean(np.array(new_coord_side1) - np.array(new_coord_side2)) - 1.
    return dumnode_RF, length_right, length_top, strain_x, strain_y

def save_plots(o1, IOdir_name, job_name):
    # Save plots
    myViewport = session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=160, height=160)
    myViewport.makeCurrent()
    myViewport.maximize()
    myViewport.setValues(displayedObject=o1)
    myViewport.odbDisplay.commonOptions.setValues(deformationScaling=UNIFORM,uniformScaleFactor=1.0)
    myViewport.odbDisplay.setFrame(step=0, frame=-1)

    myViewport.odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF, ))
    myViewport.odbDisplay.setPrimaryVariable(variableLabel='U', outputPosition=NODAL,refinement=(COMPONENT, 'U1'),)
    myViewport.view.fitView()
    session.printToFile(fileName=IOdir_name+'/U_'+job_name, format=TIFF, 
                        canvasObjects=(myViewport,))

    myViewport.odbDisplay.setPrimaryVariable(variableLabel='S',outputPosition=INTEGRATION_POINT,
        refinement=(INVARIANT, 'Mises'), )
    myViewport.view.fitView()
    session.printToFile(fileName=IOdir_name+'/VonMises_'+job_name, format=TIFF, 
                        canvasObjects=(myViewport,))
    #myViewport.odbDisplay.setPrimaryVariable(variableLabel='S',outputPosition=INTEGRATION_POINT,
    #    refinement=(COMPONENT, 'S11'), )
    #myViewport.view.fitView()
    #session.printToFile(fileName=IOdir_name+'/S11_'+job_name, format=TIFF, 
    #                    canvasObjects=(myViewport,))
    #myViewport.odbDisplay.setPrimaryVariable(variableLabel='PEEQ',outputPosition=INTEGRATION_POINT, )
    #myViewport.view.fitView()
    #session.printToFile(fileName=IOdir_name+'/PEEQ_'+job_name, format=TIFF, 
    #                    canvasObjects=(myViewport,))
    #myViewport.odbDisplay.setPrimaryVariable(variableLabel='LE',outputPosition=INTEGRATION_POINT,
    #    refinement=(COMPONENT, 'LE11'), )
    #myViewport.view.fitView()
    #session.printToFile(fileName=IOdir_name+'/LE11_'+job_name, format=TIFF, 
    #                    canvasObjects=(myViewport,))


################## Non-used functions ################

def element_type(mesh_type, n_nodes_per_el):
    if mesh_type.lower() == 'tri':
        meshShape = TRI
        if n_nodes_per_el == 3:
            el_type, el_code, n_int_pts = CPE3, CPE4R, 1
        elif n_nodes_per_el == 6:
            el_type, el_code, n_int_pts = CPE6, CPE8R, 3
    elif mesh_type.lower() == 'quad':
        meshShape = QUAD
        if n_nodes_per_el == 4:
            el_type, el_code, n_int_pts = CPE4, CPE4R, 4
        elif n_nodes_per_el == 8:
            el_type, el_code, n_int_pts = CPE8, CPE8R, 9
    return meshShape, el_type, el_code, n_int_pts

def save_at_integration(o1, el_type, n_int_pts, IOdir_name, job_name, ind_frames,
    S_list = ['S11','S22','S12'], LE_list = ['LE11','LE22','LE12'], PE_list = ['PE11','PE22','PE12']) :
    """ Save quantities of interest at integration points """
    myInstance = o1.rootAssembly.instances['INSTANCE-1']
    numNodes = len(myInstance.nodes)
    numElements = len(myInstance.elements)
    n_frames = len(ind_frames)
    frameRepository = o1.steps['STEP-1'].frames
    n_pts = numElements*n_int_pts

    if len(LE_list) != 0: 
        el_LE = np.zeros((n_pts, n_frames, len(LE_list)))
    else: 
        el_LE = None
    if len(S_list) != 0: 
        el_S = np.zeros((n_pts, n_frames, len(S_list)))
    else: 
        el_S = None
    if len(PE_list) != 0: 
        el_PE = np.zeros((n_pts, n_frames, len(PE_list)))
    else: 
        el_PE = None
    el_Mises = np.zeros((n_pts, n_frames))
    el_PEEQ = np.zeros((n_pts, n_frames))
    el_Vol = np.zeros((n_pts, n_frames))
    el_COORD = np.zeros((n_pts, n_frames, 2))
    
    # Save values at centroid of elements
    tmp_dict = {'S': S_list, 'LE': LE_list, 'PE': PE_list}
    for fr_ind, fr in enumerate(ind_frames):
        for key,val in tmp_dict.items():
            if len(val) != 0:
                for i, comp_label in enumerate(val):
                    scalar_field = frameRepository[fr].fieldOutputs[key].getScalarField(
                        componentLabel=comp_label).getSubset(
                        position=INTEGRATION_POINT).values
                    mask = [(v.elementLabel-1)*n_int_pts+(v.integrationPoint-1) for v in scalar_field]
                    if key == 'S':
                        el_S[mask,fr_ind,i] = [v.data for v in scalar_field]
                    elif key == 'LE':
                        el_LE[mask,fr_ind,i] = [v.data for v in scalar_field]
                    elif key == 'PE':
                        el_PE[mask,fr_ind,i] = [v.data for v in scalar_field]

        scalar_field = frameRepository[fr].fieldOutputs['S'].getScalarField(invariant=MISES).getSubset(
            position=INTEGRATION_POINT).values
        mask = [(v.elementLabel-1)*n_int_pts+(v.integrationPoint-1) for v in scalar_field]
        el_Mises[mask,fr_ind] = [v.data for v in scalar_field]

        scalar_field = frameRepository[fr].fieldOutputs['PEEQ'].getSubset(position=INTEGRATION_POINT
            ).values
        mask = [(v.elementLabel-1)*n_int_pts+(v.integrationPoint-1) for v in scalar_field]
        el_PEEQ[mask, fr_ind] = [v.data for v in scalar_field]

        scalar_field = frameRepository[fr].fieldOutputs['COORD'].getSubset(position=INTEGRATION_POINT
            ).values
        mask = [(v.elementLabel-1)*n_int_pts+(v.integrationPoint-1) for v in scalar_field]
        el_COORD[mask, fr_ind, 0] = [v.data[0] for v in scalar_field]
        el_COORD[mask, fr_ind, 1] = [v.data[1] for v in scalar_field]

        scalar_field = frameRepository[fr].fieldOutputs['IVOL'].values
        mask = [(v.elementLabel-1)*n_int_pts+(v.integrationPoint-1) for v in scalar_field]
        el_Vol[mask, fr_ind] = [v.data for v in scalar_field]

    outputs_integration = {'int_S':el_S, 'int_LE':el_LE, 'int_PE':el_PE, 'int_Mises':el_Mises, 'int_PEEQ': el_PEEQ,
        'int_Vol':el_Vol, 'int_COORD':el_COORD}
    with open(IOdir_name+'/'+job_name + '_outputs_integration.pkl', 'wb') as f:
        pickle.dump(outputs_integration, f)
        
def save_Mises_at_centroids(myInstance, frameRepository, el_type, IOdir_name, 
    job_name, ind_frames_centroid):
    
    numElements = len(myInstance.elements)
    n_frames_centroid = len(ind_frames_centroid)
    el_Mises = np.zeros((numElements, n_frames_centroid))
    el_Vol = np.zeros((numElements, n_frames_centroid))
    for fr_ind, fr in enumerate(ind_frames_centroid):
        # Save Mises
        scalar_field = frameRepository[fr].fieldOutputs['S'].getScalarField(invariant=MISES).getSubset(
            position=CENTROID, elementType=str(el_type)).values
        mask = [v.elementLabel-1 for v in scalar_field]
        el_Mises[mask, fr_ind] = [v.data/1e6 for v in scalar_field]
        # Save volume
        scalar_field = frameRepository[fr].fieldOutputs['EVOL'].values
        mask = [v.elementLabel-1 for v in scalar_field]
        el_Vol[mask, fr_ind] = [v.data for v in scalar_field]
        
    outputs_centroid = {'el_Mises':el_Mises, 'el_Vol':el_Vol}#, 'el_COORD':el_COORD}
    with open(IOdir_name+'/'+job_name + '_outputs_centroid.pkl', 'wb') as f:
        pickle.dump(outputs_centroid, f)