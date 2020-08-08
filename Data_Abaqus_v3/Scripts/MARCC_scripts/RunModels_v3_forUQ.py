#### Last modified by Audrey Olivier on 2018/10/30
#### 2D matrix with fibers, periodic boundary conditions
import os
import sys
import glob
row_data_idx = int(sys.argv[-2])
extension_file = sys.argv[-1]

IOdir_name = 'IOfolder'+extension_file

from abaqus import *
from abaqusConstants import *
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
from odbAccess import openOdb
#### This file contains the main functions to build the model
execfile('abaqusModel_functions_v3_forUQ.py')

import time
import csv
#### read data from csv file
def process_inputs_from_csv(inputsDict):
    for strain_ in ['strain_x', 'strain_y']:
        if strain_ in inputsDict.keys() and inputsDict[strain_] != 'None':
            inputsDict[strain_] = float(inputsDict[strain_])
        else:
            inputsDict[strain_] = None
            print('haha')
    inputsDict['dt'] = float(inputsDict['dt'])
    fibers_center = inputsDict['fibers_center'][2:-2].split('), (')
    inputsDict['fibers_center'] = [tuple(map(float, stuff.split(', '))) for stuff in fibers_center]
    inputsDict['fibers_radius'] = float(inputsDict['fibers_radius'])
    point_matrix = inputsDict['point_in_matrix'][1:-1].split(', ')
    inputsDict['point_in_matrix'] = (float(point_matrix[0]), float(point_matrix[1]))
    for domain in ['MATRIX', 'FIBERS']:
        inputsDict[domain.lower()+'_properties'] = tuple(map(float, inputsDict[domain.lower()+'_properties'][1:-1].split(',')))
    inputsDict['mesh_size'] = float(inputsDict['mesh_size'])
    return inputsDict
    
inputsDict = {}
with open(IOdir_name+'/input_fixed_dict'+extension_file+'.csv') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        inputsDict.update(dict(row))

model_name = 'Model-'+str(row_data_idx)
job_name = 'JOB-'+str(row_data_idx)

t0 = time.time()
with open(IOdir_name+'/input_variable_dict'+extension_file+'.csv') as csv_file:
    reader = csv.DictReader(csv_file)
    for row_number, row in enumerate(reader):
        if row_number == row_data_idx:
            inputsDict.update(dict(row))
            inputsDict = process_inputs_from_csv(inputsDict)
            break

#### Define the periodic boundary conditions, DefMat[0] refers to BC applied to vertical nodes
print(inputsDict)
if inputsDict['strain_x'] is not None and inputsDict['strain_y'] is None:
    inputsDict['PBC'] = [(inputsDict['strain_x'], UNSET), (UNSET, UNSET)]
elif inputsDict['strain_x'] is None and inputsDict['strain_y'] is not None:
    inputsDict['PBC'] = [(UNSET, UNSET), (UNSET, inputsDict['strain_y'])]
else:
    raise ValueError('wrong PBC')
t1 = time.time()-t0
print('time to read data = {}'.format(t1))

#### Create and run the model
modelRVE, partRVE, assemblyRVE = createModelGeometry(model_name, inputsDict)
createStepPBC(modelRVE, partRVE, assemblyRVE, inputsDict)
createJob(model_name, job_name)
t2 = time.time()-t0-t1
print('time to analysis = {}'.format(t2))
saveOutputDict(job_name, IOdir_name, inputsDict)
t3 = time.time()-t0-t2
print('time to post-process = {}'.format(t3))
