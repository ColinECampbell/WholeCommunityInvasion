'''
Here we consider our 1000 interaction graphs. For each, we:
    1. choose 100 random starting communities
    2. fix half of the plants and half of the pollinators to be OFF
    3. Allow each to dynamically evolve to a stable community, C
    4. return to steps 2-3 with the other half of the species fixed OFF

For each iteration, we record:
    1. (list) spectral nestedness at each time step
    2. (list) # of present species at each time step
    3. (list) connectance at each time step
    4. (int) the time step where the graph initially stabilizes
    

Master workflow:
    1. What are the species interaction graphs? (create_graphs.py)
    2. What are the stable communities in these graphs, if we segregate them
       in to two "half communities" to enforce isolated communities that
       nonetheless have inter-community interactions for when invasions occur? 
       (THIS FILE)

The file is structured to run either on a local machine or submitted to a 
High-Performance Computing center. See comments.
(This research was performed via the Ohio  Supercomputer Center, osc.edu)

Python 3.x 

Colin Campbell
campbeco@mountunion.edu
'''

import function_library as nv
import networkx as nx
import numpy as np
import pickle
import time
import os

TIME1 = time.time()
#PBS_ITERATOR=int(os.environ["PBS_ARRAYID"])       #We use this variable to slice up a submission across an array on a HPC. Use: qsub -t 0-99 pbs_script.sh
PBS_ITERATOR=0                                     #Or dummy variable for local debugging

# Load graphs from memory ------------------------------------------------------
#Local 
f = open(r'graphlist.data','rb')  
# OSC
#f = open(r'/path/to/graphlist.data','rb')
graphs = pickle.load(f)
f.close()

graphs=graphs[10*PBS_ITERATOR:(10*PBS_ITERATOR)+10]

N = nx.number_of_nodes(graphs[0])
nodes = nv.node_sorter(graphs[0].nodes())
data = {}

# Perform analysis -------------------------------------------------------------
for graph_index in range(len(graphs)):
    TIME2 = time.time()
    print('Beginning graph {0} at {1:.2f} minutes elapsed.'.format(graph_index,(TIME2-TIME1)/60))
    data[10*PBS_ITERATOR+graph_index]={}
    for start_index in range(100):
        # if start_index % 10 == 0:
        #     TIME2 = time.time()
        #     print('\tBeginning iteration {0} at {1:.2f} minutes elapsed.'.format(start_index,(TIME2-TIME1)/60))            
        #print('Start index = {0}'.format(start_index))
        #Initialize graph and find initial attractor
        #inner dictionaries that will hold the data for this iteration
        inner_data_A={'start_state':[],'stable_state':[],'sys_state':[],'SN':[],'C':[],'size':[],'benchmark':0,'adj':[]}
        inner_data_B={'start_state':[],'stable_state':[],'sys_state':[],'SN':[],'C':[],'size':[],'benchmark':0,'adj':[]}
        start_state = ''.join(['0' if np.random.randint(2) == 0 else '1' for x in range(N)])    #choose a random state for this graph
        
        # trial A with first half of the species forced OFF; trial B with second half
        # (assume 50 plants, 150 pollinators, so 25 and 75 of each, respectively)
        off_list_A = list(range(25))+list(range(50,125))
        off_list_B = list(range(25,50))+list(range(125,200))
        
        # execute trial and store data for run A
        G = graphs[graph_index].copy()
        G,sys_state = nv.force_state(G,start_state)
        stable_state,sys_state,SN_vals,C_vals,size_vals,adj1 = nv.check_sufficiency_mod(G,sys_state,posweight=4,ind=off_list_A,TRANS_TYPE=1, FORCE_EXTINCT=True,extra_data=True)      #determine the resulting stable configuration

        inner_data_A['start_state'].append(start_state)                                                       
        inner_data_A['stable_state'].append(stable_state)
        inner_data_A['sys_state'].append(sys_state)
        inner_data_A['SN'].append(SN_vals)                                         
        inner_data_A['C'].append(C_vals)
        inner_data_A['size'].append(size_vals)
        inner_data_A['benchmark']=len(size_vals)

        # execute trial and store data for run B   
        G = graphs[graph_index].copy()
        G,sys_state = nv.force_state(G,start_state)        
        stable_state,sys_state,SN_vals,C_vals,size_vals,adj1 = nv.check_sufficiency_mod(G,sys_state,posweight=4,ind=off_list_B,TRANS_TYPE=1, FORCE_EXTINCT=True,extra_data=True)      #determine the resulting stable configuration        

        inner_data_B['start_state'].append(start_state)                                                       
        inner_data_B['stable_state'].append(stable_state)
        inner_data_B['sys_state'].append(sys_state)
        inner_data_B['SN'].append(SN_vals)                                         
        inner_data_B['C'].append(C_vals)
        inner_data_B['size'].append(size_vals)
        inner_data_B['benchmark']=len(size_vals)

        #Finally, append this data to the master dictionary
        data[10*PBS_ITERATOR+graph_index][start_index]=[inner_data_A.copy(),inner_data_B.copy()]

# Write to file ----------------------------------------------------------------
f = open(r'attractors_'+str(PBS_ITERATOR)+'.data','wb')
#f = open(r'/path/to/attractors_'+str(PBS_ITERATOR)+'.data','wb')
pickle.dump(data,f)
f.close()

TIME2 = time.time()
print('Script completed in {0:.2f} minutes.'.format(((TIME2-TIME1)/60)))

