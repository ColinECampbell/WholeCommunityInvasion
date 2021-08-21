"""
Once we have determined the attractors of the "half interaction graphs", we 
here start the system in a state that corresponds to an attractor from half A
invaded by random species from half B (and vice versa -- exhaustive combinations).

The # of random species chosen from the invading half mirrors the number of species
in the attractors from that half, so the output here and of invasions_whole.py are
number controlled.

We allow the system to re-stabilize and capture the resulting attractor along 
with standard network properties (e.g. nestedness).

Master workflow:
    1. What are the species interaction graphs? (create_graphs.py)
    2. What are the stable communities in these graphs, if we segregate them
       in to two "half communities" to enforce isolated communities that
       nonetheless have inter-community interactions for when invasions occur? 
       (find_attractors.py)
    3. What happens if two stable communities are combined and allowed to
        restabilize? (THIS FILE and invasions_whole.py)
        
The file is structured to run either on a local machine or submitted to a 
High-Performance Computing center. See comments.
(This research was performed via the Ohio  Supercomputer Center, osc.edu)
    
Python 3.x 

Colin Campbell
campbeco@mountunion.edu
"""

import pickle
import glob
#import numpy as np
import function_library as nv
from itertools import product
import time
import random
import networkx as nx
import os
import sys

# Function definitions --------------------------------------------------------
def run_invasion(invader_state,host_state,host):
    '''
    Considers two "half interaction graphs": an invading community 
    ('invader_state') from one half invades the other half, which is in state
    'host_state' to start. Parameter 'host' indicates which half (A=0,B=1)
    is the host community.
    
    The system is started from invader_state + host_state and all species
    in the host community + invading species are allowed to interact in
    subsequent dynamics. The stable state is identified and key information
    is stored and returned in a dict.
    '''
    
    #inner dictionary that will hold the data for this iteration
    inner_data={'start_state':[],'stable_state':[],'sys_state':[],
                'SN':[],'C':[],'size':[],'benchmark':0,'adj':[],
                } 
    
    
    G = graphs[start].copy()

    #If either state is a LC, choose a constituent state at random
    if type(invader_state) == str: start_inv = invader_state[:]
    else: 
        invader_state_copy = invader_state.copy()
        random.shuffle(invader_state_copy)
        start_inv = invader_state_copy.pop()
    
    if type(host_state) == str: start_host = host_state[:]
    else:
        host_state_copy = host_state.copy() 
        random.shuffle(host_state_copy)
        start_host = host_state_copy.pop()
    
    # Modification compared to attractors_comb_seg.py: Now randomize invaders
    invader_count = start_inv.count('1')
    if host == 'A':
        possible_invaders = list(range(25,50))+list(range(125,200))
    else:
        possible_invaders = list(range(25))+list(range(50,125))
    random.shuffle(possible_invaders)
    start_inv = possible_invaders[:invader_count]
    # go from list of 'on' indices to state string of 0s and 1s
    start_inv = ''.join(['1' if x in start_inv else '0' for x in range(200)])
    # Below resume the previous procedure
    
    # now combine these starting states
    start_state = ''.join(['0' if start_inv[x] == '0' and start_host[x] == '0' else '1' for x in range(N)])        
    
    # Whichever half is hosting, the species in the -other- half are by default
    # excluded from participating in the dynamics...
    if host == 'A':
        off_list = list(range(25,50))+list(range(125,200))
    elif host == 'B':
        off_list = list(range(25))+list(range(50,125))
    
    #... with the exception of those species that are invading from the other
    # half.
    for position,val in enumerate(start_state):
        if val == '1' and position in off_list:
            off_list.remove(position)
    
    # execute trial and store data
    G,sys_state = nv.force_state(G,start_state)
    stable_state,sys_state,SN_vals,C_vals,size_vals,adj1 = nv.check_sufficiency_mod(G,sys_state,posweight=4,ind=off_list,TRANS_TYPE=1, FORCE_EXTINCT=True,extra_data=True)      #determine the resulting stable configuration

    inner_data['start_state'].append(start_state)                                                       
    inner_data['stable_state'].append(stable_state)
    inner_data['sys_state'].append(sys_state)
    inner_data['SN'].append(SN_vals)                                         
    inner_data['C'].append(C_vals)
    inner_data['size'].append(size_vals)
    inner_data['benchmark']=len(size_vals)

    
    return inner_data

            
# End function definitions ----------------------------------------------------


TIME0 = time.time()

# Which set of interaction graphs are we considering? -------------------------
#PBS_ITERATOR=int(os.environ["PBS_ARRAYID"])       #We use this variable to slice up the job over 100 jobs on OSC. Use: qsub -t 0-99 pbs_script.sh
PBS_ITERATOR=10                                      #Or dummy variable for local debugging

# Optionally do not overwrite existing output files
# if os.path.exists('/to/path/attractors_comb_random_'+str(PBS_ITERATOR)+'.data'):
#     sys.exit("Output file already exists.")
    
# Load graphs from memory -----------------------------------------------------
#Local 
f = open(r'../Python Output/graphlist.data','rb')   
# OSC
#f = open(r'/to/path/graphlist.data','rb')
graphs = pickle.load(f)
f.close()

graphs=graphs[10*PBS_ITERATOR:(10*PBS_ITERATOR)+10]
N = nx.number_of_nodes(graphs[0])

# Local vs OSC two lines below
for fname in glob.glob(r'attractors_'+str(PBS_ITERATOR)+'.data'):
# for fname in glob.glob(r'/to/path/attractors_'+str(PBS_ITERATOR)+'.data'):
    # Open the file for this batch of 10 interaction networks ----------------
    f = open(fname,'rb')
    data = pickle.load(f)
    f.close()
    
    '''
    structure of 'data' is as follows: data[graph_index][start_index]=[inner_data for A, inner_data for B]
    where:
        graph_index is from 0 to 999 (which interaction graph?)
        start_index is from 0 to 99 (which 'choose a starting state' trial)
        and 'inner_data' is the inner dictionary for community half A and B; 
        they notably have keys 'stable state' and 'size'.
    '''
    
    out_data = {}
    

    for start,graph_index in enumerate(data.keys()):
        print('Starting graph...')
        out_data[10*PBS_ITERATOR+graph_index]={}
       
        LC_list_A, LC_list_B = [], []
        LC_set_list_A, LC_set_list_B = [], []
        SS_list_A,  SS_list_B = [], []  
       
        # First just identify the unique attractors for half A, B -------------
        for start_index in data[graph_index].keys():
            stable_state_A = data[graph_index][start_index][0]['stable_state']
            size_A = data[graph_index][start_index][0]['size']
            SN_A = data[graph_index][start_index][0]['SN'][0][-1]
            C_A = data[graph_index][start_index][0]['C'][0][-1]
            
            stable_state_B = data[graph_index][start_index][1]['stable_state']
            size_B = data[graph_index][start_index][1]['size']
            SN_B = data[graph_index][start_index][1]['SN'][0][-1]
            C_B = data[graph_index][start_index][1]['C'][0][-1]
            
            # For A...
            # If this is an attractor we haven't seen yet, add it to the appropriate list
            if type(stable_state_A[0]) == list:
                # we have a LC; need to count a->b->c->a same as b->c->a->b
                # do this by comparing to a separate list of LCs as sets
                # but also store one copy of LC as usual list of strings for further analysis
                x = set(stable_state_A[0])
                if x not in LC_set_list_A: 
                    LC_set_list_A.append(x)
                    LC_list_A.append([stable_state_A[0],SN_A,C_A,size_A])
            elif type(stable_state_A[0])==str and stable_state_A[0] not in SS_list_A and stable_state_A[0].count('1') > 0:
                # capture steady states here
                # other possibility (outside of this if/elif block) is a repeat attractor
                # or 'chaotic' if none is found -- very unlikely
                # or 'all dead' -- possible but not of interest here
                SS_list_A.append([stable_state_A[0],SN_A,C_A,size_A])

            # Repeat For B...
            # If this is an attractor we haven't seen yet, add it to the appropriate list
            if type(stable_state_B[0]) == list:
                # we have a LC; need to count a->b->c->a same as b->c->a->b
                # do this by comparing to a separate list of LCs as sets
                # but also store one copy of LC as usual list of strings for further analysis
                x = set(stable_state_B[0])
                if x not in LC_set_list_B: 
                    LC_set_list_B.append(x)
                    LC_list_B.append([stable_state_B[0],SN_B,C_B,size_B])
            elif type(stable_state_B[0])==str and stable_state_B[0] not in SS_list_B and stable_state_B[0].count('1') > 0:
                # capture steady states here
                # other possibility (outside of this if/elif block) is a repeat attractor
                # or 'chaotic' if none is found -- very unlikely
                # or 'all dead' -- possible but not of interest here
                SS_list_B.append([stable_state_B[0],SN_B,C_B,size_B])

                
        # Now start the network at a state corresponding to the combination of
        # two of the above attractors (A into B, then B into A) and allow 
        # dynamics to proceed; record final state as in find_attractors.py ---
        
        counter = 0
        counter_max = (len(SS_list_A) + len(LC_list_A))*(len(SS_list_B) + len(LC_list_B)) # nested for loop

        counter_p = 0
        for i,j in product(SS_list_A + LC_list_A,SS_list_B+LC_list_B):
            # periodically report on progress
            if 100*counter/counter_max > counter_p:
                TIME1 = time.time()
                print('{0:.2f}% through combination analysis (total = {1:.2g}) at {2:.2f} minutes elapsed.'.format(counter_p,counter_max,(TIME1-TIME0)/60.0))
                counter_p += 10
            
            inner_data_AB = run_invasion(i[0],j[0],host='B')   # A attractor i invades B network in state j
            inner_data_BA = run_invasion(j[0],i[0],host='A')  # B attractor j invades A network in state i
    
            counter += 1
            
            #Finally, append this data to the master dictionary
            out_data[10*PBS_ITERATOR+graph_index][counter]=[inner_data_AB, inner_data_BA,{'SN':[i[1],j[1]],'C':[i[2],j[2]],'size':[i[2],j[2]]}]
        
# Write to file ----------------------------------------------------------------
f = open(r'attractors_comb_random_'+str(PBS_ITERATOR)+'.data','wb')
#f = open(r'/to/path/attractors_comb_random_'+str(PBS_ITERATOR)+'.data','wb')
#pickle.dump(out_data,f)
#f.close()

TIME1 = time.time()
print('Script completed in {0:.2f} minutes.'.format(((TIME1-TIME0)/60)))
