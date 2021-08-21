"""
Here we analyze the output of invasions_whole.py and invasions_random.py: 
we have attractors that correspond to starting from a combination of states. 


Master workflow:
    1. What are the species interaction graphs? (create_graphs.py)
    2. What are the stable communities in these graphs, if we segregate them
       in to two "half communities" to enforce isolated communities that
       nonetheless have inter-community interactions for when invasions occur? 
       (find_attractors.py)
    3. What happens if two stable communities are combined and allowed to
        restabilize? (invasions_whole.py and invasions_random.py)
    4. What do the results from (3) look like? (THIS FILE)

The file is structured so the pre-processing near the top (to about line 360)
is saved to file. The top can then be commented out and the final analysis
can be run to load those results, generate figures, run stats, etc.

Some pre-publication figures are included for benefit of anyone who gets
this far and would like to start playing around with their own analysis.

Python 3.x 

Colin Campbell
campbeco@mountunion.edu  
"""

import pickle
import glob
import numpy as np
import function_library as nv
import matplotlib.pyplot as plt
import time
from scipy.stats import spearmanr, ks_2samp

TIME0 = time.time()


# Run analysis ===============================================================

# # Load in graphs
# #Local 
# f = open(r'graphlist.data','rb')   

# graphs = pickle.load(f)
# f.close()

# # Stable community invasions --------------------
# s_overlap = []    # compare source state (stable community + invaders) to stable state
# s_start_size = [] # how many species in the start state?
# s_end_size = []   # how many species in the end attractor (SS or LC superset)?
# s_delta_size = []
# s_nestedness = []
# s_connectance = []
# s_invasive_species = []
# s_invasive_species_end = []
# s_nestedness_final = []
# s_connectance_final = []
# s_native_survive = []
# s_dbl_positive = []
# s_exception_count = 0


# for fname in glob.glob(r'attractors_comb_whole_[0-9]*.data'):
#     # Open the file for this batch of interaction networks
#     f = open(fname,'rb')
#     data = pickle.load(f)
#     f.close()
    
#     PBS_ITERATOR = int(fname[fname.rfind('_')+1:fname.rfind('.')])

#     TIME1 = time.time()
#     print('Whole invasion, starting file {0} at {1:.2f} minutes elapsed.'.format(PBS_ITERATOR,(TIME1-TIME0)/60))
    
#     for graph_nmbr in data.keys():
#         graph_number = graph_nmbr - 10*PBS_ITERATOR
        
#         # Load up this graph so we can look at species-species interactions below
#         graph = graphs[graph_number]
        
#         for counter in data[graph_nmbr].keys():
#             # pull out parent states and final state
#             start_AB = data[graph_nmbr][counter][0]['start_state'][0]
#             start_BA = data[graph_nmbr][counter][1]['start_state'][0]
                        
#             end_AB = data[graph_nmbr][counter][0]['stable_state']
#             end_BA = data[graph_nmbr][counter][1]['stable_state']
            
#             #parents could be SS or (more likely) LC, so form superset
#             if type(end_AB[0]) == list:
#                 end_AB = nv.superset(end_AB[0])
#                 # end_AB = end_AB[0][0]         # alternative: choose just 1 state to consider
#             else:
#                 end_AB = end_AB[0]
#             if type(end_BA[0]) == list:
#                 end_BA = nv.superset(end_BA[0])
#                 # end_BA = end_BA[0][0]
#             else:
#                 end_BA = end_BA[0]
            
#             # track fraction of edges that are double positive (mutually beneficial)
#             edge_count_AB = 0
#             edge_count_BA = 0
#             dbl_positive_count_AB = 0
#             dbl_positive_count_BA = 0
#             for edge in graph.edges():

#                 # need to translate from e.g. 'po_1' to position in the state list
#                 # indices 0-49 are plants 0-49 and indices 50-199 are pollinators 0-149
                
#                 # first node
#                 if edge[0][:2] == 'pl': edge0_index = int(edge[0][3:])
#                 else: edge0_index = int(edge[0][3:])+50
                
#                 # second node
#                 if edge[1][:2] == 'pl': edge1_index = int(edge[1][3:])
#                 else: edge1_index = int(edge[1][3:])+50
                
#                 # we only care about edges between present species
#                 if start_AB[edge0_index] == '1' and start_AB[edge1_index] == '1': 
#                     edge_count_AB += 1
#                     # and track seperately the edges that are also double positive
#                     if graph[edge[0]][edge[1]]['data'] == 1.0 and graph[edge[1]][edge[0]]['data'] == 1.0: dbl_positive_count_AB += 1
                
#                 # repeat for BA invasion
#                 if start_BA[edge0_index] == '1' and start_BA[edge1_index] == '1': 
#                     edge_count_BA += 1
#                     if graph[edge[0]][edge[1]]['data'] == 1.0 and graph[edge[1]][edge[0]]['data'] == 1.0: dbl_positive_count_BA += 1
            
#             # once we've gone through all the edges, record the fraction of double positive edges
#             # Because start_state can be a single state from a LC, occasionally we pick up
#             # a state with no edges whatsoever and the ratio here gives a divide-by-zero error. Record as 0 but
#             # keep track of how many 
#             if edge_count_AB == 0:
#                 s_exception_count += 1
#                 s_dbl_positive.append(0)
#             else:
#                 s_dbl_positive.append(dbl_positive_count_AB/edge_count_AB)
#             if edge_count_BA == 0:
#                 s_exception_count += 1
#                 s_dbl_positive.append(0)
#             else: s_dbl_positive.append(dbl_positive_count_BA/edge_count_BA)
            
            
#             # track # of invader species
#             # start
#             A_invaders = start_AB[:25]+start_AB[50:125]
#             B_invaders = start_BA[25:50]+start_BA[125:200]
#             s_invasive_species.append(A_invaders.count('1'))
#             s_invasive_species.append(B_invaders.count('1'))
            
#             # end
#             A_invaders_end = end_AB[:25]+end_AB[50:125]
#             B_invaders_end = end_BA[25:50]+end_BA[125:200]
#             s_invasive_species_end.append(A_invaders_end.count('1'))
#             s_invasive_species_end.append(B_invaders_end.count('1'))
            
#             # compare the starting state (source + invaders) to final stable state            
#             s_overlap.append(nv.overlap(end_AB,start_AB))
#             s_overlap.append(nv.overlap(end_BA,start_BA))
            
#             s_start_size.append(data[graph_nmbr][counter][0]['start_state'][0].count('1'))
#             s_start_size.append(data[graph_nmbr][counter][1]['start_state'][0].count('1'))
            
#             s_end_size.append(end_AB.count('1'))
#             s_end_size.append(end_BA.count('1'))
            
#             s_delta_size.append(100*(s_end_size[-2] - start_AB.count('1'))/start_AB.count('1'))
#             s_delta_size.append(100*(s_end_size[-1] - start_BA.count('1'))/start_BA.count('1'))
            
#             s_nestedness.append(data[graph_nmbr][counter][0]['SN'][0][0])
#             s_nestedness.append(data[graph_nmbr][counter][1]['SN'][0][0])
            
#             s_connectance.append(data[graph_nmbr][counter][0]['C'][0][0])
#             s_connectance.append(data[graph_nmbr][counter][1]['C'][0][0])
            
#             s_nestedness_final.append(data[graph_nmbr][counter][0]['SN'][0][-1])
#             s_nestedness_final.append(data[graph_nmbr][counter][1]['SN'][0][-1])
            
#             s_connectance_final.append(data[graph_nmbr][counter][0]['C'][0][-1])
#             s_connectance_final.append(data[graph_nmbr][counter][1]['C'][0][-1])    
            
#             # track % of native species that make it to final state
#             A_native = start_AB[25:50]+start_AB[125:200]
#             A_end = end_AB[25:50]+end_AB[125:200]

#             B_native = start_BA[:25] + start_BA[:125]
#             B_end = end_BA[:25]+end_BA[:125]
            
#             A_native_cnt = A_native.count('1')
#             A_end_cnt = 0
#             for i in range(len(A_native)):
#                 if A_native[i] == '1' and A_end[i] == '1': A_end_cnt += 1
#             s_native_survive.append(A_end_cnt/A_native_cnt)    
            
#             B_native_cnt = B_native.count('1')
#             B_end_cnt = 0
#             for i in range(len(B_native)):
#                 if B_native[i] == '1' and B_end[i] == '1': B_end_cnt += 1
#             s_native_survive.append(B_end_cnt/B_native_cnt)
            

# print('s exception fraction = {0:.5g}'.format(s_exception_count/(s_exception_count + len(s_dbl_positive))))

# TIME1 = time.time()
# print('Halfway done in {0:.2f} minutes elapsed.'.format((TIME1-TIME0)/60))

# # Random community invasions --------------------
# r_overlap = []    # compare source state (stable community + invaders) to stable state
# r_start_size = [] # how many species in the start state?
# r_end_size = []   # how many species in the end attractor (SS or LC superset)?
# r_delta_size = []
# r_nestedness = []
# r_connectance = []
# r_invasive_species = []
# r_invasive_species_end = []
# r_nestedness_final = []
# r_connectance_final = []
# r_native_survive = []
# r_dbl_positive = []
# r_exception_count = 0

# for fname in glob.glob(r'attractors_comb_random_[0-9]*.data'):
#     # Open the file for this batch of interaction networks
#     f = open(fname,'rb')
#     data = pickle.load(f)
#     f.close()

#     PBS_ITERATOR = int(fname[fname.rfind('_')+1:fname.rfind('.')])

#     TIME1 = time.time()
#     print('Random invasion, starting file {0} at {1:.2f} minutes elapsed.'.format(PBS_ITERATOR,(TIME1-TIME0)/60))
    
#     for graph_nmbr in data.keys():
#         graph_number = graph_nmbr - 10*PBS_ITERATOR

#         # Load up this graph so we can look at species-species interactions below
#         graph = graphs[graph_number]
        
#         for counter in data[graph_nmbr].keys():
#             # pull out parent states and final state
#             start_AB = data[graph_nmbr][counter][0]['start_state'][0]
#             start_BA = data[graph_nmbr][counter][1]['start_state'][0]
                        
#             end_AB = data[graph_nmbr][counter][0]['stable_state']
#             end_BA = data[graph_nmbr][counter][1]['stable_state']
            
#             #parents could be SS or (more likely) LC, so form superset
#             if type(end_AB[0]) == list:
#                 end_AB = nv.superset(end_AB[0])
#                 # end_AB = end_AB[0][0]
#             else:
#                 end_AB = end_AB[0]
#             if type(end_BA[0]) == list:
#                 end_BA = nv.superset(end_BA[0])
#                 # end_BA = end_BA[0][0]
#             else:
#                 end_BA = end_BA[0]

#             # track fraction of edges that are double positive (mutually beneficial)
#             edge_count_AB = 0
#             edge_count_BA = 0
#             dbl_positive_count_AB = 0
#             dbl_positive_count_BA = 0
#             for edge in graph.edges():

#                 # need to translate from e.g. 'po_1' to position in the state list
#                 # indices 0-49 are plants 0-49 and indices 50-199 are pollinators 0-149
                
#                 # first node
#                 if edge[0][:2] == 'pl': edge0_index = int(edge[0][3:])
#                 else: edge0_index = int(edge[0][3:])+50
                
#                 # second node
#                 if edge[1][:2] == 'pl': edge1_index = int(edge[1][3:])
#                 else: edge1_index = int(edge[1][3:])+50
                
#                 # we only care about edges between present species
#                 if start_AB[edge0_index] == '1' and start_AB[edge1_index] == '1': 
#                     edge_count_AB += 1
#                     # and track seperately the edges that are also double positive
#                     if graph[edge[0]][edge[1]]['data'] == 1.0 and graph[edge[1]][edge[0]]['data'] == 1.0: dbl_positive_count_AB += 1
                
#                 # repeat for BA invasion
#                 if start_BA[edge0_index] == '1' and start_BA[edge1_index] == '1': 
#                     edge_count_BA += 1
#                     if graph[edge[0]][edge[1]]['data'] == 1.0 and graph[edge[1]][edge[0]]['data'] == 1.0: dbl_positive_count_BA += 1
            
#             # once we've gone through all the edges, record the fraction of double positive edges
#             # Because start_state can be a single state from a LC, occasionally we pick up
#             # a state with no edges whatsoever and the ratio here gives a divide-by-zero error. Just skip
#             # and keep track of how many of these there are            
#             if edge_count_AB == 0:
#                 r_exception_count += 1
#                 r_dbl_positive.append(0)
#             else:
#                 r_dbl_positive.append(dbl_positive_count_AB/edge_count_AB)
#             if edge_count_BA == 0:
#                 r_exception_count += 1
#                 r_dbl_positive.append(0)
#             else: r_dbl_positive.append(dbl_positive_count_BA/edge_count_BA)

#             # track # of invader species
#             # start
#             A_invaders = start_AB[:25]+start_AB[50:125]
#             B_invaders = start_BA[25:50]+start_BA[125:200]
#             r_invasive_species.append(A_invaders.count('1'))
#             r_invasive_species.append(B_invaders.count('1'))
            
#             # end
#             A_invaders_end = end_AB[:25]+end_AB[50:125]
#             B_invaders_end = end_BA[25:50]+end_BA[125:200]
#             r_invasive_species_end.append(A_invaders_end.count('1'))
#             r_invasive_species_end.append(B_invaders_end.count('1'))
            
#             # compare the starting state (source + invaders) to final stable state            
#             r_overlap.append(nv.overlap(end_AB,start_AB))
#             r_overlap.append(nv.overlap(end_BA,start_BA))
            
#             r_start_size.append(data[graph_nmbr][counter][0]['start_state'][0].count('1'))
#             r_start_size.append(data[graph_nmbr][counter][1]['start_state'][0].count('1'))
            
#             r_end_size.append(end_AB.count('1'))
#             r_end_size.append(end_BA.count('1'))
            
#             r_delta_size.append(100*(r_end_size[-2] - start_AB.count('1'))/start_AB.count('1'))
#             r_delta_size.append(100*(r_end_size[-1] - start_BA.count('1'))/start_BA.count('1'))
            
#             r_nestedness.append(data[graph_nmbr][counter][0]['SN'][0][0])
#             r_nestedness.append(data[graph_nmbr][counter][1]['SN'][0][0])
            
#             r_connectance.append(data[graph_nmbr][counter][0]['C'][0][0])
#             r_connectance.append(data[graph_nmbr][counter][1]['C'][0][0])

#             r_nestedness_final.append(data[graph_nmbr][counter][0]['SN'][0][-1])
#             r_nestedness_final.append(data[graph_nmbr][counter][1]['SN'][0][-1])
            
#             r_connectance_final.append(data[graph_nmbr][counter][0]['C'][0][-1])
#             r_connectance_final.append(data[graph_nmbr][counter][1]['C'][0][-1])       
            
#             # track % of native species that make it to final state
#             A_native = start_AB[25:50]+start_AB[125:200]
#             A_end = end_AB[25:50]+end_AB[125:200]

#             B_native = start_BA[:25] + start_BA[:125]
#             B_end = end_BA[:25]+end_BA[:125]
            
#             A_native_cnt = A_native.count('1')
#             A_end_cnt = 0
#             for i in range(len(A_native)):
#                 if A_native[i] == '1' and A_end[i] == '1': A_end_cnt += 1
#             r_native_survive.append(A_end_cnt/A_native_cnt)    
            
#             B_native_cnt = B_native.count('1')
#             B_end_cnt = 0
#             for i in range(len(B_native)):
#                 if B_native[i] == '1' and B_end[i] == '1': B_end_cnt += 1
#             r_native_survive.append(B_end_cnt/B_native_cnt)


# print('r exception fraction = {0:.5g}'.format(r_exception_count/(r_exception_count + len(r_dbl_positive))))
            
# TIME1 = time.time()
# print('Completed in {0:.2f} minutes elapsed.'.format((TIME1-TIME0)/60))

# End analysis ===============================================================

# Dump or Load Above Analysis ================================================
# Optionally dump this data to file (then load below to speed up figure tweaks)
# data = [r_overlap, r_start_size, r_end_size, r_delta_size, r_nestedness, r_connectance, r_invasive_species, r_nestedness_final, r_connectance_final, r_native_survive, r_dbl_positive, r_invasive_species_end,  
#         s_overlap, s_start_size, s_end_size, s_delta_size, s_nestedness, s_connectance, s_invasive_species, s_nestedness_final, s_connectance_final, s_native_survive, s_dbl_positive, s_invasive_species_end]
# f = open(r'attractors_analysis_data.data','wb')
# pickle.dump(data,f)
# f.close()

#Load the above (using destination LC superset)
f = open(r'attractors_analysis_data.data','rb')
data = pickle.load(f)
f.close()

r_overlap, r_start_size, r_end_size, r_delta_size, r_nestedness, r_connectance, r_invasive_species, r_nestedness_final, r_connectance_final, r_native_survive, r_dbl_positive, r_invasive_species_end,\
    s_overlap, s_start_size, s_end_size, s_delta_size, s_nestedness, s_connectance, s_invasive_species, s_nestedness_final, s_connectance_final, s_native_survive, s_dbl_positive, s_invasive_species_end = data

# End Dump or Load ============================================================


# Data Processing / Figure Generation =========================================

# # Jaccard Index ---------------------------------
plt.figure(figsize=(8,4))
ax_J1 = plt.subplot(121)
h_J1 = ax_J1.hist2d(s_start_size,s_overlap,bins=40,cmap = 'Purples', cmin = 0, vmin=1, vmax = 6000)
ax_J1.set_xlabel('Amalgamated Community # Species')
ax_J1.set_ylabel('Amalgamated vs. Stable Community Jaccard Index')
ax_J1.set_title('Whole Community Invasion')
# plt.colorbar(h1[3],ax=ax1)

ax_J2 = plt.subplot(122)
h_J2 = ax_J2.hist2d(r_start_size,r_overlap,bins=40,cmap = 'Purples', cmin = 0, vmin=1, vmax = 6000)
ax_J2.set_xlabel('Amalgamated Community # Species')
ax_J2.set_title('Random Species Invasion')

cax_J = plt.axes([0.9,0.146,0.01,0.912-0.146])
plt.colorbar(h_J2[3],cax=cax_J)

plt.subplots_adjust(top=0.912, bottom=0.146, left=0.083, right=0.881, hspace=0.2, wspace=0.139)

# Stats
print('Jaccard:')
print('\tStable Invaders average: {0:.3f}'.format(sum(s_overlap)/len(s_overlap)))
print('\tRandom Invaders average: {0:.3f}'.format(sum(r_overlap)/len(r_overlap)))

# # # Size Change -----------------------------------
plt.figure(figsize=(8,4))
ax_S1 = plt.subplot(121)
h_S1 = ax_S1.hist2d(s_start_size,s_end_size,bins=40,cmap = 'summer', cmin=1)
ax_S1.set_aspect('equal')
ax_S1.plot([0,80],[0,80],'k-')
ax_S1.set_xlim(0,80)
ax_S1.set_ylim(0,80)

ax_S1.set_xlabel('Amalgamated Community # Species')
ax_S1.set_ylabel('Stable Community # Species')

ax_S1.hist2d(s_start_size,s_end_size,bins=40,range = ([0,80],[0,80]), cmap = 'Greens', cmin = 0, vmin=0,vmax=15000)
ax_S1.set_aspect('equal')
ax_S1.plot([0,80],[0,80],'k-')
ax_S1.set_xlim(0,80)
ax_S1.set_ylim(0,80)
ax_S1.set_title('Whole Community Invasion')

ax_S2 = plt.subplot(122)
ax_S2.set_xlabel('Amalgamated Community # Species')
ax_S2.set_aspect('equal')
ax_S2.plot([0,80],[0,80],'k-')
ax_S2.set_xlim(0,80)
ax_S2.set_ylim(0,80)

h_S2 = ax_S2.hist2d(r_start_size,r_end_size,bins=40, range = ([0,80],[0,80]), cmap = 'Greens', cmin = 0, vmin=0,vmax=15000)
ax_S2.set_aspect('equal')
ax_S2.plot([0,80],[0,80],'k-')
ax_S2.set_xlim(0,80)
ax_S2.set_ylim(0,80)
ax_S2.set_title('Random Species Invasion')

cax_S = plt.axes([0.9,0.146,0.01,0.912-0.146])
plt.colorbar(h_S2[3],cax=cax_S)

plt.subplots_adjust(top=0.912, bottom=0.146, left=0.083, right=0.881, hspace=0.2, wspace=0.139)

# Stats
s_size_compare = []
r_size_compare = []
for i in range(len(s_start_size)):
    s_size_compare.append(100*(s_end_size[i]-s_start_size[i])/s_start_size[i])
for i in range(len(r_start_size)):
    r_size_compare.append(100*(r_end_size[i]-r_start_size[i])/r_start_size[i])

print('Size comparison:')
print('\tStable Invaders average % change: {0:.3f}'.format(sum(s_size_compare)/len(s_size_compare)))
print('\tRandom Invaders average % change: {0:.3f}'.format(sum(r_size_compare)/len(r_size_compare)))
    
# # # Connectance and Nestedness vs size --------------------
# plt.figure(figsize=(8,8))
# ax_CN1 = plt.subplot(221)
# h_CN1 = ax_CN1.hist2d(s_nestedness,s_end_size,bins=40,cmap = 'summer', cmin=1, vmin = 1, vmax = 10000)
# ax_CN1.set_xlabel('Amalgamated Community Nestedness')
# ax_CN1.set_ylabel('Stable Community # Species')
# ax_CN1.set_title('Whole Community Invasion')
# ax_CN1.set_ylim(0,80)
# ax_CN1.set_xlim(0,7)

# ax_CN2 = plt.subplot(222)
# h_CN2 = ax_CN2.hist2d(r_nestedness,r_end_size,bins=40,cmap = 'summer', cmin=1, vmin = 1, vmax = 10000 )
# ax_CN2.set_xlabel('Amalgamated Community Nestedness')
# ax_CN2.set_title('Random Species Invasion')
# ax_CN2.set_ylim(0,80)
# ax_CN2.set_xlim(0,7)

# cax_CN12 = plt.axes([0.9,0.4825+0.073,0.01,0.91*(0.956-0.073)/2])
# plt.colorbar(h_CN2[3],cax=cax_CN12)


# ax_CN3 = plt.subplot(223)
# h_CN3 = ax_CN3.hist2d(s_connectance,s_end_size,bins=40,cmap = 'autumn', cmin=1, vmin = 1, vmax = 50000)
# ax_CN3.set_xlabel('Amalgamated Community Connectance')
# ax_CN3.set_ylabel('Stable Community # Species')
# ax_CN3.set_ylim(0,80)
# ax_CN3.set_xlim(0,1)

# ax_CN4 = plt.subplot(224)
# h_CN4 = ax_CN4.hist2d(r_connectance,r_end_size,bins=40,cmap = 'autumn', cmin=1, vmin = 1, vmax = 50000)
# ax_CN4.set_xlabel('Amalgamated Community Connectance')
# ax_CN4.set_ylim(0,80)
# ax_CN3.set_xlim(0,1)

# cax_CN34 = plt.axes([0.9,0.073,0.01,0.91*(0.956-0.073)/2])
# plt.colorbar(h_CN4[3],cax=cax_CN34)

# plt.subplots_adjust(top=0.956, bottom=0.073, left=0.103, right=0.887, hspace=0.198, wspace=0.24)

# # Stats
CN_s1 = spearmanr(s_nestedness, s_end_size)
CN_s2 = spearmanr(r_nestedness, r_end_size)
CN_s3 = spearmanr(s_connectance, s_end_size)
CN_s4 = spearmanr(r_connectance, r_end_size)

print('Nestedness vs. Size Spearman correlation, pval:')
print('\tWhole: {0:.3g} \t {1:.3g}'.format(CN_s1[0],CN_s1[1]))
print('\tRandom: {0:.3g} \t {1:.3g}'.format(CN_s2[0],CN_s2[1]))

print('Connectance vs. Size Spearman correlation, pval:')
print('\tWhole: {0:.3g} \t {1:.3g}'.format(CN_s3[0],CN_s3[1]))
print('\tRandom: {0:.3g} \t {1:.3g}'.format(CN_s4[0],CN_s4[1]))


# # # # Connectance and Nestedness pre vs post ------
# plt.figure(figsize=(8,8))
# plt.subplots_adjust(top=0.956, bottom=0.073, left=0.083, right=0.887, hspace=0.198, wspace=0.178)
# ax_CNb1 = plt.subplot(221)
# h_CNb1 = ax_CNb1.hist2d(s_nestedness,s_nestedness_final,bins=40,cmap = 'summer', cmin=1, vmin = 1, vmax = 20000)
# ax_CNb1.plot([0,0],[7,7],'k-',lw=2)

# ax_CNb1.set_xlabel('Amalgamated Community Nestedness')
# ax_CNb1.set_ylabel('Stable Community Nestedness')
# ax_CNb1.set_title('Whole Community Invasion')
# ax_CNb1.set_ylim(0,7)
# ax_CNb1.set_xlim(0,7)

# ax_CNb2 = plt.subplot(222)
# h_CNb2 = ax_CNb2.hist2d(r_nestedness,r_nestedness_final,bins=40,cmap = 'summer', cmin=1, vmin = 1, vmax = 20000 )
# ax_CNb2.plot([0,0],[7,7],'k-',lw=2)
# ax_CNb2.plot([0,0],[7,7],'k-',lw=2)
# ax_CNb2.set_xlabel('Amalgamated Community Nestedness')
# ax_CNb2.set_title('Random Species Invasion')
# ax_CNb2.set_ylim(0,7)
# ax_CNb2.set_xlim(0,7)

# box = ax_CNb2.get_position() #box.bounds will give (x0, y0, width, height)
# cax_CNb2 = plt.axes([0.9,box.bounds[1],0.01,box.bounds[3]])
# plt.colorbar(h_CNb2[3],cax=cax_CNb2)

# ax_CNb3 = plt.subplot(223)
# ax_CNb3.plot([0,0],[1,1],'k-')
# h_CNb3 = ax_CNb3.hist2d(s_connectance,s_connectance_final,bins=40,cmap = 'autumn', cmin=1, vmin = 1, vmax = 90000)
# ax_CNb3.set_xlabel('Amalgamated Community Connectance')
# ax_CNb3.set_ylabel('Stable Community Connectance')
# # ax_CNb3.set_ylim(-100,1200)

# ax_CNb4 = plt.subplot(224)
# h_CNb4 = ax_CNb4.hist2d(r_connectance,r_connectance_final,bins=40,cmap = 'autumn', cmin=1, vmin = 1, vmax = 90000)
# ax_CNb4.plot([0,0],[1,1],'k-')
# ax_CNb4.set_xlabel('Amalgamated Community Connectance')
# # ax_CNb4.set_ylim(-100,1200)

# box = ax_CNb4.get_position() #box.bounds will give (x0, y0, width, height)
# cax_CNb4 = plt.axes([0.9,box.bounds[1],0.01,box.bounds[3]])
# plt.colorbar(h_CNb4[3],cax=cax_CNb4)

print('Nestedness pre vs. post:')
print('\tWhole\t pre: {0:.3g} \t post: {1:.3g}'.format(sum(s_nestedness)/len(s_nestedness),sum(s_nestedness_final)/len(s_nestedness_final)))
print('\2 sample KS test: {0:.3g}'.format(ks_2samp(s_nestedness,s_nestedness_final)[1]))
print('\tRandom\t pre: {0:.3g} \t post: {1:.3g}'.format(sum(r_nestedness)/len(r_nestedness),sum(r_nestedness_final)/len(r_nestedness_final)))
print('\2 sample KS test: {0:.3g}'.format(ks_2samp(r_nestedness,r_nestedness_final)[1]))

print('Connectance pre vs. post:')
print('\tWhole\t pre: {0:.3g} \t post: {1:.3g}'.format(sum(s_connectance)/len(s_connectance),sum(s_connectance_final)/len(s_connectance_final)))
print('\2 sample KS test: {0:.3g}'.format(ks_2samp(s_connectance,s_connectance_final)[1]))
print('\tRandom\t pre: {0:.3g} \t post: {1:.3g}'.format(sum(r_connectance)/len(r_connectance),sum(r_connectance_final)/len(r_connectance_final)))
print('\2 sample KS test: {0:.3g}'.format(ks_2samp(r_connectance,r_connectance_final)[1]))


# # Size (detailed) -------------------------------
# fig_SD = plt.figure(figsize=(8,9))
# plt.subplots_adjust(top=0.977,bottom=0.065,left=0.1,right=0.89,hspace=0.276,wspace=0.22)

# # top two panels for overall # species vs final % species
# ax_SD1 = plt.subplot(321)
# h_SD1 = ax_SD1.hist2d(s_start_size,s_delta_size,bins=40,cmap = 'summer', cmin=1, vmin = 1, vmax = 35000)
# ax_SD1.set_xlabel('Amalgamated State # Species')
# ax_SD1.set_xlim(0,60)
# ax_SD1.set_ylim(-100,1200)
# # ax_SD1.set_ylabel('Amalgamated to Child Change in # Species (%)')

# ax_SD2 = plt.subplot(322)
# h_SD2 = ax_SD2.hist2d(r_start_size,r_delta_size,bins=40,cmap = 'summer', cmin=1, vmin = 1, vmax = 35000)
# ax_SD2.set_xlabel('Amalgamated State # Species')
# ax_SD2.set_xlim(0,60)
# ax_SD2.set_ylim(-100,1200)

# box = ax_SD2.get_position() #box.bounds will give (x0, y0, width, height)
# cax_SD12 = plt.axes([0.9,box.bounds[1],0.01,box.bounds[3]])
# plt.colorbar(h_SD2[3],cax=cax_SD12)

# # middle two panels change horizontal axes to # invasive species
# ax_SD3 = plt.subplot(323)
# h_SD3 = ax_SD3.hist2d(s_invasive_species,s_delta_size,bins=40,cmap = 'autumn', cmin=1, vmin = 1, vmax = 35000)
# ax_SD3.set_xlabel('# Invasive Species')
# ax_SD3.set_xlim(0,60)
# ax_SD3.set_ylabel('Amalgamated to Stable Change in # Species (%)')
# ax_SD3.set_ylim(-100,1200)

# ax_SD4 = plt.subplot(324)
# h_SD4 = ax_SD4.hist2d(r_invasive_species,r_delta_size,bins=40,cmap = 'autumn', cmin=1, vmin = 1, vmax = 35000)
# ax_SD4.set_xlabel('# Invasive Species')
# ax_SD4.set_xlim(0,60)
# ax_SD4.set_ylim(-100,1200)

# box = ax_SD4.get_position() #box.bounds will give (x0, y0, width, height)
# cax_SD34 = plt.axes([0.9,box.bounds[1],0.01,box.bounds[3]])
# plt.colorbar(h_SD4[3],cax=cax_SD34)

# # bottom two panels change horizontal axes to # host species
s_xvals = [s_start_size[i] - s_invasive_species[i] for i in range(len(s_start_size))]
r_xvals = [r_start_size[i] - r_invasive_species[i] for i in range(len(r_start_size))]

# ax_SD5 = plt.subplot(325)
# h_SD5 = ax_SD5.hist2d(s_xvals,s_delta_size,bins=40,cmap = 'winter', cmin=1, vmin = 1, vmax = 35000)
# ax_SD5.set_xlabel('# Invaded Species')
# ax_SD5.set_xlim(0,60)
# # ax_SD5.set_ylabel('Amalgamated to Child Change in # Species (%)')
# ax_SD5.set_ylim(-100,1200)

# ax_SD6 = plt.subplot(326)
# h_SD6 = ax_SD6.hist2d(r_xvals,r_delta_size,bins=40,cmap = 'winter', cmin=1, vmin = 1, vmax = 35000)
# ax_SD6.set_xlabel('# Invaded Species')
# ax_SD6.set_xlim(0,60)
# ax_SD6.set_ylim(-100,1200)

# box = ax_SD6.get_position() #box.bounds will give (x0, y0, width, height)
# cax_SD56 = plt.axes([0.9,box.bounds[1],0.01,box.bounds[3]])
# plt.colorbar(h_SD6[3],cax=cax_SD56)

# # Stats
SD_s1 = spearmanr(s_start_size, s_delta_size)
SD_s2 = spearmanr(r_start_size, r_delta_size)
SD_s3 = spearmanr(s_invasive_species, s_delta_size)
SD_s4 = spearmanr(r_invasive_species, r_delta_size)
SD_s5 = spearmanr(s_xvals, s_delta_size)
SD_s6 = spearmanr(r_xvals, r_delta_size)

print('Size (detailed) Spearman correlation, pval:')
print('\tpanel 1: {0:.3g} \t {1:.3g}\tpanel 2: {2:.3g} \t {3:.3g}'.format(SD_s1[0],SD_s1[1],SD_s2[0],SD_s2[1]))
print('\tpanel 3: {0:.3g} \t {1:.3g}\tpanel 4: {2:.3g} \t {3:.3g}'.format(SD_s3[0],SD_s3[1],SD_s4[0],SD_s4[1]))
print('\tpanel 5: {0:.3g} \t {1:.3g}\tpanel 6: {2:.3g} \t {3:.3g}'.format(SD_s5[0],SD_s5[1],SD_s6[0],SD_s6[1]))


# # Size (detailed v2) --------------------------
# # same vertical axis as earlier Size figure
# fig_SD = plt.figure(figsize=(8,9))
# plt.subplots_adjust(top=0.977,bottom=0.065,left=0.1,right=0.89,hspace=0.276,wspace=0.22)

# # top two panels for overall # species vs final % species
# ax_SD1 = plt.subplot(321)
# h_SD1 = ax_SD1.hist2d(s_start_size,s_end_size,bins=40,cmap = 'summer', cmin=1, vmin = 1, vmax = 15000)
# ax_SD1.set_xlabel('Amalgamated State # Species')
# ax_SD1.set_xlim(0,80)
# ax_SD1.set_ylim(0,80)
# ax_SD1.plot([0,80],[0,80],'k-')
# # ax_SD1.set_ylabel('Amalgamated to Child Change in # Species (%)')

# ax_SD2 = plt.subplot(322)
# h_SD2 = ax_SD2.hist2d(r_start_size,r_end_size,bins=40,cmap = 'summer', cmin=1, vmin = 1, vmax = 15000)
# ax_SD2.set_xlabel('Amalgamated State # Species')
# ax_SD2.set_xlim(0,80)
# ax_SD2.set_ylim(0,80)
# ax_SD2.plot([0,80],[0,80],'k-')

# box = ax_SD2.get_position() #box.bounds will give (x0, y0, width, height)
# cax_SD12 = plt.axes([0.9,box.bounds[1],0.01,box.bounds[3]])
# plt.colorbar(h_SD2[3],cax=cax_SD12)

# # middle two panels change horizontal axes to # invasive species
# ax_SD3 = plt.subplot(323)
# h_SD3 = ax_SD3.hist2d(s_invasive_species,s_end_size,bins=40,cmap = 'autumn', cmin=1, vmin = 1, vmax = 8000)
# ax_SD3.set_xlabel('# Invasive Species')
# ax_SD3.set_xlim(0,80)
# ax_SD3.set_ylabel('Stable Community # Species')
# ax_SD3.set_ylim(0,80)
# ax_SD3.plot([0,80],[0,80],'k-')

# ax_SD4 = plt.subplot(324)
# h_SD4 = ax_SD4.hist2d(r_invasive_species,r_end_size,bins=40,cmap = 'autumn', cmin=1, vmin = 1, vmax = 8000)
# ax_SD4.set_xlabel('# Invasive Species')
# ax_SD4.set_xlim(0,80)
# ax_SD4.set_ylim(0,80)
# ax_SD4.plot([0,80],[0,80],'k-')

# box = ax_SD4.get_position() #box.bounds will give (x0, y0, width, height)
# cax_SD34 = plt.axes([0.9,box.bounds[1],0.01,box.bounds[3]])
# plt.colorbar(h_SD4[3],cax=cax_SD34)

# # bottom two panels change horizontal axes to # host species
s_xvals = [s_start_size[i] - s_invasive_species[i] for i in range(len(s_start_size))]
r_xvals = [r_start_size[i] - r_invasive_species[i] for i in range(len(r_start_size))]

# ax_SD5 = plt.subplot(325)
# h_SD5 = ax_SD5.hist2d(s_xvals,s_end_size,bins=40,cmap = 'winter', cmin=1, vmin = 1, vmax = 8000)
# ax_SD5.set_xlabel('# Invaded Species')
# ax_SD5.set_xlim(0,80)
# # ax_SD5.set_ylabel('Amalgamated to Child Change in # Species (%)')
# ax_SD5.set_ylim(0,80)
# ax_SD5.plot([0,80],[0,80],'k-')

# ax_SD6 = plt.subplot(326)
# h_SD6 = ax_SD6.hist2d(r_xvals,r_end_size,bins=40,cmap = 'winter', cmin=1, vmin = 1, vmax = 8000)
# ax_SD6.set_xlabel('# Invaded Species')
# ax_SD6.set_xlim(0,80)
# ax_SD6.set_ylim(0,80)
# ax_SD6.plot([0,80],[0,80],'k-')

# box = ax_SD6.get_position() #box.bounds will give (x0, y0, width, height)
# cax_SD56 = plt.axes([0.9,box.bounds[1],0.01,box.bounds[3]])
# plt.colorbar(h_SD6[3],cax=cax_SD56)

# # Stats
SD_s1 = spearmanr(s_start_size, s_end_size)
SD_s2 = spearmanr(r_start_size, r_end_size)
SD_s3 = spearmanr(s_invasive_species, s_end_size)
SD_s4 = spearmanr(r_invasive_species, r_end_size)
SD_s5 = spearmanr(s_xvals, s_end_size)
SD_s6 = spearmanr(r_xvals, r_end_size)

print('Size (detailed v2) Spearman correlation, pval:')
print('\tpanel 1: {0:.3g} \t {1:.3g}\tpanel 2: {2:.3g} \t {3:.3g}'.format(SD_s1[0],SD_s1[1],SD_s2[0],SD_s2[1]))
print('\tpanel 3: {0:.3g} \t {1:.3g}\tpanel 4: {2:.3g} \t {3:.3g}'.format(SD_s3[0],SD_s3[1],SD_s4[0],SD_s4[1]))
print('\tpanel 5: {0:.3g} \t {1:.3g}\tpanel 6: {2:.3g} \t {3:.3g}'.format(SD_s5[0],SD_s5[1],SD_s6[0],SD_s6[1]))



# % of species that survive from native community
# plt.figure()
# ax_ss1 = plt.subplot(121)
# ax_ss1.hist(s_native_survive)
# ax_ss1.set_xlabel('Whole')

# ax_ss2 = plt.subplot(122)
# ax_ss2.hist(r_native_survive)
# ax_ss2.set_xlabel('Random')

print('Native species survival:')
print('\t Whole Community: {0:.3g}'.format(sum(s_native_survive)/len(s_native_survive)))
print('\t Random Community: {0:.3g}'.format(sum(r_native_survive)/len(r_native_survive)))

# # Fraction of double-positive edges vs. size of stable community
# fig_DPS = plt.figure()
# # plt.subplots_adjust(top=0.977,bottom=0.065,left=0.1,right=0.89,hspace=0.276,wspace=0.22)


# ax_DPS1 = plt.subplot(121)
# h_DPS1 = ax_DPS1.hist2d([100*x for x in s_dbl_positive],s_end_size,bins=40,cmap = 'summer', cmin=1, vmin = 1, vmax = 15000)

# ax_DPS1.set_xlabel('Mutualistic Interactions (%)')
# ax_DPS1.set_ylabel('Stable Community # Species')
# ax_DPS1.set_title('Whole Community Invasion')
# # ax_DPS1.set_xlim(0,80)
# # ax_DPS1.set_ylim(0,80)

# ax_DPS2 = plt.subplot(122)
# h_DPS2 = ax_DPS2.hist2d([100*x for x in r_dbl_positive],r_end_size,bins=40,cmap = 'summer', cmin=1, vmin = 1, vmax = 15000)

# ax_DPS2.set_xlabel('Mutualistic Interactions (%)')
# ax_DPS2.set_title('Random Species Invasion')

# # ax_DPS1.set_xlim(0,80)
# # ax_DPS1.set_ylim(0,80)

DPS_s1 = spearmanr(s_dbl_positive, s_end_size)
DPS_s2 = spearmanr(r_dbl_positive, r_end_size)

print('Mutualisms vs. end size; Spearman correlation, pval:')
print('\tWhole Community: {0:.3g}'.format(DPS_s1[0]))
print('\Random Species: {0:.3g}'.format(DPS_s2[0]))

DPJ_s1 = spearmanr(s_dbl_positive, s_overlap)
DPJ_s2 = spearmanr(r_dbl_positive, r_overlap)

print('Mutualisms vs. Jaccard; Spearman correlation, pval:')
print('\tWhole Community: {0:.3g}'.format(DPS_s1[0]))
print('\Random Species: {0:.3g}'.format(DPS_s2[0]))

# Balance of native vs. invasive species pre vs post ------
s_balance_pre = []
s_balance_post = []
s_exception_count = 0
for i in range(len(s_invasive_species)):
    native_start = s_start_size[i] - s_invasive_species[i]
    inv_start = s_invasive_species[i]
    
    native_end = s_end_size[i] - s_invasive_species_end[i]
    inv_end = s_invasive_species_end[i]
    
    
    if native_start == 0 or inv_start == 0 or native_end == 0 or inv_end == 0:
        s_exception_count += 1
        continue
    
    if native_start >= inv_start:
        # just report the ratio as 1, 1.5, 2, or whatever, but shift "1" to value
        # of "0" on the axis for ease of plotting
        s_balance_pre.append(native_start/inv_start-1)
    else:
        # transform the ratio so we get equal spacing for "majority inv" and
        # "majority native"; so native/invader = 0.5 is flipped and recorded
        # as 2, then shifted to -1 so 0.5 is the same distance from 1:1 (plotted at 0) 
        # as a ratio of 2
        s_balance_pre.append(-inv_start/native_start + 1)
    
    # repeat for ending
    if native_end >= inv_end:
        s_balance_post.append(native_end/inv_end-1)
    else:
        s_balance_post.append(-inv_end/native_end + 1)    

print('Species balance, whole community exception count = {0} vs. {1} plotted.'\
      .format(s_exception_count,len(s_balance_pre) + len(s_balance_post)))

# Repeat all of above for random invasion
r_balance_pre = []
r_balance_post = []
r_exception_count = 0
for i in range(len(r_invasive_species)):
    native_start = r_start_size[i] - r_invasive_species[i]
    inv_start = r_invasive_species[i]
    
    native_end = r_end_size[i] - r_invasive_species_end[i]
    inv_end = r_invasive_species_end[i]
    
    
    if native_start == 0 or inv_start == 0 or native_end == 0 or inv_end == 0:
        s_exception_count += 1
        continue
    
    if native_start >= inv_start:
        # just report the ratio as 1, 1.5, 2, or whatever, centered so 1:1 is at 0
        r_balance_pre.append(native_start/inv_start - 1)
    else:
        # transform the ratio so we get equal spacing for "majority inv" and
        # "majority native"; so native/invader = 0.5 is flipped and recorded
        # as 2, then shifted to -1 so 0.5 is the same distance from 1 as a ratio
        # of 2
        r_balance_pre.append(-inv_start/native_start + 1)
    
    # repeat for ending
    if native_end >= inv_end:
        r_balance_post.append(native_end/inv_end - 1)
    else:
        r_balance_post.append(-inv_end/native_end + 1)  

print('Species balance, random community exception count = {0} vs. {1} plotted.'\
      .format(r_exception_count,len(r_balance_pre) + len(r_balance_post)))  
        
# Now plot
plt.figure(figsize=(8,4))
ax_bal1 = plt.subplot(121)
h_bal1 = ax_bal1.hist2d(s_balance_pre,s_balance_post,bins=40,cmap = 'Reds', range = [[-6,6],[-6,6]], vmin=1, vmax = 12000)

ax_bal1.plot([-6,6],[-6,6], ls='-', c='k', lw=0.5)
ax_bal1.plot([-6,6],[0,0], ls='-', c='k', lw=0.5)
ax_bal1.plot([0,0],[-6,6], ls='-', c='k', lw=0.5)

ax_bal1.set_xlabel('Amalgamated Community\nNative Species:Invasive Species')
ax_bal1.set_ylabel('Final Stable Community\nNative Species:Invasive Species')

ax_bal1.set_xticks([-6,-4,-2,0,2,4,6])
ax_bal1.set_xticklabels(['1:7','1:5','1:3','1:1','3:1','5:1','7:1'],fontsize='small')

ax_bal1.set_yticks([-6,-4,-2,0,2,4,6])
ax_bal1.set_yticklabels(['1:7','1:5','1:3','1:1','3:1','5:1','7:1'],fontsize='small')

ax_bal1.set_title('Whole Community Invasion')
# plt.colorbar(h1[3],ax=ax1)

ax_bal2 = plt.subplot(122)
h_bal2 = ax_bal2.hist2d(r_balance_pre,r_balance_post,bins=40,cmap = 'Reds', range = [[-6,6],[-6,30]], vmin=1, vmax = 12000)

ax_bal2.plot([-6,30],[-6,30], ls='-', c='k', lw=0.5)
ax_bal2.plot([-6,6],[0,0], ls='-', c='k', lw=0.5)
ax_bal2.plot([0,0],[-6,30], ls='-', c='k', lw=0.5)

ax_bal2.set_xlabel('Amalgamated Community\nNative Species:Invasive Species')
# ax_bal2.set_ylabel('Final Stable Community\nNative Species:Invasive Species')

ax_bal2.set_xticks([-6,-4,-2,0,2,4,6])
ax_bal2.set_xticklabels(['1:7','1:5','1:3','1:1','3:1','5:1','7:1'],fontsize='small')

ax_bal2.set_yticks([-6,0,6,13,20,27])
ax_bal2.set_yticklabels(['1:7','1:1','7:1','14:1','21:1','28:1'],fontsize='small')

ax_bal2.set_title('Random Species Invasion')

cax_bal = plt.axes([0.92,0.16,0.01,0.91-0.16])
plt.colorbar(h_bal2[3],cax=cax_bal)

plt.subplots_adjust(top=0.91,
bottom=0.16,
left=0.11,
right=0.9,
hspace=0.2,
wspace=0.25)

# report correlation b/w dbl positive edges and # of invasive species
DP_s_inv = spearmanr(s_dbl_positive, s_invasive_species)
DP_r_inv = spearmanr(r_dbl_positive, r_invasive_species)

print('Mutualisms vs. Nmbr Invasive Species Spearman correlation, pval:')
print('\tWhole: {0:.3g} \t {1:.3g}'.format(DP_s_inv[0],DP_s_inv[1]))
print('\tRandom: {0:.3g} \t {1:.3g}'.format(DP_r_inv[0],DP_r_inv[1]))

# Repeat for # native species, but need to generate that list first
s_native = []
r_native = []
for i in range(len(r_invasive_species)):
    s_native.append(s_start_size[i] - s_invasive_species[i])
    r_native.append(r_start_size[i] - r_invasive_species[i])
    
DP_s_nat = spearmanr(s_dbl_positive, s_native)
DP_r_nat = spearmanr(r_dbl_positive, r_native)

print('Mutualisms vs. Nmbr Native Species Spearman correlation, pval:')
print('\tWhole: {0:.3g} \t {1:.3g}'.format(DP_s_nat[0],DP_s_nat[1]))
print('\tRandom: {0:.3g} \t {1:.3g}'.format(DP_r_nat[0],DP_r_nat[1]))


# Final draw command ----------------------------
plt.show()
