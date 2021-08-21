'''
Generates given # of species interaction graphs. 

Python 3.x 

Colin Campbell
campbeco@mountunion.edu
'''

import function_library as nv
import numpy as np
import pickle
import time

def graph_generator(x):
    graph = nv.form_network(plants=50,pollinators=150)
    if x%10==0: 
        T2 = time.time()
        print('Generated graph {0} at {1:.2f} minutes elapsed.'.format(x,(T2-T1)/60.0))
    return graph


T1 = time.time()

graphs=[graph_generator(i) for i in range(1000)]

#Save graphs
f=open('graphlist.data','wb')
pickle.dump(graphs,f)
f.close()

T2 = time.time()

print('{0:.2f} minutes for algorithm to elapse.'.format(((T2-T1)/60)))

