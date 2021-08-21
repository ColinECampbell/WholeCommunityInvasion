'''
This module contains the 'master functions' used to generate and analyze
networks in this project.

Python 3.x 

Colin Campbell
campbeco@mountunion.edu
'''

import numpy as np
import scipy.special as sp
import math

import networkx as nx
from networkx.algorithms import bipartite as bp  # needed for configuration_model


def form_network(plants=10,pollinators=10,\
po_mean=3.5,po_stdev=3.272565,po_skewness=1.43,\
pl_mean=3.5,pl_stdev=2.579218,pl_skewness=.73,\
po_decay=1.273077,po_cutoff=.328301,\
pl_decay=.876667,pl_cutoff=.141464):
    '''
    OVERVIEW:
    Forms a bipartite network of plants and pollinators. Plants and pollinators
    are:
    1. Assigned characteristic proboscis/stem lengths from a skewed normal
       distribution.
    2. Assigned edges (inter-type only) via an exponentially cut off power law
       distribution.
    The edges are then made bidirectional, with weights according to the general
    schematic:
        plants prefer prob<=stem
        pollinators prefer prob>=stem
    Edge weights are assigned values of -1, or 1. A plant->pollinator edge is
    assigned -1 if prob<stem (bad), 1 if prob>stem (good), or 1 if
    prob=stem (preferred). We consider prob=stem if the % difference is <=10%.
    The percent difference b/w x,y, is defined as 100*abs(x-y)/avg(x,y).


    INPUT PARAMETERS:
    * pollinators, plants = # of ea. type of species

    * mean, stdev, and skewness = mean, stdev, and skewness of distribution from
      which the species' characteristic lengths will be drawn. alpha = 0 for
      normal dist, - (+) for left (right) skew.

    The decay function used for assigning edges b/w plants and pollinators is:
    p(x) = x^(-decay) * exp(-cutoff*x)
    * decay = characteristic decay rate for power law
    * cutoff = exponential cutoff term

    Default parameter values are drawn from:
    * Stang et al, Annals of Botany 103: 1459-1469, 2009 (for length
    distributions)
    * Jordano et al, Ecology Letters 6: 69-81, 2003 (for decay function)

    PROCESS:
    1. Generate range of values from the specified distributions.
    2. Assign proboscis/stem lengths according to skewed normal distribution.
    3. Assign edges b/w pollinators and plants according to cut off exponential
       distribution.
    4. Make edges bidirectional, and assign weights according to a 3-valued
       fitting function.
    '''

    #1. Generate range of values from specified distributions
    def cdf_skew(X,sgn):
        '''
        Returns the cumulative density function of the skew normal distribution.
        '''
        def dens(X,sgn):
            '''
            Returns the probability density function of the skew normal distribution.
            See http://en.wikipedia.org/wiki/Skew_normal_distribution for details.
            sp.ndtr returns the area under the standard Gaussian probability density
            function, integrated from minus infinity to (arg).
            '''
            if sgn=='po':
                #Compute scaling parameters from supplied information (see word doc for derivation)
                po_skew = min([0.99,po_skewness]) # For lognormal the upper limit is 1; must assume sampling bias if value exceeds
                arg1=2*po_skew/(4-math.pi)
                delta=math.sqrt(math.pi/2.0)*arg1**(1/3)/((1+arg1**(2/3))**.5)
                omega=math.sqrt((math.pi*po_stdev*po_stdev)/(math.pi-2*delta*delta))
                alpha=delta/(math.sqrt(1-delta*delta))
                epsilon=po_mean-omega*delta*math.sqrt(2/math.pi)

                arg2=(X-epsilon)/omega
                Y = (2./po_stdev)*np.exp(-(arg2)**2/2)/np.sqrt(2*np.pi)
                Y *= sp.ndtr(alpha*arg2)                    #The max value should be 1, but for some reason it is near 1.41424
                return Y                                    #This is a normalization issue we account for below (I haven't tracked down the source)
            elif sgn=='pl':
                #Compute scaling parameters from supplied information (see word doc for derivation)
                pl_skew = min([0.99,pl_skewness]) # For lognormal the upper limit is 1; must assume sampling bias if value exceeds
                arg1=2*pl_skew/(4-math.pi)
                delta=math.sqrt(math.pi/2.0)*arg1**(1/3)/((1+arg1**(2/3))**.5)
                omega=math.sqrt((math.pi*pl_stdev*pl_stdev)/(math.pi-2*delta*delta))
                alpha=delta/(math.sqrt(1-delta*delta))
                epsilon=pl_mean-omega*delta*math.sqrt(2/math.pi)

                arg2=(X-epsilon)/omega
                Y = (2./pl_stdev)*np.exp(-(arg2)**2/2)/np.sqrt(2*np.pi)
                Y *= sp.ndtr(alpha*arg2)                    #The max value should be 1, but for some reason it is near 1.41424
                return Y                                    #This is a normalization issue we account for below (I haven't tracked down the source)

        if sgn=='pl':
            Y_pl = dens(X,'pl')
            Y_pl = np.cumsum(Y_pl)*(X[1]-X[0])
            return Y_pl
        elif sgn=='po':
            Y_po = dens(X,'po')
            Y_po = np.cumsum(Y_po)*(X[1]-X[0])
            return Y_po

    def cutoff_fn(sgn):
        '''For power law/exponential cutoff:
           We generate this with a standard exponential distribution, and force
           the cutoff behavior by accepting each generated value with probability
           p=(x/xmin)^(-cutoff).
           Source for approach:
           http://arxiv.org/PS_cache/arxiv/pdf/0706/0706.1062v2.pdf (pg.40)

           we assume xmin=1. (i.e. the lower cutoff for range where power law
           applies is 1)'''

        if sgn=='pl':
            decay,cutoff=pl_decay,pl_cutoff
        elif sgn=='po':
            decay,cutoff=po_decay,po_cutoff

        #Form exponential values
        Y_cutoff=np.random.exponential(scale=1.0/cutoff,size=10000)
        def cutoff_filter(x):
            p=x**(-decay)
            comparison=np.random.ranf()
            return p>comparison

        #Remove some according to power law test
        Y_cutoff=list(filter(cutoff_filter,Y_cutoff))

        #Fill in empty values
        temp=np.array([])
        while len(temp)<(10000-len(Y_cutoff)):
            x=np.random.exponential(scale=1.0/cutoff)
            p=x**(-decay)
            comparison=np.random.ranf()
            if p>comparison:
                temp=np.append(temp,x)

        Y_cutoff=np.append(Y_cutoff,temp)
        return Y_cutoff

    #Form x range for distributions
    X_pl = np.arange(pl_mean-50, pl_mean+100, 0.1)     #These ranges characterize an appropriate range for the default values
    X_po = np.arange(po_mean-50, po_mean+100, 0.1)     #If nondefault values are used, care should be taken that the leftmost...
                                                        #...value ~0 and the rightmost ~1
    #Form skew normal distribution
    '''
    Procedure for generating values from a skew normal distribution:
    1. Form cdf of skew distribution w/ given params.
       - Interpret values as CDF(x)=y; y=cdf value for input x
    2. Take a value, x1, from uniform distribution[0,1]
    3. Find largest value x2 s.t. CDF(x2)<=x1
    4. x2 is the output from the end distribution.
    '''
    Y_pl,Y_po = cdf_skew(X_pl,'pl'),cdf_skew(X_po,'po')
    Y_pl_skew,Y_po_skew=[],[]
    Y_pl_max=max(Y_pl)                                  #The normalization factors alluded to in the cdf function defn
    Y_po_max=max(Y_po)
    for i in Y_pl:
        x1=np.random.ranf()*(Y_pl_max)                      #If cdf gave a max value of 1
        x2=X_pl[Y_pl.searchsorted(x1)-1]                    # we'd just call np.random.ranf() here
        if x2>0:
            Y_pl_skew+=[x2]
    for i in Y_po:
        x1=np.random.ranf()*(Y_po_max)                      #If cdf gave a max value of 1
        x2=X_po[Y_po.searchsorted(x1)-1]                    #we'd just call np.random.ranf() here
        if x2>0:
            Y_po_skew+=[x2]

    #Form cutoff power law distribution
    pl_cutoff=cutoff_fn('pl')
    po_cutoff=cutoff_fn('po')

    #2. Assign proboscis/stem lengths
    po_lengths=np.arange(0,pollinators)
    pl_lengths=np.arange(0,plants)

    def assign_lengths_pl(x):
        index=np.random.randint(0,len(Y_pl_skew))
        return Y_pl_skew[index]

    def assign_lengths_po(x):
        index=np.random.randint(0,len(Y_po_skew))
        return Y_po_skew[index]

    po_lengths=list(map(assign_lengths_po,po_lengths))
    pl_lengths=list(map(assign_lengths_pl,pl_lengths))

    #3. Assign edges b/w pollinators and plants according to cut off exponential distribution.
    def obtain_sequence(n,cutoff):
        #Obtains degree distributions from the cutoff powerlaw distribution
        out=[]
        for i in range(0,n):
            loc=np.random.randint(0,np.size(cutoff))
            out+=[int(math.ceil(cutoff[loc]))]         #ceiling function rounds up values to nearest int (avoids 0 assigned degree)
        return out

    #Obtain degree sequences from exponential cut off degree distribution
    aseq=obtain_sequence(pollinators,po_cutoff)
    bseq=obtain_sequence(plants,pl_cutoff)
    #Their sums must be equal in order to form graph
    while sum(aseq)!=sum(bseq):
        aseq=obtain_sequence(pollinators,po_cutoff)
        bseq=obtain_sequence(plants,pl_cutoff)
    #Form bipartite graph (networkx documentation claims using Graph() may lead to inexact degree sequences)
    G=bp.configuration_model(aseq, bseq,create_using=nx.Graph())

    #Relabel the nodes...
    names={}
    for i in range(0,pollinators):
        names[i]='po_'+str(i)
    for i in range(0,plants):
        names[i+pollinators]='pl_'+str(i)

    G=nx.relabel_nodes(G,names)
    G=G.to_directed()           #now every edge a-b is duplicated & directed (a->b and b->a)

    #4. Assign edge weights according to fitting function
    #First, assign nodes their type & lengths from po_lengths and pl_lengths
    #(better storage of values now that the network is actually formed)
    counter=0
    for i in po_lengths:
        G.nodes['po_'+str(counter)]['type'] = 'pollinator'
        G.nodes['po_'+str(counter)]['length'] = i
        counter+=1
    counter=0
    for i in pl_lengths:
        G.nodes['pl_'+str(counter)]['type'] = 'plant'
        G.nodes['pl_'+str(counter)]['length'] = i
        counter+=1

    #Then assign edge weight according to the lengths
    for i in G.edges():
        a,b=G.nodes[i[0]]['length'],G.nodes[i[1]]['length']     #prob/stem lengths
        diff=100.*abs(a-b)/((a+b)/2.0)
        #If they're w/in the specified range, both edges (a->b and b->a) are assigned 2
        if diff<=10.0:
            G[i[0]][i[1]]['data']=1.
            G[i[1]][i[0]]['data']=1.
        elif a<b:                                       #length1<length2
            G[i[1]][i[0]]['data']=-1.
            G[i[0]][i[1]]['data']=1.
        else:                                           #length1>length2
            G[i[1]][i[0]]['data']=1.
            G[i[0]][i[1]]['data']=-1.

    #Finally, output the graph
    return G

def force_state(Graph,state_string,mod=False):
    '''
    This function forces the network into the state specified by a binary
    string (e.g. '001110101'), where the first position applies to pl_0 and
    the last to po_m (m=# of pollinators).

    It returns the modified graph with node property 'state' assigned
    according to state_string.

    It is called internally in other functions (e.g. steady_states,
    active_edges).

    the "mod" parameter was added to accomodate transient_analysis, when we
    have removed nodes in a nonuniform manner. leaving mod=False is faster,
    but mod=True allows the function to work appropriately in this case.
    '''
    if mod==False:
        #We iterate first over all plant nodes
        sc=0                                #state counter
        nc=0                                #node counter
        while 1==1:
            node='pl_'+str(nc)
            if Graph.has_node(node):
                Graph.nodes[node]['state']=state_string[sc]
                nc+=1
                sc+=1
            else:
                break

        #And then over pollinator nodes
        nc=0
        while 1==1:
            node='po_'+str(nc)
            if Graph.has_node(node):
                Graph.nodes[node]['state']=state_string[sc]
                nc+=1
                sc+=1
            else:
                break

    else:
        def mapper(x):
            try:
                return int(x[-2:])  #for e.g. pl_15
            except ValueError:
                return int(x[-1])   #for e.g. pl_5
        #Determine maximum label value
        labels=Graph.nodes()
        labels=list(map(mapper,labels))
        nc=max(labels)+1
        sc=0
        for i in range(nc):
            node='pl_'+str(i)
            if Graph.has_node(node):
                Graph.nodes[node]['state']=state_string[sc]
                sc+=1

        #And then over pollinator nodes
        for i in range(nc):
            node='po_'+str(i)
            if Graph.has_node(node):
                Graph.nodes[node]['state']=state_string[sc]
                sc+=1

    #After we've iterated over all nodes, we also should have iterated
    #over all entries in the state string
    if len(state_string)!=sc:
        print(sc, len(state_string),len(Graph.nodes()))
        print(state_string)
        print('error in assigning node states in function \'force_state\'!')

    #We also want a dictionary of the states
    sys_state={}
    for i in Graph.nodes():
        sys_state[i]={}
        sys_state[i][0]=int(Graph.nodes[i]['state'])

    return Graph, sys_state

def update_network(Graph,sys_state,threshold=1,posweight=1,mode='sync',negweight=-1,negspecial=-2,posspecial=4,negnodes=[],extra_data=False):
    '''
    This function performs one timestep on the network. For an edge to be active
    after the update, the total incoming edge weights from active nodes must be
    >= threshold. "posweight" allows for the weight of positive edges to be
    effectively increased, which allows for another variation on update rules
    (instead of just varying the threshold).

    mode is 'sync' by default, which updates all the nodes. The only other mode
    currently supported is 'omit1', which randomly selects 1 node to not update.

    negweight is the weight of a negative edges (recent addition).

    negnodes are nodes which have their weights modified to values indicated
    by 'posspecial' and 'negspecial.' (negnodes is a deprecated name; it
    originally only handled modifications to negative edges.)

    It returns the updated network and an updated dictionary of states.

    This function should not normally need to be called by the end user. It is
    called internally in other functions, which repeatedly update the network
    under some conditions.

    if extra_data==True, the fn performs analysis of spectral nestedness,
    connectance, and size at every time step.
    '''
    def edge_filter(x):
        #used for filtering list of edges to give only edges ending at target
        return x[1]==i
    def state_filter(x):
        #used for filtering list of edges to give only edges with active source node
        return sys_state[x[0]][counter-1]==1 or sys_state[x[0]][counter-1]=='1'
    def weight_map(x):
        #maps list of edges into list of their weights (-1/1), modified to posweight, negweight, and negspecial
        if x[0] in negnodes and x[2]['data']==-1:
            return negspecial                           #negative edges originating from negnode list are changed from negweight to negspecial
        if x[0] in negnodes and x[2]['data']==1:
            return posspecial                           #positive edges originated from negnode list are changed from posweight to posspecial
        if x[2]['data']==-1:
            return negweight                            #negative edges are changed from -1 to negweight
        return posweight                                #positives are increased from 1 to posweight

    counter=list(sys_state.values())
    counter=len(counter[0].values())  #determines which round of updating we're on

    edgelist=Graph.edges(data=True)                 #stores all edges
    if mode=='omit1':
        nodelist=Graph.nodes()                      #if we are to omit updating 1 node, make a list of the nodes
        omit_node=np.random.randint(len(nodelist))   #choose 1 index to omit
    for i in Graph.nodes():
        if mode=='omit1':
            if nodelist.index(i)==omit_node:
                sys_state[i][counter]=Graph.nodes[i]['state']    #don't update graph node dictionary, but make sure we update the state dictionary with the prev. state
                continue                        #don't update this node if we're in omit1 mode
        edges=list(filter(edge_filter,edgelist))      #edges is list of edges ending at i
        edges=list(filter(state_filter,edges))        #edges is list of edges with active source (and ending at i)
        edges=list(map(weight_map,edges))             #edges is list of weights of edges with active source nodes that end at i
        weight=sum(edges)
        if weight<threshold:
            Graph.nodes[i]['state']=0            #(all t+1 values are determined from t; NOT running update)
            sys_state[i][counter]=0
        else:
            Graph.nodes[i]['state']=1
            sys_state[i][counter]=1

    if extra_data:
        G_stripped = Graph.copy()
        for i in Graph.__iter__():
            if Graph.nodes[i]['state']==0:
                G_stripped.remove_node(i)           #for final analysis, remove absent nodes (analyze only 'existing' graph) if we're doing extra analysis here
        if nx.number_of_nodes(G_stripped) == 0:
            SN=False
            C=False
        else:
            SN = spectral_nestedness(G_stripped)
            C = nx.density(G_stripped)
        return Graph, sys_state, [SN,C,nx.number_of_nodes(G_stripped),[Graph.nodes[x]['state'] for x in Graph.nodes()]]
    else:
        return Graph, sys_state

def spectral_nestedness(G):
    A = nx.adjacency_matrix(G)                                                  # Must form an adjacency matrix based off of the interactions
    A = A.todense()
    Evals,Evects = np.linalg.eig(A)
    SR = max([abs(x.real) for x in Evals])                                      # Find the spectral radius of this graph
    return SR

def check_steady(Graph,sys_state,mode='sync'):
    '''
    Takes dictionary of states of form:
    {node1:{step0: state, step1: state,...} node2:{...},...}
    and sees if the final two states are equivalent (steady state reached). If
    not, it checks to see if the last state is equivalent to any previous state
    (limit cycle). It returns 'steady','limit', or 'chaotic', depending on the
    results.

    mode is 'sync' by default for all nodes being updated (and therefore calling
    a LC whenever a state is repeated). Currently it also supports 'omit1',
    which introduces asynchronous behavior, and the output of this function
    is always 'chaotic' or a SS.

    This function is called internally in other functions and is not meant for
    use by the end user.
    '''
    def state_mapper(x):
        #maps list of nodes to contain states of the nodes at update #count
        return sys_state[x][count]

    counter=list(sys_state.values())
    counter=len(counter[0].values())  #determines which round of updating we're on

    #First check for steady state
    nodes=Graph.nodes()
    nodes=node_sorter(nodes)            #nodes is now a properly sorted list of node names, e.g. ['pl_0','pl_1',...]
    count=counter-1
    ult=list(map(state_mapper,nodes))         #is list of the most recent states
    count-=1
    p_ult=list(map(state_mapper,nodes))       #is list of the second to most recent states
    if ult==p_ult:
        out=''
        for i in ult:
            out+=str(i)
        return out                      #return binary state vector for the steady state (as a string)

    #If that isn't true, check for limit cycle, but only if we're in 'sync' mode
    if mode=='sync':
        lc=False
        for i in range(0,counter-1):
            count=i
            x=list(map(state_mapper,nodes))
            if x==ult:                      #x is the same state as ult. All states after it (included ult) comprise the limit cycle

                lc=True
                lcstates=[]
            if lc:                          #We store these states
                state=''
                for i in x:
                    state+=str(i)
                lcstates+=[state]
        if lc:
            return lcstates                 #A list of all of the states in the limit cycle. Its length is the # of states in the lc


    #Otherwise, return 'chaotic', indicating the system is still evolving
    return 'chaotic'


def check_sufficiency(Graph,sys_state,steplimit=100,threshold=1,posweight=1,ind=False,TRANS_TYPE=1, FORCE_EXTINCT=False,mode='sync',negweight=-1,negspecial=-2,posspecial=4,negnodes=[],TRANS_LIMIT=0,extra_data=False):
    '''
    MODIFIED FROM OTHER PROJECTS:
    Here, we allow a limit to how long a TRANS_TYPE forced
    introduction/extinction lasts, with variable TRANS_LIMIT set to an int for
    the # of steps (0 -> never). Setting equal to 'inf' makes the changes
    permanent throughout the simulation.

    Note also that FORCE_EXTINCT can refer to forced introductions, as well.

    In this project we additionally output summary statistics of every network
    configuration during the dynamics.

    ----
    This function takes an initialized graph and its dictionary of states and
    updates it 'in stasis' (w/o externally introduced species) until it reaches
    a steady state, limit cycle, or 100 iterations have occurred.

    threshold is what the total edge weight of incoming edges with active source
    nodes must meet or exceed for a node to remain active after that update
    round.

    posweight is the weight of a single positive edge.

    ind is an int that corresponds to the position in the state string that will
    be forced to remain ON or OFF (based on TRANS_TYPE; 0 for 0->1 perturbations
    and 1 for 1->0 perturbations.) ind is False if no such forcing is to occur.

    It returns which condition is met and the sys_state dictionary.
    '''
    SN_out,C_out,size_out,adj_out=[],[],[],[]
    if steplimit>2**20:
        steplimit=2**20
    for i in range(steplimit):      #Counting initialization, yields total of steplimit+1 states
        #force a particular species to be ON or OFF prior to first update if we're forcing a particular type of perturbation
        if i==0 and (ind!=False or type(ind)==int):     #(0==False returns True, so account for that)
            if TRANS_TYPE==1:       #what species will be perturbed to (opposite of what they are perturbed from, i.e. TRANS_TYPE)
                target=0
            else:
                target=1
            nodes=Graph.nodes()
            nodes=node_sorter(nodes)
            end=sys_state[nodes[ind]].keys()
            sys_state[nodes[ind]][end[-1]]=target
        #perform update once we've made the perturbation
        if extra_data:
            Graph,sys_state,stats=update_network(Graph=Graph,sys_state=sys_state,threshold=threshold,posweight=posweight,mode=mode,negweight=negweight,negspecial=negspecial,posspecial=posspecial,negnodes=negnodes,extra_data=True)
            SN_out+=[stats[0]]
            C_out+=[stats[1]]
            size_out+=[stats[2]]
            adj_out+=[stats[3]]
        else:
            Graph,sys_state=update_network(Graph=Graph,sys_state=sys_state,threshold=threshold,posweight=posweight,mode=mode,negweight=negweight,negspecial=negspecial,posspecial=posspecial,negnodes=negnodes)
        #force the specified node to remain ON/OFF if FORCE_EXTINCT==True
        if (ind!=False or type(ind)==int) and FORCE_EXTINCT==True:
            if TRANS_LIMIT=='inf' or i<TRANS_LIMIT:            #Here, only perform forcing if we're at a tolerable time step (always if TRANS_LIMIT = 'inf' , otherwise only for specified #)
                end=sys_state[nodes[ind]].keys()      # Python 3 updated needed; see modification to check_sufficiency_mod below
                sys_state[nodes[ind]][end[-1]]=target

        cond=check_steady(Graph=Graph,sys_state=sys_state,mode=mode)
        if cond!='chaotic':         #Stop evolving the system if we've reached a steady state or limit cycle
            break
    if extra_data:
        return cond, sys_state, SN_out, C_out, size_out,adj_out
    else:
        return cond, sys_state

def check_sufficiency_mod(Graph,sys_state,steplimit=100,threshold=1,posweight=1,ind=False,TRANS_TYPE=1, FORCE_EXTINCT=False,mode='sync',negweight=-1,negspecial=-2,posspecial=4,negnodes=[],TRANS_LIMIT='inf',extra_data=False):
    '''
    MODIFIED VERSION OF check_sufficiency from netev2.py. Here we modify ind to
    take a list of vals.

    Also different from other versions of this fn (in other projects), we allow
    a limit to how long a TRANS_TYPE forced introduction/extinction lasts, with
    variable TRANS_LIMIT set to an int for the # of steps. setting equal to 0
    makes the changes permanent throughout the simulation.

    Note also that FORCE_EXTINCT can refer to forced introductions, as well.

    --
    This function takes an initialized graph and its dictionary of states and
    updates it 'in stasis' (w/o externally introduced species) until it reaches
    a steady state, limit cycle, or 100 iterations have occurred.

    threshold is what the total edge weight of incoming edges with active source
    nodes must meet or exceed for a node to remain active after that update
    round.

    posweight is the weight of a single positive edge.

    ind is an int that corresponds to the position in the state string that will
    be forced to remain ON or OFF (based on TRANS_TYPE; 0 for 0->1 perturbations
    and 1 for 1->0 perturbations.) ind is False if no such forcing is to occur.

    It returns which condition is met and the sys_state dictionary.
    '''
    if TRANS_TYPE==1:
        target=0
    else:
        target=1
    SN_out,C_out,size_out,adj_out=[],[],[],[]
    if steplimit>2**20:
        steplimit=2**20
    for i in range(steplimit):      #Counting initialization, yields total of steplimit+1 states
        #force a particular species to be ON or OFF prior to first update if we're forcing a particular type of perturbation
        if i==0 and (ind!=False or type(ind)==int):     #(0==False returns True, so account for that)
            if TRANS_TYPE==1:       #what species will be perturbed to (opposite of what they are perturbed from, i.e. TRANS_TYPE)
                target=0
            else:
                target=1
        nodes=Graph.nodes()
        nodes=node_sorter(nodes)
        for j in ind:
            end=sorted(sys_state[nodes[j]])     # (edited for Python 3) sorted list of dict keys: 0,1,2,... to current time step counter value
            sys_state[nodes[j]][end[-1]]=target
        #perform update
        if extra_data:
            Graph,sys_state,stats=update_network(Graph=Graph,sys_state=sys_state,threshold=threshold,posweight=posweight,mode=mode,negweight=negweight,negspecial=negspecial,posspecial=posspecial,negnodes=negnodes,extra_data=True)
            SN_out+=[stats[0]]
            C_out+=[stats[1]]
            size_out+=[stats[2]]
            adj_out+=[stats[3]]
        else:
            Graph,sys_state=update_network(Graph=Graph,sys_state=sys_state,threshold=threshold,posweight=posweight,mode=mode,negweight=negweight,negspecial=negspecial,posspecial=posspecial,negnodes=negnodes)
        #force the specified node to remain ON/OFF if FORCE_EXTINCT==True
        if ind!=False and FORCE_EXTINCT==True:
            if TRANS_LIMIT=='inf' or i<=TRANS_LIMIT:            #Here, only perform forcing if we're at a tolerable time step (always if TRANS_LIMIT = 0 , otherwise only for specified #)
                for val in ind:
                    end=sorted(sys_state[nodes[val]])
                    sys_state[nodes[val]][end[-1]]=target

        cond=check_steady(Graph=Graph,sys_state=sys_state,mode=mode)
        if cond!='chaotic':         #Stop evolving the system if we've reached a steady state or limit cycle
            break
    if extra_data:
        return cond, sys_state, SN_out, C_out, size_out,adj_out
    else:
        return cond, sys_state

def node_sorter(x):
    '''
    Since our nodes are strings, just calling nodes.sort() will put e.g. 'pl_5'
    after 'pl_10' - this function sorts them properly and returns them in a
    list.
    '''

    def node_plucker(y):
        #filters out nodes not of appropriate length
        return len(y)==mn

    pl=[]
    po=[]
    for i in x:                 #split po and pl into two lists
        if i.count('l')!=0:
            pl+=[i]
        else:
            po+=[i]

    sorter=[]
    while len(pl)>0:
        mn=min(list(map(len,pl)))           #minimum length (i.e. 1 digit, 2 digit,...)
        small=list(filter(node_plucker,pl)) #small now has the nodes of the minimum length
        pl=list(set(pl)-set(small))         #take all of 'small' out of 'pl'
        small.sort()                        #order the nodes in small
        sorter+=small                       #add these to sorter

    while len(po)>0:
        mn=min(list(map(len,po)))               #repeat process for po
        small=list(filter(node_plucker,po))     #should really rewrite this in more condensed fashion
        po=list(set(po)-set(small))
        small.sort()
        sorter+=small

    return sorter



def active_edges(Graph, states):
    '''
    This function takes as input a graph and list of state strings (e.g. as in
    the output of the function steady_states; in form ['00111','11000',...]),
    and returns a list of the edges that are active (i.e. source and target
    nodes are ON) and their edge weights.
    '''

    def state_filter(x):
        #used for filtering list of edges to give only edges with active source  and sink nodes
        return Graph.nodes[x[0]]['state']=='1' and Graph.nodes[x[1]]['state']=='1'

    edgelist=Graph.edges(data=True)
    for state in states:
        print('Considering state vector:',state)
        print('The active edges are:')
        G,gbg=force_state(Graph,state)          #here we don't care about the returned state dictionary
        edges=list(filter(state_filter,edgelist))     #edges holds only edges with active source and target node

        for i in edges:
            print(i[0],'->',i[1],'\t weight=',i[2]['data'])

def perturb_state(Graph,states,steplimit=True,threshold=1, posweight=1, report=False,negweight=-1,negspecial=-2,negnodes=[]):
    '''
    This function takes as input a graph and list of state strings (e.g. as in
    the output of the function steady_states; in form ['00111','11000',...]),
    and for each node in each state:
    1. Turns it  from ON to OFF or vice versa
    2. Allows the dynamics to evolve until either...
       - the state goes to a steady state, OR
       - the state reaches a limit cycle, OR
       - the state evolves the maximum number of times (i.e. goes to a complex
         limit cycle).
    A dictionary of the form
    {state:[(perturbation1,result,stepcount),(perturbation2,result,stepcount)...],...}
    is returned, where pertubations are each of the possible OFF-ON changes to
    each string in 'states', and the results are the strings of the steady
    states, or an int of the limit cycle length, or 'chaotic' otherwise.
    stepcount is the number of steps it takes to reach the limit cycle or
    steady state.

    NOTE: steplimit defaults to the maximum possible (2^N), but may be
    overridden with any positive integer value (as a float or int).
    '''
    def state_modifier(state,i):
        #takes a state string and forces the value at index i to be 1
        out=state[0:i]+'1'+state[i+1:]
        return out

    if report == True:
        print('Executing perturb_state...')
    out={}                          #keys will be entries in 'states'. Each will have 1 value; a list of (modified state,condition) tuples
    N=len(states[0])                #len of state vector (i.e. # of nodes)
    if steplimit==True:
        steplimit=2**N              #The maximum number of steps we can take on the transition graph for each considered steady state
    count=0                         #for printing status updates
    for state in states:            #iterate over all input states
        if report==True:
            print(100.0*count/len(states),'% complete')
        out_list=[]                 #Will be stored as a value for this state in 'out' dictionary
        for i in range(N):          #iterate over all nodes in this state
            if state[i]=='0':                               #If this node is OFF...
                new_state=state[0:i]+'1'+state[i+1:]        #Make a copy where it is ON
            else:                                           #Otherwise it is ON...
                new_state=state[0:i]+'0'+state[i+1:]        #So we make a copy where it is OFF
            G, sys_state=force_state(Graph,new_state)   #Initialize the graph to this modified state
            #Allow the system to advance in time; cond is  int of limit
            #cycle length, 'chaotic', or the steady state string vector e.g. '01101..'
            cond,sys_state=check_sufficiency(G,sys_state,int(steplimit),threshold=threshold, posweight=posweight,negweight=negweight,negspecial=negspecial,negnodes=negnodes)
            state_count=len(sys_state['pl_0'].keys())   #The number of updates the network went through
            if type(cond)==list:                         #If we enter a limit cycle we report the # of steps it takes to get TO the limit cycle
                state_count-=(1+len(cond))              #(subtract additional 1 b/c #edges=#states-1)
            elif type(cond)==str:                       #If we hit a steady state, we report the # of steps it takes to reach the SS
                state_count-=2                          #subtract 1 b/c #edges=#states-1; subtract additional 1 b/c the state dict has the steady state 2x
            out_list+=[(new_state,cond,state_count)]    #Store these results
        out[state]=out_list                                 #Store final set of perturbed states and the result of their evolution
        count+=1
    if report==True:
        print('Perturb_state finished.')

    if report==True:                #If desired, we can output the results in an easy to read format
        print('--------------')
        for i in out.keys():
            print('Reporting perturbations on state %s'%i)
            n=len(out[i])
            for j in range(n):
                #Different formatting for chaotic, limit cycle, or steady state (if steplimit=True, chaotic will never be called)
                if type(out[i][j][1])==list:
                    print('Perturbed state %s \t takes \t %d \t steps to go to a cycle of length: %s'%(out[i][j][0],out[i][j][2],out[i][j][1]))
                elif out[i][j][1]=='chaotic':
                    print('Perturbed state %s \t takes \t %d \t steps and fails to reach a limit cycle or steady state.'%(steplimit,out[i][j][0]))
                else:
                    print('Perturbed state %s \t takes \t %d \t steps to go to a steady state:    %s'%(out[i][j][0],out[i][j][2],out[i][j][1]))
            print('--------------')


    return out

def lc_comparison(LC):
    '''
    This function compares all the states in a given limit cycle, and determines
    how many of the nodes share the same state: x=(# of agreement/# of bits)
    The average of x over all pairs of states in the limit cycle is calculated.

    LC is a LIST of state STRINGS in a
    particular limit cycle. As such, 'x' is calculated for each entry in LC;
    the average value is returned (so, "what is the average homogeneity of the
    states in a limit cycle in this network?").
    '''

    vals=[]
    n=len(LC)
    m=len(LC[0])            #The # of nodes in each state

    for i in range(n):
        for j in range(i+1,n):      #Compare every pair of states (order doesn't matter)
            s1=LC[i]
            s2=LC[j]
            counter=0
            for k in range(m):      #For every bit position...
                if s1[k]==s2[k]:
                    counter+=1      #...determine if the bits from each state agree.
            vals+=[1.0*counter/m]   #Store the rating for each pair...

    out=np.mean(vals)
    return out                      #...and return the average result.



def lc_lc_comparison(lc1,lc2):
    '''
    This function takes two limit cycles and determines an average similarity
    between them. The process is to determine the average expression for each
    species among a given limit cycle's states, and then make a comparison among
    the LCs by abs(LC1_exp-LC2_exp).

    We normalize the results such that an end value of 0 corresponds to exactly
    equal expression levels for all species, and 1 corresponds to exactly
    opposite expression levels for all species.
    '''

    m=len(lc1[0])       #the number of species
    n1,n2=len(lc1),len(lc2)

    avg1,avg2=[0]*m,[0]*m       #holds the average expression (0 vs 1) among the LC states for each input LC

    #determine average expression levels in the LCs
    for i in range(n1):
        for j in range(i+1,n1):
            s1,s2=lc1[i],lc1[j]
            for k in range(m):
                if s1[k]==s2[k]:
                    avg1[k]+=1.0/(math.factorial(n1)/(2*math.factorial(n1-2)))      #each term is one of the possible state comparison sets (n1 choose 2)
    for i in range(n2):
        for j in range(i+1,n2):
            s1,s2=lc2[i],lc2[j]
            for k in range(m):
                if s1[k]==s2[k]:
                    avg2[k]+=1.0/(math.factorial(n1)/(2*math.factorial(n1-2)))

    #now compare those averages
    out=0
    for i in range(m):
        out+=(1.0/m)*abs(avg1[i]-avg2[i])

    return out

def ss_lc_comparison(SS,LC):
    '''
    This function takes a steady state and limit cycle and returns an average
    similarity between them. An average expression level for the lc is
    determined (as in lc_lc_comparision(), and compared to the ss expression
    levels.
    '''
    n=len(LC)
    m=len(LC[0])            #The # of nodes in each state
    avg=[0]*m

    for i in range(n):
        for j in range(i+1,n):      #Compare every pair of states (order doesn't matter)
            s1=LC[i]
            s2=LC[j]
            for k in range(m):
                if s1[k]==s2[k]:
                    avg[k]+=1.0/(math.factorial(n)/(2*math.factorial(n-2)))

    #compare this average LC expression level to the SS...
    out=0
    for i in range(m):
        out+=(1.0/m)*abs(avg[i]-int(SS[i]))

    return out

def ss_ss_comparison(SS1,SS2):
    '''
    This function takes two steady states and performs a bitwise comparison
    of their expression values. The average difference is returned, so a value
    of 0 corresponds to exact agreement (same SS) and a value of 1 correponds
    to exact disagreement (e.g. 000 and 111).
    '''
    n=len(SS1)
    out=0
    for i in range(n):
        out+=(1.0/n)*abs(int(SS1[i])-int(SS2[i]))

    return out

def steady_states(Graph, threshold=1,report=False):
    '''
    This function searches through all the states in a network, and determines
    how many are steady states.

    It outputs as text the # of steady states, and if report=True, it prints
    the states in binary according to the format:
    pl_0,pl_1,...pl_n, po_0,...po_m
    (where there are n, m plants, pollinators).

    NOTE: It generates an error for very large networks. This function is
    not currently used in "official" analysis (i.e. in random_graph_analysis.py).
    '''

    def update():
        '''
        This function begins updating the states of the network. If any node's
        new state is different from its last, the procedure stops and 'False' is
        returned.

        If each node keeps its old state, 'True' is returned.
        '''

        def edge_filter(x):
            #used for filtering list of edges to give only edges ending at target
            return x[1]==i
        def state_filter(x):
            #used for filtering list of edges to give only edges with active source node
            return Graph.nodes[x[0]]['state']=='1'
        def weight_map(x):
            #maps list of edges into list of their weights
            return x[2]['data']

        out=True                                    #returned variable. Becomes False if a state is updated to a new value (ON->OFF or OFF->ON)

        edgelist=Graph.edges(data=True)             #stores all edges
        for i in Graph.nodes():
            edges=list(filter(edge_filter,edgelist))      #edges is list of edges ending at i
            edges=list(filter(state_filter,edges))        #edges is list of edges with active source (and ending at i)
            edges=list(map(weight_map,edges))       #edges is list of weights of edges with active source nodes that end at i
            weight=sum(edges)
            if weight<threshold:                    #node will be updated to OFF
                if Graph.nodes[i]['state']=='1':     #if it is ON, the state has changed and we break out of the loop
                    out=False
                    break
            elif Graph.nodes[i]['state']=='0':       #node will be updated to ON; if it is OFF the state has changed and we break out of the loop
                    out=False
                    break

        return out

    #We now force the state to iteratively be every possible state, and
    #update it to determine whether or not it is a steady state.
    out=[]                                      #Will hold list of state vectors as strings
    N=len(Graph.nodes())                        #Number of nodes
    M=2**N                                      #Number of states
    tracker=0                                   #Counts # of steady states

    for i in range(0,M):                        #Iterate over all states
        if report ==True and i in (M/4.0,M/2.0,3.0*M/4.0):
            print(*100.0*i/M,'% complete! \t')
        state=bin(i)                            #state is binary string with '0b' on left side
        state=state.split('0b',1)               #state is now list ['',x] where x is binary string w/o '0b'
        state=state[1]                          #state is now just binary string
        Z=N-len(state)                          #Z stores the number of 0's to add s.t. state is of appropriate length
        state='0'*Z+state                       #state now has necessary padding of 0's
        H,gbg=force_state(Graph,state)          #Feed this state into the network   (here, we don't care about the returned dictionary)
        result=update()                         #Determine if it is a steady state
        if result==True:
            tracker+=1                          #Store successful hit
            out+=[state]

        if (result and report)==True:           #If desired (i.e. report=True), output state vector of each steady state
            print('steady state found:\t',state)

    if report==True:
        print ('%d steady states found')%tracker#Print # of steady states

    return out                                  #Return a list of their strings

def superset(strings):
    #converts a set of LC states into a superset, e.g. 0110 and 1000 becomes
    #1110
    #used internally in some below fns
    string_length=len(strings[0])
    string_count=len(strings)
    out=''
    for j in range(string_length):
        x=False
        for k in range(string_count):
            if strings[k][j]=='1':
                out+='1'                #if any state is active at this position, so will the superset
                x=True
                break                   #then move on to the next position
        if not x: out+='0'              #otherwise, this position is off
    return out


def Hamming(x,y):
    #returns Hamming distance of two lists with entries = 0 or 1
    if len(x)!=len(y): raise RuntimeError("inputs to Hamming() must be equal length.")
    return len([1 for i in range(len(x)) if x[i]!=y[i]])/float(len(x))

def overlap(x,y):
    # returns intersection/union of two steady states (so 1 == identical presence; 0 == no shared present species)
    if len(x)!=len(y): raise RuntimeError("inputs to overlap() must be equal length.")
    intersect, union = 0,0
    
    for i in range(len(x)):
        if x[i] == y[i] == '1': intersect += 1
        if x[i] == '1' or y[i] == '1': union += 1
    
    return intersect/union
        
