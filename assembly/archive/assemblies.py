import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations 
import numpy as np
import random
random.seed(0)
np.random.seed(0)


class Cortex:
    def __init__(self, areas, p):
        # make sure each area knows that this is the cortex it belongs to
        self.areas = areas
        for area in areas: area.cortex = self

        # connect every two areas with probability p
        graphs = []
        for a1, a2 in combinations(areas, 2):
            g = TwoAreaGraph(a1.graph, a2.graph, p)
            g.draw()
            graphs.append(g.nx_graph)

        # create a graph of all the areas
        self.graph = CompositeGraph(graphs)
        # self.graph.draw()


    def project(self, stimulus, stimulus_area, num_timesteps = 40):
        
        A_tm1 = np.zeros(n) # all the neurons in A start out with 0 values

        total_support = set() # track the total number of fired neurons
        total_support_size = []

        #W_sa = stimulus_area.weights_matrix # how much we decide to weight inputs from the stimulus
        #W_aa = self.areas[area].W_aa # how much we decide to weight the recurrent inputs from the area itself at the last timestep

        # for t in range(num_timesteps):
        #     # project
        #     A_t = W_sa.dot(stimulus) + W_aa.dot(A_tm1) # n vector of synaptic inputes

        #     # cap
        #     if len(np.where(A_t !=0)[0]) > self.k:
        #         indices = np.argsort(A_t)
        #         A_t[indices[:-self.k]]=0 # only get top k of n synaptic inputs
        #     A_t[np.where(A_t != 0.)[0]] = 1.0

        #     current_support = np.where(A_t != 0)[0]
        #     total_support = total_support.union(current_support)
        #     total_support_size.append(len(total_support))
           
        #     # plasticity modifications
        #     for i in np.where(A_t!=0)[0]:
        #         for j in np.where(stimulus!=0)[0]:
        #             W_sa[i,j] *= 1.+self.B

        #     for i in np.where(A_t!=0)[0]:
        #         for j in np.where(A_tm1!=0)[0]:
        #             W_aa[i,j] *= 1.+self.B

        #     # update the t-1 step
        #     A_tm1 = np.copy(A_t)

        # # plt.plot(total_support_size)
        # # plt.xlabel('iterations'); plt.ylabel('total support size')
        # # plt.show()

        # n = self.n

        # # visualize


class Area:
    def __init__(self, n, p, k, B):
        self.cortex = None
        self.neurons = np.zeros(n)

        # graph that tracks edges within this area
        adj_matrix = np.random.binomial(1,p,size=(n,n)).astype("float64")
        self.weights_matrix = adj_matrix # weights matrix starts out as adj matrix, then changes
        self.graph = SingleAreaGraph(adj_matrix, area=self)

    def get_graph(self):
        return self.graph

    def generate_random_stimulus(self):
        # clear all current neuron values
        self.neurons = np.zeros(n)
        # assign k of the n neuron values to 1
        self.neurons[ np.random.permutation(n)[:k] ] = 1.0
        # run the project and cap operations throughout all brain areas
        self.cortex.project(stimulus=self.neurons, stimulus_area=self)

class Graph:        
    def get_adj_matrix(self):
        return self.adj_matrix
    def get_nx_graph(self):
        return self.nx_graph
    
    def draw(self):
        nx.draw(self.nx_graph)
        plt.show()

class SingleAreaGraph(Graph):
    """ Graph of neurons in one area """
    def __init__(self, adj_matrix, area):
        self.area=area
        self.adj_matrix = adj_matrix
        self.nx_graph = nx.convert_matrix.from_numpy_array(self.adj_matrix, create_using=nx.DiGraph)    
    # def update_neuron_activity(self):
    #     nx.set_node_attributes(self.nx_graph, 'activity', self.area.neurons)
    def draw(self):
        self.update_neuron_activity()
        super().draw()

class CompositeGraph(Graph):
    """ Graph of lots of areas """
    def __init__(self, graphs):
        self.nx_graph = nx.compose_all(graphs)

class TwoAreaGraph(Graph):
    """ Graph of neurons connecting two areas """
    def __init__(self, g1, g2, p):
        """
        g1 = graph 1
        g2 = graph 2
        p = probability of connection between a neuron in graph 1 and neuron in graph 2
        """
        
        self.g1 = g1
        self.g2 = g2

        # create an adjacency matrix for this new two area graph
        self.adj_matrix = np.zeros(shape=(2*n,2*n))
        self.adj_matrix[0:n,0:n] = g1.get_adj_matrix() # make the top left corner the existing connections within graph 1
        self.adj_matrix[n:2*n,n:2*n] = g2.get_adj_matrix() # make the bottom right corner the existing connections within graph 2
        self.adj_matrix[0:n,n:2*n] = np.random.binomial(1,p,size=(n,n)).astype("float64") # make the other two corners connections with probability p
        self.adj_matrix[n:2*n,0:n] = np.random.binomial(1,p,size=(n,n)).astype("float64")

        # create the graph from the adjacency matrix
        self.nx_graph = nx.convert_matrix.from_numpy_array(self.adj_matrix, create_using=nx.DiGraph)
       
        # relabel the nodes in g2 so that it matches up with the adjacency matrix
        def mapping(x): return x+n
        H=nx.relabel_nodes(self.g2.nx_graph, mapping, copy=False)

    def draw(self):
        graph = self.nx_graph
        pos = nx.bipartite_layout(graph, range(0,n), align='horizontal') # nx.shell_layout(graph)  # positions for all nodes
        
        # color g1 nodes one color
        # color g2 nodes another color
        # make excited neurons a darker shade than non-excited neurons
        nx.draw_networkx_nodes(graph, pos,
            nodelist=range(0,n),
            node_color='r',
            node_size=100,
            alpha=0.4)
        #c = (self.g2.area.neurons*n+n).astype(int)
        #print(c)
        #print(range(n))
        nx.draw_networkx_nodes(graph, pos,
            nodelist=range(n,2*n),
            node_color=range(n),
            cmap=plt.cm.Blues,
            # vmin=100,
            # vmax=255,
            node_size=100,
            alpha=0.4)

        # color g1 edges one color
        # color g2 edges another color
        nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5, connectionstyle='arc3,rad=0.2')
        nx.draw_networkx_edges(graph, pos, self.g1.get_nx_graph().edges, edge_color='r', width=1.0, alpha=1, connectionstyle='arc3,rad=0.2')
        nx.draw_networkx_edges(graph, pos, self.g2.get_nx_graph().edges, edge_color='b', width=1.0, alpha=1, connectionstyle='arc3,rad=0.2')
        
        plt.show()

if __name__ == "__main__":
    # ---------------------------
    # initialize constants
    # ---------------------------
    n = 9 #int(1e4) # number of neurons in each area (10**7 is num neurons in the medial temporal lobe)
    p = 2e-2 #1e-2 # probability of recurrent/afferent synaptic connective (10**-3 usually)
    k = 3 #100 # max number of firing neurons in an area (10**4 )
    B = .1 #.1 # plasticity coefficient --  weight of synapse is multiplied by (1+B) when fired

    # ---------------------------
    # initialize areas of the cortex
    # ---------------------------
    stimulus_area = Area(n,p,k,B)
    output_area = Area(n,p,k,B)
    c = Cortex([stimulus_area, output_area], p)

    # ---------------------------
    # fire k neurons at random, and propagate out the neuron firings to all connected areas
    # ---------------------------
    stimulus_area.generate_random_stimulus() 

    # learn what a 0 and a 1 is (...to create a 0 assembly and a 1 assembly)

    # learn how to AND two bits
    # training = {(0,0):0, (0,1):0, (1,0):0, (1,1):1}

    # perform AND of two bits
    # c.project_stimulus(stimulus=np.array([1,0]), area=0)
