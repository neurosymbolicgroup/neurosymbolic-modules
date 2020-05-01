import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

np.random.seed(70)


d, k, p, B = 2, 2, 1e-1, 0.1 #100, 100, 1e-2, 0.1

def set_input(bit, d):
    """set a pattern for an input bit"""
    arr = np.zeros((d,d))
    if bit == 0:
        arr[3*d//8:5*d//8,3*d//8] = 1.
        arr[3*d//8:5*d//8,5*d//8] = 1.
        arr[3*d//8,3*d//8:5*d//8] = 1.
        arr[5*d//8,3*d//8:5*d//8] = 1.
        arr[5*d//8, 5*d//8] = 1.
    if bit == 1:
        arr[d//8:7*d//8,d//2] = 1.
        arr[7*d//8,3*d//8:5*d//8] = 1.
    return arr.ravel()

def train_cap(arr, k, desired_op):
    """
    perform the cap operation in the output assembly:
    first half of neurons in the output assembly correspond to zero
    and the second half correspond to one
    """
    n = arr.shape[0]
    if len(np.where(arr !=0)[0]) > k:
        indices = np.argsort(arr)
        arr[indices[:-k]]=0

    arr[np.where(arr != 0.)[0]] = 1.0
    if desired_op==0:
        arr[n//2:] = 0.
    else:
        arr[:n//2] = 0.
    return arr

def train_operation(W_o1, W_o2, W_oo, num_timesteps=10, k=100):
    """
    main training function
    """
    y_tm1 = np.zeros(W_oo.shape[0])
    d = int(np.sqrt(W_o1.shape[1]))

    for t in range(num_timesteps):
        # draw binary inputs and set output
        b1, b2 = np.random.binomial(1, 0.5), np.random.binomial(1, 0.5)
        desired_op = b1&b2
        ip1, ip2 = set_input(b1,d), set_input(b2,d)

        # one step of firing impulses
        y_t = W_o1.dot(ip1) + W_o2.dot(ip2) + W_oo.dot(y_tm1)
        y_t = train_cap(y_t, k, desired_op)
        y_tm1 = np.copy(y_t)

        # plasticity modifications
        for i in np.where(y_t!=0)[0]:
            for j in np.where(ip1!=0)[0]:
                W_o1[i,j] *= 1.+B

        for i in np.where(y_t!=0)[0]:
            for j in np.where(ip2!=0)[0]:
                W_o2[i,j] *= 1.+B

        for i in np.where(y_t!=0)[0]:
            for j in np.where(y_tm1!=0)[0]:
                W_oo[i,j] *= 1.+B
    return W_o1, W_o2, W_oo

def compute_output(ip1, ip2, W_o1, W_o2, W_oo, num_timesteps=1, k=100):
    """
    compute the output given two binary inputs
    """
    y_tm1 = np.zeros(W_oo.shape[0])

    for t in range(num_timesteps):
        y_t = W_o1.dot(ip1) + W_o2.dot(ip2) + W_oo.dot(y_tm1)
        if len(np.where(y_t !=0)[0]) > k:
            indices = np.argsort(y_t)
            y_t[indices[:-k]]=0

        y_t[np.where(y_t != 0.)[0]] = 1.0
        y_tm1 = np.copy(y_t)

    return y_t

def draw_graph(ip1, ip2, W_o1, W_o2, W_oo):
    """
    ip1 are the input neuron values for input 1
    ip2 are the input neuron values for input 2
    W_o1 are the weights for edges connecting input1 to output
    W_o2 are the weights for edges connecting input2 to output
    W_oo are the recurrent weights in the output
    (Note that there are no recurrent weights in the input)
    """
    # print(np.reshape(ip1, (10,10)))
    # print(np.reshape(ip2, (10,10)))

    # print(W_o1.shape)
    print(W_oo)

    # create adjacency matrix for edges 
    n = 4*d*d
    adj = np.zeros(shape=(n,n))
    adj[0:d*d,      int(n/2):n]  = W_o1.T
    adj[d*d:2*d*d,  int(n/2):n]  = W_o2.T
    adj[2*d*d:4*d*d, int(n/2):n] = W_oo.T
    # print(adj)
    # self.adj_matrix[0:n,0:n] = g1.get_adj_matrix() # make the top left corner the existing connections within graph 1
    # self.adj_matrix[n:2*n,n:2*n] = g2.get_adj_matrix() # make the bottom right corner the existing connections within graph 2
    # self.adj_matrix[0:n,n:2*n] = np.random.binomial(1,p,size=(n,n)).astype("float64") # make the other two corners connections with probability p
    # self.adj_matrix[n:2*n,0:n] = np.random.binomial(1,p,size=(n,n)).astype("float64")

    color_map = []
    for node in range(n):
        if node < d*d: color_map.append('blue')
        elif node < 2*d*d: color_map.append('green')
        else: color_map.append('red')      
        



    graph = nx.convert_matrix.from_numpy_array(adj, create_using=nx.DiGraph)
    nx.draw(graph, node_color=color_map, with_labels=True) # , with_labels=True, but make the labels the node values
    plt.show()




W_o1 = np.random.binomial(1,p,size=(2*d*d,d*d)).astype("float64")
W_o2 = np.random.binomial(1,p,size=(2*d*d,d*d)).astype("float64")
W_oo = np.random.binomial(1,p,size=(2*d*d,2*d*d)).astype("float64")

draw_graph(set_input(1,d), set_input(0,d), W_o1, W_o2, W_oo)


#W_o1, W_o2, W_oo = train_operation(W_o1, W_o2, W_oo, k=k)


"""
Operation:
	The AND operation

Inputs:
	There are 2 input areas of 100 (dxd) neurons each, 

Outputs:
	There is an output area of 200 (2xdxd) neurons. 
		The left 100 neurons correspond to an output of zero 
		and the right 100 neurons correspond to an output of one.
	The output area is restricted to 10 neurons firing total in the left and right sections.
"""
# op = compute_output(set_input(0,d), set_input(0,d), W_o1, W_o2, W_oo, k=k)
# print("when input is (0,0): ", sum(op[:d*d]), sum(op[d*d:]))

# op = compute_output(set_input(1,d), set_input(0,d), W_o1, W_o2, W_oo, k=k)
# print("when input is (1,0): ", sum(op[:d*d]), sum(op[d*d:]))

# op = compute_output(set_input(0,d), set_input(1,d), W_o1, W_o2, W_oo, k=k)
# print("when input is (0,1): ", sum(op[:d*d]), sum(op[d*d:]))

# op = compute_output(set_input(1,d), set_input(1,d), W_o1, W_o2, W_oo, k=k)
# print("when input is (1,1): ", sum(op[:d*d]), sum(op[d*d:]))
