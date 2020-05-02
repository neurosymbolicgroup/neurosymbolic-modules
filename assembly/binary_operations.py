import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)

d, k, p, B = 3, 3, 5e-2, 0.1 #100, 100, 1e-2, 0.1

def set_input(bit, d):
    """set a pattern for an input bit"""
    arr = np.zeros((d,d))
    if bit == 0:
        # arr[3*d//8:5*d//8,3*d//8] = 1.
        # arr[3*d//8:5*d//8,5*d//8] = 1.
        # arr[3*d//8,3*d//8:5*d//8] = 1.
        # arr[5*d//8,3*d//8:5*d//8] = 1.
        # arr[5*d//8, 5*d//8] = 1.
        arr[d//2:d,0:d] = 1.
    if bit == 1:
        # arr[d//8:7*d//8,d//2] = 1.
        # arr[7*d//8,3*d//8:5*d//8] = 1.
        arr[0:d//2,0:d] = 1.
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
        # with slightly higher probability of selecting a 1 over a 0
        #   b/c otherwise, for the AND operation, the output is going to be 0 more often than not
        b1, b2 = np.random.binomial(1, 0.7), np.random.binomial(1, 0.7)
        desired_op = b1&b2
        ip1, ip2 = set_input(b1,d), set_input(b2,d)

        # i steps of firing impulses
        for i in range(0,3):
            y_t = W_o1.dot(ip1) + W_o2.dot(ip2) + W_oo.dot(y_tm1)
            y_t = train_cap(y_t, k, desired_op)
            y_tm1 = np.copy(y_t)

        # for lots of projects
        # for i in range(0,1): 
        #     y_t = W_o1.dot(ip1) + W_oo.dot(y_tm1)
        #     y_t = train_cap(y_t, k, desired_op)
        #     y_tm1 = np.copy(y_t)

        print(b1, b2, desired_op)
        draw_graph(ip1, ip2, W_o1, W_o2, W_oo, y_t)

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

    draw_graph(ip1, ip2, W_o1, W_o2, W_oo, y_t)

    return y_t

def draw_graph(ip1, ip2, W_o1, W_o2, W_oo, y):
    """
    ip1 are the input neuron values for input 1
    ip2 are the input neuron values for input 2
    y is the neuron values for the output (before the input has been processed)

    (There are no recurrent weights in the input)
    W_o1 are the weights for edges connecting input1 to output
    W_o2 are the weights for edges connecting input2 to output
    W_oo are the recurrent weights in the output
    """
    # print(np.reshape(ip1, (10,10)))
    # print(np.reshape(ip2, (10,10)))

    # create adjacency matrix for edges 
    n = 4*d*d
    adj = np.zeros(shape=(n,n))
    adj[0:d*d,      int(n/2):n]  = W_o1.T
    adj[d*d:2*d*d,  int(n/2):n]  = W_o2.T
    adj[2*d*d:4*d*d, int(n/2):n] = W_oo.T

    # turn into nx graph
    graph = nx.convert_matrix.from_numpy_array(adj, create_using=nx.DiGraph)

    # create the labels of node values
    labels_list = np.concatenate((ip1, ip2, y))
    labels = {}
    nodes = graph.nodes()
    i=0
    for node in graph.nodes(): 
        labels[node] = labels_list[i]; i+=1   

    # color the inputs and outputs different colors
    color_map = []
    for node in range(n):
        if node < d*d: color_map.append('blue')
        elif node < 2*d*d: color_map.append('green')
        else: 
            if labels[node]==1:
                color_map.append('red') # output neuron that is activated
            else:
                color_map.append('black') # output neuron that is dead


    # get edge labels
    edge_labels = nx.get_edge_attributes(graph,'weight')
    for edge, label in edge_labels.items():
        edge_labels[edge] = round(label, 2)

    # draw graph
    pos = nx.circular_layout(graph)

    nx.draw_networkx_nodes(graph, pos, node_color=color_map, alpha=1) 
    nx.draw_networkx_labels(graph, pos, labels, font_color="white", font_size=8)

    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)#, connectionstyle='arc3,rad=0.2')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)

    plt.show()



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

def main():

    print("-----TRAINING-----")

    W_o1 = np.random.binomial(1,p,size=(2*d*d,d*d)).astype("float64")
    W_o2 = np.random.binomial(1,p,size=(2*d*d,d*d)).astype("float64")
    W_oo = np.random.binomial(1,p,size=(2*d*d,2*d*d)).astype("float64")

    W_o1, W_o2, W_oo = train_operation(W_o1, W_o2, W_oo, k=k)

    print("-----TESTING-----")

    # print("when input is (0,0): ")#, sum(op[:d*d]), sum(op[d*d:]))
    op_a = compute_output(set_input(0,d), set_input(0,d), W_o1, W_o2, W_oo, k=k)
    # op_a_0, op_a_1 = sum(op_a[:d*d]), sum(op_a[d*d:])

    # print("when input is (1,0): ")#, sum(op[:d*d]), sum(op[d*d:]))
    op_b = compute_output(set_input(1,d), set_input(0,d), W_o1, W_o2, W_oo, k=k)
    # op_b_0, op_b_1 = sum(op_b[:d*d]), sum(op_b[d*d:])

    # print("when input is (0,1): ")#, sum(op[:d*d]), sum(op[d*d:]))
    op_c = compute_output(set_input(0,d), set_input(1,d), W_o1, W_o2, W_oo, k=k)
    # op_c_0, op_c_1 = sum(op_c[:d*d]), sum(op_c[d*d:])

    # print("when input is (1,1): ")#, sum(op[:d*d]), sum(op[d*d:]))
    op_d = compute_output(set_input(1,d), set_input(1,d), W_o1, W_o2, W_oo, k=k)
    # op_d_0, op_d_1 = sum(op_d[:d*d]), sum(op_d[d*d:])

    # if (op_d_1 > op_d_0) and (op_c_0 > op_c_1):
    #     print("success")

# fun/sad fact
# it actually comes up with the same result before training and after training...
# so it's not clear that training is helping this guy along...
main()
