"""
Classify 0 and 1 images in MNIST
Using Hebbian learning, instead of the more common gradient descent
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# load MNIST set from https://github.com/aiddun/binary-mnist
x_train, y_train = np.load("data/binary_digits_binary_pixels/x_train.npy"), np.load("data/binary_digits_all_pixels/y_train.npy")
# print(y_train[91])
# plt.matshow(x_train[91].reshape(28,28), cmap="gray")
# plt.show()

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

def train_operation(W_o1, W_oo, num_timesteps=10, k=100):
    """
    main training function
    """
    y_tm1 = np.zeros(W_oo.shape[0])
    d = int(np.sqrt(W_o1.shape[1]))

    for t in range(num_timesteps):
        # draw binary input and set output
        b1 = np.random.binomial(1, 0.5)
        ip1 = set_input(b1,d)

        # i steps of firing impulses
        for i in range(0,3):
            y_t = W_o1.dot(ip1) + W_oo.dot(y_tm1)
            y_t = train_cap(y_t, k, b1)
            y_tm1 = np.copy(y_t)

        title = "Training: {}".format(b1)
        if DRAW_GRAPHS: draw_graph(ip1, W_o1, W_oo, y_t, title)

        # plasticity modifications
        for i in np.where(y_t!=0)[0]:
            for j in np.where(ip1!=0)[0]:
                W_o1[i,j] *= 1.+B

        for i in np.where(y_t!=0)[0]:
            for j in np.where(y_tm1!=0)[0]:
                W_oo[i,j] *= 1.+B


    return W_o1, W_oo

def compute_output(b1, W_o1, W_oo, num_timesteps=1, k=100):
    """
    compute the output given two binary inputs
    """

    ip1 = set_input(b1,d)

    y_tm1 = np.zeros(W_oo.shape[0])

    for t in range(num_timesteps):
        y_t = W_o1.dot(ip1) + W_oo.dot(y_tm1)
        if len(np.where(y_t !=0)[0]) > k:
            indices = np.argsort(y_t)
            y_t[indices[:-k]]=0

        y_t[np.where(y_t != 0.)[0]] = 1.0
        y_tm1 = np.copy(y_t)

    title = "Testing: {}".format(b1)
    if DRAW_GRAPHS:draw_graph(ip1, W_o1, W_oo, y_t, title)

    return y_t

def draw_graph(ip1, W_o1, W_oo, y, title):
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
    n = 3*d*d
    adj = np.zeros(shape=(n,n))
    adj[0:d*d,      d*d:3*d*d]  = W_o1.T
    adj[d*d:3*d*d,  d*d:3*d*d]  = W_oo.T

    # turn into nx graph
    graph = nx.convert_matrix.from_numpy_array(adj, create_using=nx.DiGraph)

    # create the labels of node values
    labels_list = np.concatenate((ip1, y))
    labels = {}
    nodes = graph.nodes()
    i=0
    for node in graph.nodes(): 
        labels[node] = labels_list[i]; i+=1   

    # color the inputs and outputs different colors
    color_map = []
    for node in range(n):
        if node < d*d: color_map.append('blue')
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

    # issue -- not drawing self loops
    plt.title(title)
    plt.show()



"""
Operation:
    Classification

Inputs:
    There is 1 input area of 784 (dxd) neurons, 

Outputs:
    There is an output area of 1568 (2xdxd) neurons. 
        The left 784 neurons correspond to an output of zero 
        and the right 784 neurons correspond to an output of one.
    The output area is restricted to 27 (k) neurons firing total in the left and right sections.
"""

def run(i):
    
    # weights from input to output
    W_o1 = np.random.binomial(1,p,size=(2*d*d,d*d)).astype("float64")
    # recurrent weights between output and output
    W_oo = np.random.binomial(1,p,size=(2*d*d,2*d*d)).astype("float64")

    # print("-----PRE-TESTING-----")

    # op_a = compute_output(0, W_o1, W_oo, k=k)
    # op_a_0, op_a_1 = sum(op_a[:d*d]), sum(op_a[d*d:])

    # op_b = compute_output(1, W_o1, W_oo, k=k)
    # op_b_0, op_b_1 = sum(op_b[:d*d]), sum(op_b[d*d:])

    print("-----TRAINING-----")

    W_o1, W_oo = train_operation(W_o1, W_oo, k=k)

    # print("-----TESTING-----")

    # op_a = compute_output(0, W_o1, W_oo, k=k)
    # op_a_0, op_a_1 = sum(op_a[:d*d]), sum(op_a[d*d:])

    # op_b = compute_output(1, W_o1, W_oo, k=k)
    # op_b_0, op_b_1 = sum(op_b[:d*d]), sum(op_b[d*d:])

# for i in range(50):
#     np.random.seed(i)
#     run(i)
DRAW_GRAPHS=True
np.random.seed(24)
run(24)
