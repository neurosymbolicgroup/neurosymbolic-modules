"""
Classify 0 and 1 images in MNIST
Using Hebbian learning, instead of the more common gradient descent
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from math import sqrt

import warnings
warnings.filterwarnings("ignore")

# ---------------------
# load 2-way, LeNet representations
# ---------------------
train_activations = np.load('data/new_representations/binary_digits_binary_pixels/train_activations.npy', allow_pickle=True).item()
x_train = train_activations['fc2'] # This should be a sth x 84 numpy array
y_train = np.load("data/new_representations/binary_digits_all_pixels/y_train.npy")

test_activations = np.load('data/new_representations/binary_digits_binary_pixels/test_activations.npy', allow_pickle=True).item()
x_test = test_activations['fc2'] # This should be a sth x 84 numpy array
y_test = np.load("data/new_representations/binary_digits_all_pixels/y_test.npy")

# ---------------------
# load 10-way, raw representations from https://github.com/aiddun/binary-mnist
# ---------------------
# x_train, y_train = np.load("data/new_representations/all_digits_binary_pixels/x_train.npy"), np.load("data/new_representations/all_digits_all_pixels/y_train.npy")
# x_test, y_test = np.load("data/new_representations/all_digits_binary_pixels/x_test.npy"), np.load("data/new_representations/all_digits_all_pixels/y_test.npy")

# ---------------------
# load 2-way, raw representations from https://github.com/aiddun/binary-mnist
# ---------------------
# x_train, y_train = np.load("data/old_representations/binary_digits_binary_pixels/x_train.npy"), np.load("data/old_representations/binary_digits_all_pixels/y_train.npy")
# x_test, y_test = np.load("data/old_representations/binary_digits_binary_pixels/x_test.npy"), np.load("data/old_representations/binary_digits_all_pixels/y_test.npy")

# ---------------------
# load dummy data, for graphing
# ---------------------
# x_train, y_train = np.array([[1,1,0,0], [0,0,1,1], [1,1,0,0], [0,0,1,1]]), np.array([0,1,0,1])
# x_test, y_test = np.array([[1,1,0,0], [0,0,1,1]]), np.array([0,1])

# print(x_train.shape)
# print(y_train[20])
# plt.matshow(x_train[20].reshape(10,10), cmap="gray")
# plt.show()

d  = 84 # d=input neuron size
NUM_OUTPUT_AREAS = 2 # number of output values
AREA_SIZE = None #n//NUM_OUTPUT_AREAS # output neurons / number of output values
B = 0.1 # B = speed of learning parameter

DRAW_GRAPHS=False
np.random.seed(3)

INITIAL_PROJECTION=True

def cap(arr):
    """
    perform the cap operation in any assembly
    """
    if len(np.where(arr !=0)[0]) > k:
        indices = np.argsort(arr)
        arr[indices[:-k]]=0
    arr[np.where(arr != 0.)[0]] = 1.0
    return arr

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
    for i in range(NUM_OUTPUT_AREAS):
        if desired_op==i: # zero out everything except the desired output area
            arr[0:AREA_SIZE*i] = 0.
            arr[AREA_SIZE*(i+1):n] = 0.
    return arr

def train_operation(W_rp, W_o1, W_oo, num_train_examples):
    """
    main training function
    """
    y_tm1 = np.zeros(W_oo.shape[0])


    for t in range(num_train_examples):
        # draw binary input and set output
        b1 = y_train[t] # the input bit
        ip1 = x_train[t] # the set of dxd neurons

        # do initial projection and cap
        if INITIAL_PROJECTION: ip1 = ip1.dot(W_rp.T); ip1 = cap(ip1)

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

        # every 10th plasticity timestep, normalize the weights
        if t % 10 == 0:
            EPS = 1e-20
            W_o1 = np.diag(1./(EPS+W_o1.dot(np.ones(W_o1.shape[1])))).dot(W_o1)
            W_oo = np.diag(1./(EPS+W_oo.dot(np.ones(W_oo.shape[1])))).dot(W_oo)        #



    return W_o1, W_oo

def compute_output(b1, ip1, W_rp, W_o1, W_oo, num_timesteps=1):
    """
    compute the output given two binary inputs
    """

    y_tm1 = np.zeros(W_oo.shape[0])

    # do initial projection and cap
    if INITIAL_PROJECTION: ip1 = ip1.dot(W_rp.T); ip1 = cap(ip1)

    for t in range(num_timesteps):
        y_t = W_o1.dot(ip1) + W_oo.dot(y_tm1)
        if len(np.where(y_t !=0)[0]) > k:
            indices = np.argsort(y_t)
            y_t[indices[:-k]]=0

        y_t[np.where(y_t != 0.)[0]] = 1.0
        y_tm1 = np.copy(y_t)

    
    if DRAW_GRAPHS: title = "Testing: {}".format(b1); draw_graph(ip1, W_o1, W_oo, y_t, title)

    n = y_t.shape[0]
    votes = np.zeros(NUM_OUTPUT_AREAS)
    for i in range(NUM_OUTPUT_AREAS):
        votes[i] = sum(y_t[AREA_SIZE*i : AREA_SIZE*(i+1)])

    return np.argmax(votes) #e.g. if the votes in the 3rd index are the most, return 3 as output

def test_operation(W_rp, W_o1, W_oo, num_test_examples):

    total_correct = 0
    total = num_test_examples
    for i in range(num_test_examples):
        b1, ip1 = y_test[i], x_test[i]
        out = compute_output(b1, ip1, W_rp, W_o1, W_oo)
        # print("when input is", b1, "output is", out)
        if b1 == out: total_correct+=1

    # print(total_correct/total*100,"% Correct")
    return total_correct/total*100


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
    N = 3*d # size of input area (d) + size of output area (2d) OLD!  now size of output area is n.
    adj = np.zeros(shape=(N,N))
    adj[0:d,      d:3*d]  = W_o1.T
    adj[d:3*d,  d:3*d]  = W_oo.T

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
        if node < d: color_map.append('blue')
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
print("n,accuracy")
for n in [1000, 5000, 10000]: # 10, 100, 500, 
    accuracies = []

    # n = 1000 # n=total assembly size
    k = int(sqrt(n)) # k = number of capped neurons
    p= 1/k #1e-1 # p = probability of edge connection
    AREA_SIZE = n//NUM_OUTPUT_AREAS # output neurons / number of output values

    for trial in range(20):
        if INITIAL_PROJECTION: 
            # weights from input to input projection
            W_rp = np.random.binomial(1,p,size=(n,d)).astype("float64")
            # weights from input projection to output
            W_o1 = np.random.binomial(1,p,size=(2*n,n)).astype("float64")
        else:
            # weights from input to input projection
            W_rp = []#np.random.binomial(1,p,size=(n,d)).astype("float64")
            # weights from input projection to output
            W_o1 = np.random.binomial(1,p,size=(2*n,d)).astype("float64")
        # recurrent weights between output and output
        W_oo = np.random.binomial(1,p,size=(2*n,2*n)).astype("float64")

        #print("-----PRE-TESTING-----")
        test_operation(W_rp, W_o1, W_oo, num_test_examples=y_test.shape[0])

        #print("-----TRAINING-----")
        W_o1, W_oo = train_operation(W_rp, W_o1, W_oo, num_train_examples=y_train.shape[0])

        #print("-----TESTING-----")
        accuracy = test_operation(W_rp, W_o1, W_oo, num_test_examples=y_test.shape[0])
        accuracies.append(accuracy)
    print(n,   ",",    ",".join(map(str, accuracies)))









