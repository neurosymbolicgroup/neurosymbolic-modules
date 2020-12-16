import io
import random
import numpy as np

from consts import *
from gen_io import *

# filename: str (path of the dataset to read)
# num_dsl_ops: int (number of DSL operators)
# return: (np.array([num_points, input_length], int),
#          np.array([num_points, input_length], int),
#          np.array([num_points, output_length], int),
#          np.array([num_points, n_gram_length], int),
#          np.array([num_points, num_dsl_ops], int))
#         (a (input values, output values, dsl ops, label) tuple)
def read_train_dataset(filename, num_dsl_ops):
    
    f = open(DATA_PATH + '/' + filename)

    input_values_0 = []
    input_values_1 = []
    output_values = []

    for i in range(5):
        input_values_0.append([])
        input_values_1.append([])
        output_values.append([])
    
    dsl_ops = []
    labels = []

    counter = 0

    for line in f:

        if counter%100000 == 0:
            print 'Read ' + str(counter)
        if counter >= 1000000:
            break
        counter += 1

        # Obtain (DSL ops, input values, output values) tuples
        toks = line[2:-3].split('], [')

        for i in range(5):
            input_values_0[i].append(_process_list(toks[i]))
            input_values_1[i].append(_process_list(toks[i+5]))
            output_values[i].append(_process_list(toks[i+10]))

        dsl_ops.append(_process_list(toks[15]))
        labels.append(_process_list(toks[16]))

    print 'Total read: ' + str(len(labels))

    return (np.array(input_values_0), np.array(input_values_1), np.array(output_values), np.array(dsl_ops), np.array(labels))

# filename: str (path of the dataset to read)
# return: (np.array([num_points, input_length], int),
#          np.array([num_points, output_length], int))
#         (a (input values, output values) tuple)
def read_test_dataset(filename):
    f = open(TMP_PATH + '/' + filename)

    input_values_0 = []
    input_values_1 = []
    output_values = []
    for i in range(5):
        input_values_0.append([])
        input_values_1.append([])
        output_values.append([])
    dsl_ops = []
    
    for line in f:
        # Step 1: Obtain (DSL ops, input values, output values) tuples
        toks = line[2:-3].split('], [')

        # Step 2: Process values
        for i in range(5):
            input_values_0[i].append(_process_list(toks[i]))
            input_values_1[i].append(_process_list(toks[i+5]))
            output_values[i].append(_process_list(toks[i+10]))
        
        dsl_ops.append(_process_list(toks[15]))

    return (np.array(input_values_0), np.array(input_values_1), np.array(output_values), np.array(dsl_ops))
    
# s: str
# return: [int]
def _process_list(s):
    return [int(v) for v in s.split(', ')]

# dataset: (np.array([num_points, input_length], int),
#           np.array([num_points, input_length], int),
#           np.array([num_points, output_length], int),
#           np.array([num_points, n_gram_length], int))
#          (a (input values, output values, label) tuple)
# train_frac: float (proportion of points to use for training)
# return: (train_dataset, test_dataset) where train_dataset, test_dataset each have the same type as dataset
def split_train_test(dataset, train_frac):
    input_values_0 = [_split_train_test_single(input_value_0, train_frac) for input_value_0 in dataset[0]]
    input_values_1 = [_split_train_test_single(input_value_1, train_frac) for input_value_1 in dataset[1]]
    output_values = [_split_train_test_single(output_value, train_frac) for output_value in dataset[2]]
    dsl_ops = _split_train_test_single(dataset[3], train_frac)
    labels = _split_train_test_single(dataset[4], train_frac)
    return tuple(tuple([[input_values_0[i][t] for i in range(5)],
                        [input_values_1[i][t] for i in range(5)],
                        [output_values[i][t] for i in range(5)],
                        dsl_ops[t],
                        labels[t]]) for t in range(2))

# dataset_single: np.array([num_points, num_vals], int)
# train_frac: float (proportion of points to use for training)
def _split_train_test_single(dataset_single, train_frac):
    n_train = int(train_frac*len(dataset_single))
    return (dataset_single[:n_train], dataset_single[n_train:])
