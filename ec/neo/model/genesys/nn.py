import shutil
import numpy as np
import tensorflow as tf

from consts import *

# num_vals:             int (number of possible input values)
# val_embedding_dim:    int (dimension of the vector embedding of values)
# input_length:         int (length of input lists)
# output_length:        int (length of output lists)
# num_dsl_ops:          int (number of dsl operators)
# dsl_embedding_dim:    int (dimension of the vector embedding of dsl operators)
# ngram_length:         int (length of dsl operator ngrams)
# hidden_layer_dim:     int (number of nodes in each hidden layer)
class DeepCoderModelParams:
    def __init__(self, num_vals, val_embedding_dim, input_length, output_length, num_dsl_ops, dsl_embedding_dim, ngram_length, hidden_layer_dim):
        self.num_vals = num_vals
        self.val_embedding_dim = val_embedding_dim
        self.input_length = input_length
        self.output_length = output_length
        self.num_dsl_ops = num_dsl_ops
        self.dsl_embedding_dim = dsl_embedding_dim
        self.ngram_length = ngram_length
        self.hidden_layer_dim = hidden_layer_dim

# num_epochs: int (number of training epochs)
# batch_size: int (number of datapoints per batch)
# step_size:  float (step length in gradient descent)
# save_path:  str (path to save the neural net)
# load_prev:  bool (whether to load an existing save file)
class DeepCoderTrainParams:
    def __init__(self, num_epochs, batch_size, step_size, save_path, load_prev):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.save_path = save_path
        self.load_prev = load_prev

# save_path:  str (path where the neural net is saved)
class DeepCoderTestParams:
    def __init__(self, save_path):
        self.save_path = save_path
        
# input_values:  tensorflow tensor of dimension [?, input_length] and values in {0, 1, ..., num_vals}
# output_values: tensorflow tensor of dimension [?, output_length] and values in {0, 1, ..., num_vals}
# dsl_op_scores: tensorflow tensor of dimension [?, n_gram_length] and values in {0, 1, ..., num_dsl_ops}
# labels:        tensorflow tensor of dimension [?, num_dsl_ops] and values in {0.0, 1.0}
# loss:          tensorflow loss function
# accuracy:      tensorflow accuracy function
class DeepCoderModel:
    # params: DeepCoderModelParams (parameters for the deep coder model)
    def __init__(self, params):

        # Step 1: Variables

        # input value embedding
        value_embeddings = tf.get_variable('value_embeddings', [params.num_vals + 1, params.val_embedding_dim])

        # dsl op embedding
        dsl_op_embeddings = tf.get_variable('dsl_op_embeddings', [params.num_dsl_ops + 1, params.dsl_embedding_dim])

        # hidden layers
        embedding_dim = 2 * params.input_length * params.val_embedding_dim
        embedding_dim += params.output_length * params.val_embedding_dim
        embedding_dim += params.ngram_length * params.dsl_embedding_dim
        
        W0 = tf.Variable(tf.truncated_normal([embedding_dim, params.hidden_layer_dim], stddev=0.1))
        b0 = tf.Variable(tf.constant(0.1, shape=[params.hidden_layer_dim]))
        
        W1 = tf.Variable(tf.truncated_normal([params.hidden_layer_dim, params.hidden_layer_dim], stddev=0.1))
        b1 = tf.Variable(tf.constant(0.1, shape=[params.hidden_layer_dim]))
        
        W2 = tf.Variable(tf.truncated_normal([params.hidden_layer_dim, params.hidden_layer_dim], stddev=0.1))
        b2 = tf.Variable(tf.constant(0.1, shape=[params.hidden_layer_dim]))

        # Step 2: Layers

        # Step 2a: Inputs (input list, output list, DSL operator n-gram)
        self.input_values_0 = []
        self.input_values_1 = []
        self.output_values = []

        for i in range(5):
            self.input_values_0.append(tf.placeholder(tf.int32, [None, params.input_length]))
            self.input_values_1.append(tf.placeholder(tf.int32, [None, params.input_length]))
            self.output_values.append(tf.placeholder(tf.int32, [None, params.output_length]))

        self.dsl_ops = tf.placeholder(tf.int32, [None, params.ngram_length])

        # Step 2b: Embeddings
        embedded_input_values_0_flat = []
        embedded_input_values_1_flat = []
        embedded_output_values_flat = []
        
        for i in range(5):
            # input value embedding
            embedded_input_values_0 = tf.nn.embedding_lookup(value_embeddings, self.input_values_0[i])
            embedded_input_values_0_flat.append(tf.reshape(embedded_input_values_0, [-1, params.input_length * params.val_embedding_dim]))

            embedded_input_values_1 = tf.nn.embedding_lookup(value_embeddings, self.input_values_1[i])
            embedded_input_values_1_flat.append(tf.reshape(embedded_input_values_1, [-1, params.input_length * params.val_embedding_dim]))

            # output value embedding
            embedded_output_values = tf.nn.embedding_lookup(value_embeddings, self.output_values[i])
            embedded_output_values_flat.append(tf.reshape(embedded_output_values, [-1, params.output_length * params.val_embedding_dim]))

        # dsl ngram embedding
        embedded_dsl_ops = tf.nn.embedding_lookup(dsl_op_embeddings, self.dsl_ops)
        embedded_dsl_ops_flat = tf.reshape(embedded_dsl_ops, [-1, params.ngram_length * params.dsl_embedding_dim])

        # Step 2c: Concatenation
        merged = []
        for i in range(5):
            merged.append(tf.concat([embedded_input_values_0_flat[i], embedded_input_values_1_flat[i], embedded_output_values_flat[i], embedded_dsl_ops_flat], 1))

        # Step 2d: Hidden
        hidden2 = []
        for i in range(5):
            hidden0 = tf.nn.relu(tf.matmul(merged[i], W0) + b0)
            hidden1 = tf.nn.relu(tf.matmul(hidden0, W1) + b1)
            hidden2.append(tf.matmul(hidden1, W2) + b2)

        # Step 2e: Average pooling
        pool = tf.reduce_mean(hidden2, 0)

        # Step 2f: Logits
        dsl_op_logits = tf.layers.dense(inputs=pool, units=params.num_dsl_ops, activation=None)

        # Step 6: Output (probability of each DSL operator)
        self.dsl_op_scores = tf.nn.softmax(dsl_op_logits)

        # Step 7: Loss layer
        self.labels = tf.placeholder(tf.float32, [None, params.num_dsl_ops])
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=dsl_op_logits))

        # Step 8: Accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(self.dsl_op_scores, 1), tf.int32), tf.cast(tf.argmax(self.labels, 1), tf.int32)), tf.float32))

    # input_values_0_train: np.array([num_train, input_length])
    # input_values_1_train: np.array([num_train, input_length])
    # output_values_train:  np.array([num_train, output_length])
    # dsl_ops_train:        np.array([num_train, ngram_length])
    # labels_train:         np.array([num_train, num_dsl_ops])
    # input_values_0_test:  np.array([num_test, input_length])
    # input_values_1_test:  np.array([num_test, input_length])
    # output_values_test:   np.array([num_test, output_length])
    # dsl_ops_test:         np.array([num_test, ngram_length])
    # labels_test:          np.array([num_test, num_dsl_ops])
    # params:               DeepCoderTrainParams
    def train(self, input_values_0_train, input_values_1_train, output_values_train, dsl_ops_train, labels_train, input_values_0_test, input_values_1_test, output_values_test, dsl_ops_test, labels_test, params):
        # Step 1: Save path
        save_path = self.save_path(params)
        
        # Step 2: Compute number of batches
        num_batches = len(labels_train)/params.batch_size

        # Step 3: Training step
        train_step = tf.train.AdamOptimizer(params.step_size).minimize(self.loss)

        # Step 4: Training
        with tf.Session() as sess:
            # Step 4a: Global variables initialization
            sess.run(tf.global_variables_initializer())

            # Step 4b: Load existing model
            if params.load_prev and tf.train.checkpoint_exists(save_path):
                tf.train.Saver().restore(sess, save_path)
                print 'Loaded deep coder model in: %s' % save_path

            min_loss = None

            for i in range(params.num_epochs):
                print 'epoch: %d' % i
                for j in range(num_batches):

                    if j%10000 == 0:
                        print 'Batch:', j
                    
                    # Step 4c: Compute batch bounds
                    lo = j*params.batch_size
                    hi = (j+1)*params.batch_size
                    
                    # Step 4d: Compute batch
                    feed_dict = {}
                    feed_dict[self.dsl_ops] = dsl_ops_train[lo:hi]
                    feed_dict[self.labels] = labels_train[lo:hi]
                    for i in range(5):
                        feed_dict[self.input_values_0[i]] = input_values_0_train[i][lo:hi]
                        feed_dict[self.input_values_1[i]] = input_values_1_train[i][lo:hi]
                        feed_dict[self.output_values[i]] = output_values_train[i][lo:hi]
                        
                    # Step 4e: Run training step
                    sess.run(train_step, feed_dict=feed_dict)

                    if j%1000 == 0:

                        # Step 4f: Save model
                        if min_loss is None:
                            tf.train.Saver().save(sess, save_path)
                            
                        # Step 4g: Test set accuracy
                        (loss, accuracy) = self.test(input_values_0_test, input_values_1_test, output_values_test, dsl_ops_test, labels_test, params)
                        
                        # Step 4h: Save model
                        if loss <= min_loss:
                            tf.train.Saver().save(sess, save_path)
                            print 'Saved deep coder neural net in: %s' % save_path
                            min_loss = loss
                            
    # input_values_1: np.array([num_test, input_length])
    # input_values_0: np.array([num_test, input_length])
    # output_values:  np.array([num_test, output_length])
    # dsl_ops:        np.array([num_test, ngram_length])
    # labels_test:    np.array([num_test, num_dsl_ops])
    # params:         DeepCoderTrainParams | DeepCoderTestParams
    def test(self, input_values_0, input_values_1, output_values, dsl_ops, labels, params):
        with tf.Session() as sess:
            # Step 1: Directory path
            save_path = self.save_path(params)

            # Step 2: Load neural net
            tf.train.Saver().restore(sess, save_path)
            print 'Loaded deep coder model in: %s' % save_path

            # Test neural net
            feed_dict = {}
            feed_dict[self.dsl_ops] = dsl_ops
            feed_dict[self.labels] = labels
            for i in range(5):
                feed_dict[self.input_values_0[i]] = input_values_0[i]
                feed_dict[self.input_values_1[i]] = input_values_1[i]
                feed_dict[self.output_values[i]] = output_values[i]
            loss = sess.run(self.loss, feed_dict=feed_dict)
            print 'Loss: %g' % loss
            accuracy = sess.run(self.accuracy, feed_dict=feed_dict)
            print 'Accuracy: %g' % accuracy

        return (loss, accuracy)

    # input_values_0: np.array([num_run, input_length])
    # input_values_1: np.array([num_run, input_length])
    # output_values:  np.array([num_run, output_length])
    # dsl_ops:        np.array([num_test, ngram_length])
    # params:         DeepCoderTestParams
    def run(self, input_values_0, input_values_1, output_values, dsl_ops, params):

        with tf.Session() as sess:
            # Step 1: Directory path
            save_path = self.save_path(params)

            # Step 2: Load neural net
            tf.train.Saver().restore(sess, save_path)
            print 'Loaded deep coder model in: %s' % save_path

            # Step 3: Build inputs
            feed_dict = {}
            feed_dict[self.dsl_ops] = dsl_ops
            for i in range(5):
                feed_dict[self.input_values_0[i]] = input_values_0[i]
                feed_dict[self.input_values_1[i]] = input_values_1[i]
                feed_dict[self.output_values[i]] = output_values[i]

            # Step 4: Run prediction
            scores = sess.run(self.dsl_op_scores, feed_dict=feed_dict)

        return scores

    # params: DeepCoderTrainParams | DeepCoderTestParams
    def save_path(self, params):
        return DATA_PATH + '/' + params.save_path
