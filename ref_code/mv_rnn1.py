import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Good ref:
# http://colah.github.io/posts/2015-08-Understanding-LSTMs/


# import hw4_1_preprocess
# train_data = np.loadtxt('train_data.txt', dtype=int)
# train_labels = np.loadtxt('train_labels.txt', dtype=int)
# test_data = np.loadtxt('test_data.txt', dtype=int)
# test_labels = np.loadtxt('test_labels.txt', dtype=int)import tensorflow as tf

class LSTM():
    def __init__(self):
        # set random seed for comparing the two result calculations
        tf.set_random_seed(42)

        # hyperparameters
        self.lr = 0.001
        self.training_iters = 10000
        self.batch_size = 128

        self.n_inputs = 28   # MNIST data input (img shape: 28*28)
        self.n_steps = 28    # time steps
        self.n_hidden_units = 128   # neurons in hidden layer
        self.n_classes = 10      # MNIST classes (0-9 digits)

        
        self.build_model()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


    def build_model(self):
        # tf Graph input
        x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs], name='x')
        y = tf.placeholder(tf.float32, [None, self.n_classes], name='y')

        # Define weights
        weights = {
            # (28, 128)
            'in': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units])),
            # (128, 10)
            'out': tf.Variable(tf.random_normal([self.n_hidden_units, self.n_classes]))
        }
        biases = {
            # (128, )
            'in': tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ])),
            # (10, )
            'out': tf.Variable(tf.constant(0.1, shape=[self.n_classes, ]))
        }


        def RNN(X, weights, biases):
            # hidden layer for input to cell
            ########################################

            # transpose the inputs shape from
            # X ==> (128 batch * 28 steps, 28 inputs)
            X = tf.reshape(X, [-1, self.n_inputs])

            # into hidden
            # X_in = (128 batch * 28 steps, 128 hidden)
            X_in = tf.matmul(X, weights['in']) + biases['in']
            # X_in ==> (128 batch, 28 steps, 128 hidden)
            X_in = tf.reshape(X_in, [-1, self.n_steps, self.n_hidden_units])

            # cell
            ##########################################

            # basic LSTM Cell.
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden_units, forget_bias=1.0, state_is_tuple=True)
            else:
                cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units)
            # lstm cell is divided into two parts (c_state, h_state)
            init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
            results = tf.matmul(final_state[1], weights['out']) + biases['out']

            return results


        pred = RNN(x, weights, biases)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(cost)

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        self.accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    def train(self, mnist):
        self.sess.run(tf.global_variables_initializer())
        step = 0
        while step * self.batch_size < self.training_iters:
            batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)
            batch_xs = batch_xs.reshape([self.batch_size, self.n_steps, self.n_inputs])
            self.sess.run(self.train_op, feed_dict={
                'x:0': batch_xs,
                'y:0': batch_ys
            })
            if step % 20 == 0:
                print(self.sess.run(self.accuracy_op, feed_dict={
                'x:0': batch_xs,
                'y:0': batch_ys
                }))
            step += 1

if __name__ == "__main__":
    # this is data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    lstm = LSTM()
    lstm.train(mnist)
    del lstm