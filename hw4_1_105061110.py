import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Good ref:
# http://colah.github.io/posts/2015-08-Understanding-LSTMs/
# https://zhuanlan.zhihu.com/p/44424550

class RNN():
    def __init__(   self,
                    lr = 0.001,
                    training_iters = 50000,
                    batch_size = 128,
                    n_inputs = 28,
                    n_steps = 28,
                    n_hidden_units = 128,
                    n_classes = 10,
                    cell = 'GRU'):

        tf.set_random_seed(42)
        self.lr = lr
        self.training_iters = training_iters
        self.batch_size = batch_size
        self.n_inputs = n_inputs                # word vector dim or row of pixels
        self.n_steps = n_steps                  # number of words or image height
        self.n_hidden_units = n_hidden_units    # neurons in hidden layer
        self.n_classes = n_classes              # labels
        self.cell = cell
        self.embedding = False
        self.embedding_size = 128
        
        self.build_model()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


    def build_model(self):
        x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs], name='x')
        y = tf.placeholder(tf.float32, [None, self.n_classes], name='y')

        # Define weights before cell and after cell
        weights = {
            'in': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_units, self.n_classes]))
        }
        biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[self.n_classes, ]))
        }

        if self.embedding:
            '''
            Ref: https://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/
            word_embeddings = tf.get_variable([vocabulary_size, embedding_size])
            embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, word_ids)
            word_ids: a sentence represented in an integer vector.
            '''
            word_embeddings = tf.Variable(tf.random_uniform([10000, self.embedding_size], -1, 1), name='word_embeddings')
            embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, x)

            # Modify the original dimension of the weights.
            weights['in'] = tf.Variable(tf.random_normal([embedding_size, self.n_hidden_units]))

        def build_cell(X, weights, biases):
            # transpose the inputs shape to
            # X ==> (batch * steps, inputs)
            if self.embedding:
                X = tf.reshape(X, [-1, self.embedding_size])
            else:
                X = tf.reshape(X, [-1, self.n_inputs])                

            # X_in = (batch * steps, hidden)
            X_in = tf.matmul(X, weights['in']) + biases['in']
            # X_in ==> (batch, steps, hidden)
            X_in = tf.reshape(X_in, [-1, self.n_steps, self.n_hidden_units])


            if self.cell == 'SimpleRNN':
                cell = tf.contrib.rnn.BasicRNNCell(num_units=self.n_hidden_units)   
                # print('*****************')
                # print(cell.state_size) # hidden
                # print('*****************')

                init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
                outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
                # print('*****************')
                # print(outputs)          # Tensor("rnn/transpose_1:0", shape=(batch, step, hidden), dtype=float32)
                # print(final_state)      # Tensor("rnn/while/Exit_3:0", shape=(batch, hidden), dtype=float32) 
                # print('*****************')
                results = tf.matmul(final_state, weights['out']) + biases['out']

            elif self.cell == 'GRU':
                cell = tf.contrib.rnn.GRUCell(num_units=self.n_hidden_units)
                # print('*****************')
                # print(cell.state_size) # hidden
                # print('*****************')

                init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
                outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
                # print('*****************')
                # print(outputs)          # Tensor("rnn/transpose_1:0", shape=(batch, step, hidden), dtype=float32)
                # print(final_state)      # Tensor("rnn/while/Exit_3:0", shape=(batch, hidden), dtype=float32) 
                # print('*****************')
                results = tf.matmul(final_state, weights['out']) + biases['out']

            elif self.cell == 'LSTM':
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.n_hidden_units)
                # print('*****************')
                # print(cell.state_size) # LSTMStateTuple(c=hidden, m=hidden)
                # print('*****************')

                # lstm cell is divided into two parts (c_state, h_state)
                init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
                outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
                # print('*****************')
                # print(outputs)              # Tensor("rnn/transpose_1:0", shape=(batch, step, hidden), dtype=float32)
                # print(final_state.h)        # Tensor("rnn/while/Exit_4:0", shape=(hidden, hidden), dtype=float32) 
                # print(final_state.c)        # Tensor("rnn/while/Exit_3:0", shape=(hidden, hidden), dtype=float32)
                # print(final_state[1].shape) # (hidden, hidden)
                # print('*****************')
                results = tf.matmul(final_state[1], weights['out']) + biases['out']

            return results


        pred = build_cell(x, weights, biases)
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
    # does-it-run test 
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    lstm = RNN( lr = 0.001,
                training_iters = 50000,
                batch_size = 128,
                n_inputs = 28,
                n_steps = 28,
                n_hidden_units = 128,
                n_classes = 10,
                cell = 'GRU')
    lstm.train(mnist)
    del lstm

    # with open('data.pickle','rb') as f:
    #     train_data, train_labels, test_data, test_labels = pickle.load(f)
    # print(train_data.shape)
    # lstm = RNN( lr = 0.001,
    #             training_iters = 50000,
    #             batch_size = 128,
    #             n_inputs = 10000, # dim of word vector
    #             n_steps = 120,    # number of words in a sentence
    #             n_hidden_units = 128,
    #             n_classes = 2,    # pos or neg
    #             cell = 'GRU')
    # lstm.train(mnist)
    # del lstm
    pass