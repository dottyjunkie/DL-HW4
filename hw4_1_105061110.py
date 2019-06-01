import pickle
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
        self.embedding_size = 128
        
        self.build_model()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


    def build_model(self):
        x = tf.placeholder(tf.int32, [None, self.n_steps], name='x')
        y = tf.placeholder(tf.int32, [None], name='y')

        '''
        Ref: https://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/
        word_embeddings = tf.get_variable([vocabulary_size, embedding_size])
        embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, word_ids)
        word_ids: a sentence represented in an integer vector.
        '''
        word_embeddings = tf.Variable(tf.random_uniform([10000, self.embedding_size], -1, 1))
        embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, x)

        if self.cell == 'SimpleRNN':
            cell = tf.contrib.rnn.BasicRNNCell(num_units=self.n_hidden_units)   
            # print('*****************')
            # print(cell.state_size) # hidden
            # print('*****************')

            init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(cell, embedded_word_ids, initial_state=init_state, time_major=False)
            # print('*****************')
            # print(outputs)          # Tensor("rnn/transpose_1:0", shape=(batch, step, hidden), dtype=float32)
            # print(final_state)      # Tensor("rnn/while/Exit_3:0", shape=(batch, hidden), dtype=float32) 
            # print('*****************')
            results = tf.matmul(final_state, weights['out']) + biases['out']

        elif self.cell == 'GRU':
            lstm = tf.contrib.rnn.GRUCell(num_units=self.n_hidden_units)
            drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=0.5)
            cell = tf.contrib.rnn.MultiRNNCell([drop] * 2)

            init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(cell, embedded_word_ids, initial_state=init_state, time_major=False)

            weights = tf.truncated_normal_initializer(stddev=0.1)
            biases = tf.zeros_initializer()
            dense = tf.contrib.layers.fully_connected(outputs[:, -1],
                                num_outputs = 256,
                                activation_fn = tf.sigmoid,
                                weights_initializer = weights,
                                biases_initializer = biases)
            results = tf.contrib.layers.fully_connected(dense, 
                              num_outputs = 1, 
                              activation_fn=tf.sigmoid,
                              weights_initializer = weights,
                              biases_initializer = biases)

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

        pred = results
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(cost)

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        self.accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    def train(self, X, y):
        self.sess.run(tf.global_variables_initializer())
        step = 0
        while step * self.batch_size < self.training_iters:
            batch_gen = batch_generator(X, y)
            for i, (batch_x, batch_y) in enumerate(batch_gen):
                feed = {
                    'x:0' : batch_x,
                    'y:0' : batch_y
                }
                self.sess.run(self.train_op, feed_dict=feed)
            if step % 20 == 0:
                print(self.sess.run(self.accuracy_op, feed_dict=feed))
            step += 1


def one_hot(dict_size, target):
    embedding = np.identity(dict_size, dtype=np.int32)
    dim = len(target.shape)

    if dim == 1:
        result = np.zeros((len(target), dict_size))
        for w in range(len(target)):
            result[w,:] = embedding[target[w],:]
        return result

    elif dim == 2:
        batch_size = target.shape[0]
        sentence_len = target.shape[1]
        result = np.zeros((batch_size, sentence_len, dict_size))
        for s in range(batch_size):
            for w in range(sentence_len):
                result[s,w,:] = embedding[target[s,w],:]
        return result


def batch_generator(X, y, batch_size=128, shuffle=True, random_seed=42):
    idx = np.arange(y.shape[0])

    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
    # y = one_hot(2, y)
    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i+batch_size, :], y[i:i+batch_size])

if __name__ == "__main__":
    with open('data.pickle','rb') as f:
        train_data, train_labels, test_data, test_labels = pickle.load(f)
    print(train_data.shape)
    rnn = RNN( lr = 0.001,
                training_iters = 50000,
                batch_size = 128,
                n_inputs = 1,   # dim of word vector
                n_steps = 120,    # number of words in a sentence
                n_hidden_units = 128,
                n_classes = 1,    # pos or neg
                cell = 'GRU')
    rnn.train(train_data, train_labels)
    del lstm
