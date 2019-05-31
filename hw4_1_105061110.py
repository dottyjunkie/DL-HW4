import pickle
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Good ref:
# http://colah.github.io/posts/2015-08-Understanding-LSTMs/
# https://zhuanlan.zhihu.com/p/44424550

class RNN():
    def __init__(   self,
                    n_words,
                    seq_len,
                    n_hidden_units = 128,
                    num_layers = 1,
                    batch_size = 128,
                    lr = 0.001,
                    embedding_size = 256,
                    epochs = 20,
                    cell = 'GRU'):

        self.n_words = n_words
        self.seq_len = seq_len
        self.n_hidden_units = n_hidden_units
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lr = lr
        self.embedding_size = embedding_size
        self.epochs = epochs
        self.cell = cell
        
        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(42)
            self.build_model()
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


    def build_model(self):
        tf_x = tf.placeholder(tf.int32, shape=(self.batch_size, self.seq_len), name='tf_x')
        tf_y = tf.placeholder(tf.int32, shape=(self.batch_size), name='tf_y')
        tf_keep_prob = tf.placeholder(tf.float32, name='tf_keep_prob')

        '''
        Ref: https://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/
        word_embeddings = tf.get_variable([vocabulary_size, embedding_size])
        embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, word_ids)
        word_ids: a sentence represented in an integer vector.
        '''
        embedding = tf.Variable(
            tf.random_uniform((self.n_words, self.embedding_size), minval=-1, maxval=1),
            name='embedding')
        embedded_x = tf.nn.embedding_lookup(embedding, tf_x, name='embedded_x')


        cell_type = {
            'SimpleRNN':tf.contrib.rnn.BasicRNNCell,
            'GRU':      tf.contrib.rnn.GRUCell,
            'LSTM':     tf.contrib.rnn.BasicLSTMCell
            }

        cells = tf.contrib.rnn.MultiRNNCell([
            tf.contrib.rnn.DropoutWrapper(cell_type[self.cell](self.n_hidden_units), output_keep_prob=tf_keep_prob)
            for i in range(self.num_layers)])

        self.initial_state = tf.zero_state(self.batch_size, tf.float32)
        print('  << initial state >>', self.initial_state)

        outputs, self.final_state = tf.nn.dynamic_rnn(cells, embedded_x, initial_state=self.initial_state)
        print('  << outputs >>', outputs)
        print('  << final state >>', self.final_state)

        logits = tf.layers.dense(inputs=outputs[:,-1], units=1, activation=None, name='logits')
        logits = tf.squeeze(logits, name='logits_squeezed')
        print('  << logits >>', logits)

        y_prob = tf.nn.sigmoid(logits, name='prob')
        predictions = {
            'prob': y_prob,
            'label': tf.cast(tf.round(y_prob), tf.int32, name='label')
        }
        print('  << probabilities >>', predictions)


        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_y, logits=logits))
        train_op = tf.train.AdamOptimizer(self.lr).minimize(costm name='train_op')


    def train(self, data):
        self.sess.run(tf.global_variables_initializer())
        step = 0
        while step * self.batch_size < self.training_iters:

            # MNIST case, data=mnist
            if self.n_classes == 10: 
                batch_xs, batch_ys = data.train.next_batch(self.batch_size)
                batch_xs = batch_xs.reshape([self.batch_size, self.n_steps, self.n_inputs])
                feed = {
                    'x:0' : batch_xs,
                    'y:0' : batch_ys,
                }
                self.sess.run(self.train_op, feed_dict=feed)
                if step % 20 == 0:
                    print(self.sess.run(self.accuracy_op, feed_dict=feed))

            # HW
            else:
                batch_gen = batch_generator(
                    data[0], # X
                    data[1], # y
                    batch_size=self.batch_size,
                    shuffle=True
                )
                for i, (batch_x, batch_y) in enumerate(batch_gen):
                    feed = {
                        'x:0' : batch_x,
                        'y:0' : batch_y,
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

def batch_generator(X, y, batch_size=128, shuffle=False, random_seed=42):
    idx = np.arange(y.shape[0])
    y = one_hot(2, y)
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
    
    for i in range(0, X.shape[0], batch_size):
        yield (np.expand_dims(X[i:i+batch_size, :], axis=2), y[i:i+batch_size, :])


if __name__ == "__main__":
    # ''' does-it-run test ''' 
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # rnn = RNN( lr = 0.001,
    #             training_iters = 50000,
    #             batch_size = 128,
    #             n_inputs = 28,
    #             n_steps = 28,
    #             n_hidden_units = 128,
    #             n_classes = 10,
    #             cell = 'GRU')
    # rnn.train(mnist)
    # del rnn

    with open('data.pickle','rb') as f:
        train_data, train_labels, test_data, test_labels = pickle.load(f)
    # print(train_labels.shape)

    # batch_gen = batch_generator(
    #     train_data, # X
    #     train_labels, # y
    #     batch_size=256,
    #     shuffle=True
    # )
    # for i, (batch_x, batch_y) in enumerate(batch_gen):
    #     print(batch_x.shape, batch_y.shape)
    rnn = RNN( lr = 0.001,
                training_iters = 50000,
                batch_size = 128,
                n_inputs = 1,     # dim of word vector
                n_steps = 120,    # number of words in a sentence
                n_hidden_units = 128,
                n_classes = 2,    # pos or neg
                cell = 'GRU',
                embedding=True)
    rnn.train(data=(train_data,train_labels))
    del rnn

    pass