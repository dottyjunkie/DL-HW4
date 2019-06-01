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
