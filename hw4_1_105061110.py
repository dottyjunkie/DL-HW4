import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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

        self.training_acc = []
        self.training_loss = []
        self.testing_acc = []
        self.testing_loss = []
        
        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(42)
            self.build_model()
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()


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

        self.initial_state = cells.zero_state(self.batch_size, tf.float32)
        # print('  << initial state >>', self.initial_state)
        # -> shape=(128, 128) dtype=float32

        outputs, self.final_state = tf.nn.dynamic_rnn(cells, embedded_x, initial_state=self.initial_state)
        # print('  << outputs >>', outputs)
        # -> shape=(128, 120, 128) dtype=float32
        # print('  << final state >>', self.final_state)
        # -> shape=(128, 128) dtype=float32

        logits = tf.layers.dense(inputs=outputs[:,-1], units=1, activation=None, name='logits')
        logits = tf.squeeze(logits, name='logits_squeezed')
        # print('  << logits >>', logits)
        # -> shape=(128,) dtype=float32

        y_prob = tf.nn.sigmoid(logits, name='prob')
        predictions = {
            'prob': y_prob,
            'label': tf.cast(tf.round(y_prob), tf.int32, name='label')
        }
        # print('  << probabilities >>', predictions)
        # -> 'prob': shape=(128,) dtype=float32
        # -> 'label': shape=(128,) dtype=float32
        

        cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(tf_y, tf.float32), logits=logits), name='cost'
            )
        train_op = tf.train.AdamOptimizer(self.lr).minimize(cost, name='train_op')


        correct_predictions = tf.equal(predictions['label'], tf_y, name='correct_preds')
        accuracy = tf.reduce_mean(
            tf.cast(correct_predictions, tf.float32),
            name='accuracy'
            )
    # preds = rnn.predict(test_data)
    # y_true = test_labels[:len(preds)]
    # print("Test Acc.: {:.3f}".format(np.sum(preds == y_true) / len(y_true)))

    def train(self, training_set, validation_set=None):
        X = training_set[0]
        y = training_set[1]

        with tf.Session(graph=self.g) as sess:
            sess.run(self.init_op)
            acc_batch = 1 # accumulated batch number

            for epoch in range(self.epochs):
                state = sess.run(self.initial_state)
                epoch_loss = []
                epoch_acc = []

                batch_gen = batch_generator(X, y, batch_size=self.batch_size)
                for i, (batch_x, batch_y) in enumerate(batch_gen):
                    feed = {
                        'tf_x:0': batch_x,
                        'tf_y:0': batch_y,
                        'tf_keep_prob:0': 0.5,
                        self.initial_state: state 
                    }

                    loss, acc, _, state = sess.run(
                        ['cost:0', 'accuracy:0', 'train_op', self.final_state],
                        feed_dict=feed
                    )
                    epoch_loss.append(loss)
                    epoch_acc.append(acc)

                    if acc_batch % 25 == 0:
                        print('Epoch: {}/{} Acc_batch: {} | Training_loss: {:.5f} Training_acc:{:.3f}'.format(
                            epoch+1, self.epochs, acc_batch, loss, acc))

                    acc_batch += 1 # count every batch

                self.training_loss.append(sum(epoch_loss) / len(epoch_loss)) 
                self.training_acc.append(sum(epoch_acc) / len(epoch_acc))

                if validation_set is not None:
                    test_X = validation_set[0]
                    test_y = validation_set[1]
                    batch_gen = batch_generator(test_X, test_y, batch_size=self.batch_size)
                    epoch_loss = []
                    epoch_acc = []
                    for i, (batch_x, batch_y) in enumerate(batch_gen):
                        feed = {
                            'tf_x:0' : batch_x,
                            'tf_y:0' : batch_y,
                            'tf_keep_prob:0': 1.0
                        }
                        loss, acc = sess.run(
                            ['cost:0', 'accuracy:0'],
                             feed_dict=feed
                        )
                        epoch_loss.append(loss)
                        epoch_acc.append(acc)

                    self.testing_loss.append(sum(epoch_loss) / len(epoch_loss)) 
                    self.testing_acc.append(sum(epoch_acc) / len(epoch_acc))
                    foo = "Acc_batch: {}".format(acc_batch)
                    print('Epoch: {}/{} {} | Testing_loss : {:.5f} Testing_acc :{:.3f}'.format(
                            epoch+1, self.epochs, ' '*len(foo), self.testing_loss[-1], self.testing_acc[-1]))



                if (epoch+1) % self.epochs == 0: # save the model at last epoch
                    self.saver.save(sess, "model/{}-{}.ckpt".format(self.cell, epoch))
                    print("\"{}-{}.ckpt\" saved".format(self.cell, epoch))


    def predict(self, X, return_prob=False):
        preds=[]
        with tf.Session(graph=self.g) as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('model/'))
            test_state = sess.run(self.initial_state)
            batch_gen = batch_generator(X, None, batch_size=self.batch_size)
            
            for i, batch_x in enumerate(batch_gen, 1):
                feed = {
                    'tf_x:0': batch_x,
                    'tf_keep_prob:0': 1.0,
                    self.initial_state: test_state 
                }

                if return_prob:
                    pred, test_state = sess.run(['prob:0', self.final_state], feed_dict=feed)
                else:
                    pred, test_state = sess.run(['label:0', self.final_state], feed_dict=feed)

                preds.append(pred)

        return np.concatenate(preds)

def batch_generator(X, y=None, batch_size=64):
    n_batches = len(X) // batch_size # floor division
    X = X[:n_batches*batch_size] # trim the size to be the multiple of batch_size

    if y is not None:
        y = y[:n_batches*batch_size]

    for idx in range(0, len(X), batch_size):
        if y is not None:
            yield X[idx:idx+batch_size], y[idx:idx+batch_size]
        else:
            yield X[idx:idx+batch_size]

def plot_ROC():
    pass

def plot_PRC():
    pass

def plot_AUROC():
    pass

def plot_AUPRC():
    pass

if __name__ == "__main__":
    with open('data.pickle','rb') as f:
        train_data, train_labels, test_data, test_labels = pickle.load(f)

    epochs = 5
    rnn = RNN(  n_words = 10000,
                seq_len = 120,
                n_hidden_units = 128,
                num_layers = 1,
                batch_size = 100,
                lr = 0.001,
                embedding_size = 256,
                epochs = epochs,
                cell = 'GRU')
    rnn.train(training_set=(train_data, train_labels),
        validation_set=(test_data, test_labels))


    # Plotting
    fig1 = plt.figure(1)
    plt.plot(range(1,epochs+1), rnn.training_loss, label='training loss')
    plt.plot(range(1,epochs+1), rnn.testing_loss, label='testing loss')
    plt.xlabel('epoch')
    plt.ylabel('Cross entropy')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()

    fig2 = plt.figure(2)
    plt.plot(range(1,epochs+1), rnn.training_acc, label='training acc')
    plt.plot(range(1,epochs+1), rnn.testing_acc, label='testing acc')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Testing
    # preds = rnn.predict(test_data)
    # y_true = test_labels[:len(preds)]
    # print("Test Acc.: {:.3f}".format(np.sum(preds == y_true) / len(y_true)))
    del rnn
