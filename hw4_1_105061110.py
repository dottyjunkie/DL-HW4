import numpy as np
import tensorflow as tf
import hw4_1_preprocess
from tensorflow.examples.tutorials.mnist import input_data

# Good ref:
# http://colah.github.io/posts/2015-08-Understanding-LSTMs/


# train_data = np.loadtxt('train_data.txt', dtype=int)
# train_labels = np.loadtxt('train_labels.txt', dtype=int)
# test_data = np.loadtxt('test_data.txt', dtype=int)
# test_labels = np.loadtxt('test_labels.txt', dtype=int)


# testing sess
class LSTM():
    def __init__(self,
                random_seed=42
                ):
        self.lr = 0.001                  # learning rate
        self.training_iters = 100000     # train step 上限
        self.batch_size = 128            
        self.n_inputs = 28               # MNIST data input (img shape: 28*28)
        self.n_steps = 28                # time steps
        self.n_hidden_units = 128        # neurons in hidden layer
        self.n_classes = 10              # MNIST classes (0-9 digits)

        # g = tf.Graph()
        # with g.as_default():
        #     tf.set_random_seed(random_seed)
        #     self.build_model()
        #     self.init_op = tf.global_variables_initializer()
        #     # self.saver = tf.train.Saver()
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        # self.sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options))


        # **********att 2**************
        tf.set_random_seed(random_seed)
        self.build_model()
        self.init_op = tf.global_variables_initializer()
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


    def build_model(self):
        self.tf_x = tf.placeholder(tf.float32, shape=[None, self.n_steps, self.n_inputs], name='tf_x')
        self.tf_y = tf.placeholder(tf.float32, shape=[None, self.n_classes], name='tf_y')
        weights = {
            # (28, 128)
            'in' : tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units])),
            # (128, 10)
            'out' : tf.Variable(tf.random_normal([self.n_hidden_units, self.n_classes]))
        }

        biases ={
            # (128, )
            'in' : tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ])),
            # (10, )
            'out' : tf.Variable(tf.constant(0.1, shape=[self.n_classes, ]))
        }

        #  X: (128 batch, 28 steps, 28 inputs)
        X = tf.reshape(self.tf_x, [-1, self.n_inputs]) # (128,*28, 28 inputs)
        X_in = tf.matmul(X, weights['in']) + biases['in'] # (128*28, 128 hidden)
        X_in = tf.reshape(X_in, [-1, self.n_steps, self.n_hidden_units]) # (128 batch, 28 steps, 128 hidden)
        

        # lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        #     self.n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units)
        # lstm cell divided into two parts (c_state, m_state)
        # simple RNN has only m_state
        init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, states = tf.nn.dynamic_rnn(
            lstm_cell, X_in, initial_state=init_state, time_major=False)


        pred = tf.matmul(states[1], weights['out']) + biases['out']
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.tf_y),
            name='cost'
        )
        train_op = tf.train.AdamOptimizer(self.lr).minimize(cost, name='train_op')

        correct_pred = tf.equal(
            tf.argmax(pred, 1),
            tf.argmax(self.tf_y, 1),
            name='correct_preds'
        )
        accuracy = tf.reduce_mean(
            tf.cast(correct_pred, tf.float32),
            name='accuracy'
        )
    
    
    def train(self, mnist):
        self.sess.run(self.init_op)
        step = 0
        while step * self.batch_size < self.training_iters:
            batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)
            batch_xs = batch_xs.reshape([self.batch_size, self.n_steps, self.n_inputs])
            self.sess.run('train_op', feed_dict={
                self.tf_x: batch_xs,
                self.tf_y: batch_ys
            })
            if step % 20 == 0:
                print(self.sess.run('accuracy', feed_dict={
                    self.tf_x: batch_xs,
                    self.tf_y: batch_ys
                    }))
            step += 1

    def train_v2(self, mnist):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            # tf.initialize_all_variables() no long valid from
            # 2017-03-02 if using tensorflow >= 0.12
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                init = tf.initialize_all_variables()
            else:
                init = tf.global_variables_initializer()
            sess.run(init)
            step = 0
            while step * self.batch_size < self.training_iters:
                batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)
                batch_xs = batch_xs.reshape([self.batch_size, self.n_steps, self.n_inputs])
                sess.run(['train_op'], feed_dict={
                    self.tf_x: batch_xs,
                    self.tf_y: batch_ys
                })
                if step % 20 == 0:
                    print(sess.run(['accuracy'], feed_dict={
                    self.tf_x: batch_xs,
                    self.tf_y: batch_ys,
                    }))
                step += 1

    def predict(self):
        pass

def batch_generator():
    pass        



if __name__ == "__main__":
    # 导入数据
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    lstm = LSTM()
    lstm.train_v2(mnist)
    del lstm
