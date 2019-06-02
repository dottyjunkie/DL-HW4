# Ref: https://github.com/sachinruk/deepschool.io/blob/master/DL-Keras_Tensorflow/Lesson%2019%20-%20Seq2Seq%20-%20Date%20translator%20-%20Solutions.ipynb
import numpy as np
import matplotlib.pyplot as plt

import random
import json
import os
import time

from faker import Faker
import babel
from babel.dates import format_date

import tensorflow as tf

import tensorflow.contrib.legacy_seq2seq as seq2seq

from sklearn.model_selection import train_test_split


with open('en.txt', 'r', encoding='utf-8') as f:
  en_lines = f.read().split('\n')
  # print('num_lines:{}'.format(len(en_lines)))

with open('fr.txt', 'r', encoding='utf-8-sig') as f:
  fr_lines = f.read().split('\n')

# x = en_lines
# y = fr_lines
en_min = 60
en_max = 80
fr_min = 60
fr_max = 90

x = []
y = []
for i, (en_line, fr_line) in enumerate(zip(en_lines, fr_lines)):
  if len(en_line)>en_min and len(en_line)<en_max and len(fr_line)>fr_min and len(fr_line)<fr_max:
    x.append(en_line)
    y.append(fr_line)
print('Saved percentage: {}%'.format(len(x) / len(en_lines)))
print('Saved percentage: {}%'.format(len(y) / len(fr_lines)))
# print(x[:2])
# print(y[:2])

u_characters = set(' '.join(x))
# print(u_characters)
char2numX = dict(zip(u_characters, range(len(u_characters))))
u_characters = set(' '.join(y))
char2numY = dict(zip(u_characters, range(len(u_characters))))


char2numX['<PAD>'] = len(char2numX)
num2charX = dict(enumerate(char2numX))
# print(num2charX)
max_len_x = max([len(line) for line in x])
x = [[char2numX['<PAD>']]*(max_len_x - len(line)) +[char2numX[ch] for ch in line] for line in x]
# print(''.join([num2charX[ch] for ch in x[0]]))
x = np.array(x)


char2numY['<GO>'] = len(char2numY)
char2numY['<PAD>'] = len(char2numY)
num2charY = dict(enumerate(char2numY))
max_len_y = max([len(line) for line in y])
y = [[char2numY['<GO>']] + [char2numY[ch] for ch in line] for line in y]
y = [[char2numY['<PAD>']]*(max_len_y + 1 - len(line)) + line for line in y]
# print(''.join([num2charY[ch] for ch in y[4]]))
y = np.array(y)
# print(y.shape)

x_seq_length = len(x[0])
y_seq_length = len(y[0])- 1


def batch_data(x, y, batch_size):
    shuffle = np.random.permutation(len(x))
    start = 0
#     from IPython.core.debugger import Tracer; Tracer()()
    x = x[shuffle]
    y = y[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start+batch_size], y[start:start+batch_size]
        start += batch_size

epochs = 2
batch_size = 500
nodes = 32
embed_size = 16

tf.reset_default_graph()
sess = tf.Session()

# Tensor where we will feed the data into graph
inputs = tf.placeholder(tf.int32, (None, x_seq_length), 'inputs')
outputs = tf.placeholder(tf.int32, (None, None), 'output')
targets = tf.placeholder(tf.int32, (None, None), 'targets')

# Embedding layers
input_embedding = tf.Variable(tf.random_uniform((len(char2numX), embed_size), -1.0, 1.0), name='enc_embedding')
output_embedding = tf.Variable(tf.random_uniform((len(char2numY), embed_size), -1.0, 1.0), name='dec_embedding')
date_input_embed = tf.nn.embedding_lookup(input_embedding, inputs)
date_output_embed = tf.nn.embedding_lookup(output_embedding, outputs)

with tf.variable_scope("encoding") as encoding_scope:
    lstm_enc = tf.contrib.rnn.BasicLSTMCell(nodes)
    _, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=date_input_embed, dtype=tf.float32)

with tf.variable_scope("decoding") as decoding_scope:
    lstm_dec = tf.contrib.rnn.BasicLSTMCell(nodes)
    dec_outputs, _ = tf.nn.dynamic_rnn(lstm_dec, inputs=date_output_embed, initial_state=last_state)
#connect outputs to 
logits = tf.contrib.layers.fully_connected(dec_outputs, num_outputs=len(char2numY), activation_fn=None) 
with tf.name_scope("optimization"):
    # Loss function
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, y_seq_length]))
    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

# print(dec_outputs.get_shape().as_list()) # [None, None, 32]
# print(last_state[0].get_shape().as_list()) # [None, 32]
# print(inputs.get_shape().as_list()) # [None, 102]
# print(date_input_embed.get_shape().as_list()) # [None, 102, 16]

# print(x.shape)
# print(y.shape)
# x = x[:25000, :]
# y = y[:25000, :]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


sess.run(tf.global_variables_initializer())
epochs = 20
for epoch_i in range(epochs):
    start_time = time.time()
    for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
        _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
            feed_dict = {inputs: source_batch,
             outputs: target_batch[:, :-1],
             targets: target_batch[:, 1:]})
    accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:,1:])
    print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(
    	epoch_i, batch_loss, accuracy, time.time() - start_time))


source_batch, target_batch = next(batch_data(X_test, y_test, batch_size))
dec_input = np.zeros((len(source_batch), 1)) + char2numY['<GO>']
for i in range(y_seq_length):
    batch_logits = sess.run(logits,
                feed_dict = {inputs: source_batch,
                 outputs: dec_input})
    prediction = batch_logits[:,-1].argmax(axis=-1)
    dec_input = np.hstack([dec_input, prediction[:,None]]) 
print('Accuracy on test set is: {:>6.3f}'.format(np.mean(dec_input == target_batch)))
num_preds = 2
source_chars = [[num2charX[l] for l in sent if num2charX[l]!="<PAD>"] for sent in source_batch[:num_preds]]
dest_chars = [[num2charY[l] for l in sent] for sent in dec_input[:num_preds, 1:]]
for date_in, date_out in zip(source_chars, dest_chars):
    print(''.join(date_in)+' => '+''.join(date_out))


source_batch, target_batch = next(batch_data(X_train, y_train, batch_size))
num_preds = 10
source_chars = [[num2charX[l] for l in sent if num2charX[l]!="<PAD>"] for sent in source_batch[:num_preds]]
dest_chars = [[num2charY[l] for l in sent] for sent in dec_input[:num_preds, 1:]]

for date_in, date_out in zip(source_chars, dest_chars):
    print(''.join(date_in)+' => '+''.join(date_out))