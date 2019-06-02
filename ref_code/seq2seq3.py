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
  en_lines = f.readlines()
with open('en.txt', 'r', encoding='utf-8') as f:
  en_texts = f.read().split()
# print(en_texts[:20])

with open('fr.txt', 'r', encoding='utf-8-sig') as f:
  fr_lines = f.readlines()
with open('fr.txt', 'r', encoding='utf-8-sig') as f:
  fr_texts = f.read().split()

x = [line.split() for line in en_lines]
y = [line.split() for line in fr_lines]

u_characters = set(en_texts)
char2numX = dict(zip(u_characters, range(len(u_characters))))
u_characters = set(fr_texts)
char2numY = dict(zip(u_characters, range(len(u_characters))))


char2numX['<PAD>'] = len(char2numX)
# print('n_char2numX: {}'.format(len(char2numX))) # 228
num2charX = dict(enumerate(char2numX))
# print(num2charX)
max_len_x = max([len(words) for words in x])
# x = [[char2numX['<PAD>']]*(max_len_x - len(words)) +[char2numX[word] for word in words] for words in x]
x = [[char2numX[word] for word in words] + [char2numX['<PAD>']]*(max_len_x - len(words)) for words in x]
# print(' '.join([num2charX[word] for word in x[0]]))
x = np.array(x)
# print(x.shape)


char2numY['<GO>'] = len(char2numY)
char2numY['<PAD>'] = len(char2numY)
# print('n_char2numY: {}'.format(len(char2numY))) # 357
num2charY = dict(enumerate(char2numY))
max_len_y = max([len(words) for words in y])
y = [[char2numY['<GO>']] + [char2numY[word] for word in words] for words in y]
# y = [[char2numY['<PAD>']]*(max_len_y + 1 - len(words)) + words for words in y]
y = [words + [char2numY['<PAD>']]*(max_len_y + 1 - len(words)) for words in y]
# print(' '.join([num2charY[word] for word in y[0]]))
y = np.array(y)
# print(y.shape)

x_seq_length = len(x[0])
y_seq_length = len(y[0])- 1


def batch_data(x, y, y_len, batch_size):
    shuffle = np.random.permutation(len(x))
    start = 0
#     from IPython.core.debugger import Tracer; Tracer()()
    x = x[shuffle]
    y = y[shuffle]
    y_len = y_len[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start+batch_size], y[start:start+batch_size], y_len[start:start+batch_size]
        start += batch_size


epochs = 2 # 2
batch_size = 1120
nodes = 256 # 32 < 64 < 100
embed_size = 16 # original dim: 228 & 357

tf.reset_default_graph()
sess = tf.Session()


# Tensor where we will feed the data into graph
inputs = tf.placeholder(tf.int32, (None, x_seq_length), 'inputs')
outputs = tf.placeholder(tf.int32, (None, None), 'output')
targets = tf.placeholder(tf.int32, (None, None), 'targets')
targets_length = tf.placeholder(tf.int32, (batch_size,), 'targets_length')
# Embedding layers
input_embedding = tf.Variable(tf.random_uniform((len(char2numX), embed_size), -1.0, 1.0), name='enc_embedding')
output_embedding = tf.Variable(tf.random_uniform((len(char2numY), embed_size), -1.0, 1.0), name='dec_embedding')
date_input_embed = tf.nn.embedding_lookup(input_embedding, inputs)
date_output_embed = tf.nn.embedding_lookup(output_embedding, outputs)

with tf.variable_scope("encoding") as encoding_scope:
    # lstm_enc = tf.contrib.rnn.BasicLSTMCell(nodes)
    lstm_enc = tf.contrib.rnn.GRUCell(nodes)
    _, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=date_input_embed, dtype=tf.float32)

with tf.variable_scope("decoding") as decoding_scope:
    # lstm_dec = tf.contrib.rnn.BasicLSTMCell(nodes)
    lstm_dec = tf.contrib.rnn.GRUCell(nodes)
    dec_outputs, _ = tf.nn.dynamic_rnn(lstm_dec, inputs=date_output_embed, initial_state=last_state)
#connect outputs to 
logits = tf.contrib.layers.fully_connected(dec_outputs, num_outputs=len(char2numY), activation_fn=None) 
with tf.name_scope("optimization"):
    # masks = tf.ones([batch_size, y_seq_length])
    masks = tf.sequence_mask(targets_length, y_seq_length, dtype=tf.float32)
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, weights=masks)
    # optimizer = tf.train.RMSPropOptimizer(1e-3)
    optimizer = tf.train.AdamOptimizer(2e-3)
    # train_op = optimizer.minimize(loss)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(loss)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

# print(dec_outputs.get_shape().as_list()) # [None, None, 32]
# print(last_state[0].get_shape().as_list()) # [None, 32]
# print(inputs.get_shape().as_list()) # [None, 17]
# print(date_input_embed.get_shape().as_list()) # [None, 17, 64]
# print(x.shape) # (137760, 17)
# print(y.shape) # (137760, 24)



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
y_len_train = np.array([ len([code for code in line  if code != 356]) for line in y_train])
y_len_test = np.array([ len([code for code in line  if code != 356]) for line in y_test])

sess.run(tf.global_variables_initializer())
epochs = 50
for epoch_i in range(epochs):
    start_time = time.time()
    for batch_i, (source_batch, target_batch, y_len) in enumerate(batch_data(X_train, y_train, y_len_train, batch_size)):
        _, batch_loss, batch_logits = sess.run([train_op, loss, logits],
            feed_dict = {
             inputs: source_batch,
             outputs: target_batch[:, :-1],
             targets: target_batch[:, 1:],
             targets_length: y_len
             }
        )

    true_target_batch = target_batch.copy()
    true_target_batch[true_target_batch == 356] = -1
    result = batch_logits.argmax(axis=-1) == true_target_batch[:,1:]
    # result = batch_logits.argmax(axis=-1) == target_batch[:,1:]
    accuracy = np.mean(result)
    print('Epoch {:3} Loss: {:>6.4f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(
    	epoch_i, batch_loss, accuracy, time.time() - start_time))


source_batch, target_batch, y_len = next(batch_data(X_test, y_test, y_len_test, batch_size))
dec_input = np.zeros((len(source_batch), 1)) + char2numY['<GO>']
for i in range(y_seq_length):
    batch_logits = sess.run(logits,
                feed_dict = {
                 inputs: source_batch,
                 outputs: dec_input,
                 targets_length: y_len
                }
    )
    prediction = batch_logits[:,-1].argmax(axis=-1)
    dec_input = np.hstack([dec_input, prediction[:,None]])
target_batch[target_batch == 356] = -1
print('Accuracy on test set is: {:>6.4f}'.format(np.mean(dec_input[:,1:] == target_batch[:,1:])))
num_preds = 10
source_chars = [[num2charX[l] for l in sent if num2charX[l]!="<PAD>"] for sent in source_batch[:num_preds]]
dest_chars = [[num2charY[l] for l in sent if num2charY[l]!="<PAD>" and num2charY[l]!="<GO>"] for sent in dec_input[:num_preds, 1:]]
for date_in, date_out in zip(source_chars, dest_chars):
    print(' '.join(date_in))
    print('=> '+' '.join(date_out))
    print()
