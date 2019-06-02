import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.legacy_seq2seq as seq2seq
from sklearn.model_selection import train_test_split


class en2fr():
	def __init__(self):
		with open('en.txt', 'r', encoding='utf-8') as f:
			self.en_lines = f.readlines()
		with open('en.txt', 'r', encoding='utf-8') as f:
			self.en_texts = f.read().split()

		with open('fr.txt', 'r', encoding='utf-8-sig') as f:
			self.fr_lines = f.readlines()
		with open('fr.txt', 'r', encoding='utf-8-sig') as f:
			self.fr_texts = f.read().split()

		self.x = [line.split() for line in self.en_lines]
		self.y = [line.split() for line in self.fr_lines]
		self.epochs = 10
		self.batch_size = 1120
		self.n_hidden = 256
		self.embed_size = 16 # original dim: 228 & 357

		self.word2int()
		self.build_model()
		self.saver = tf.train.Saver()
		self.X_train, self.X_test, self.y_train, self.y_test = \
			train_test_split(self.x, self.y, test_size=0.2, random_state=42)
		

	def word2int(self):
		unique_word = set(self.en_texts)
		self.char2numX = {word:i for i, word in enumerate(unique_word)}
		self.char2numX['<PAD>'] = len(self.char2numX)
		self.num2charX = dict(enumerate(self.char2numX))
		max_len_x = max([len(words) for words in self.x])
		self.x = [[self.char2numX[word] for word in words] + [self.char2numX['<PAD>']]*(max_len_x - len(words)) for words in self.x]
		self.x = np.array(self.x)


		unique_word = set(self.fr_texts)
		self.char2numY = {word:i for i, word in enumerate(unique_word)}
		self.char2numY['<GO>'] = len(self.char2numY)
		self.char2numY['<PAD>'] = len(self.char2numY)
		self.char2numY['<EOS>'] = len(self.char2numY)
		self.num2charY = dict(enumerate(self.char2numY))
		max_len_y = max([len(words) for words in self.y])
		self.y = [[self.char2numY['<GO>']] + [self.char2numY[word] for word in words] + [self.char2numY['<EOS>']] for words in self.y]
		self.y = [words + [self.char2numY['<PAD>']]*(max_len_y + 2 - len(words)) for words in self.y]
		# print(' '.join([self.num2charY[word] for word in self.y[0]]))
		self.y = np.array(self.y)

		self.x_seq_length = len(self.x[0])
		self.y_seq_length = len(self.y[0])- 1

	def batch_data(self, x, y, y_len, batch_size):
	    shuffle = np.random.permutation(len(x))
	    start = 0
	    x = x[shuffle]
	    y = y[shuffle]
	    y_len = y_len[shuffle]
	    while start + batch_size <= len(x):
	        yield x[start:start+batch_size], y[start:start+batch_size], y_len[start:start+batch_size]
	        start += batch_size

	def build_model(self):
		tf.reset_default_graph()
		self.sess = tf.Session()

		inputs = tf.placeholder(tf.int32, (None, self.x_seq_length), 'inputs')
		outputs = tf.placeholder(tf.int32, (None, None), 'outputs')
		targets = tf.placeholder(tf.int32, (None, None), 'targets')
		targets_length = tf.placeholder(tf.int32, (self.batch_size,), 'targets_length')

		# Embedding layers
		input_embedding = tf.Variable(tf.random_uniform((len(self.char2numX), self.embed_size), -1.0, 1.0), name='enc_embedding')
		output_embedding = tf.Variable(tf.random_uniform((len(self.char2numY), self.embed_size), -1.0, 1.0), name='dec_embedding')
		date_input_embed = tf.nn.embedding_lookup(input_embedding, inputs)
		date_output_embed = tf.nn.embedding_lookup(output_embedding, outputs)

		with tf.variable_scope("encoding") as encoding_scope:
		    # lstm_enc = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
		    lstm_enc = tf.contrib.rnn.GRUCell(self.n_hidden)
		    _, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=date_input_embed, dtype=tf.float32)

		with tf.variable_scope("decoding") as decoding_scope:
		    # lstm_dec = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
		    lstm_dec = tf.contrib.rnn.GRUCell(self.n_hidden)
		    dec_outputs, _ = tf.nn.dynamic_rnn(lstm_dec, inputs=date_output_embed, initial_state=last_state)

		self.logits = tf.contrib.layers.fully_connected(dec_outputs, num_outputs=len(self.char2numY), activation_fn=None) 
		with tf.name_scope("optimization"):
		    # masks = tf.ones([self.batch_size, self.y_seq_length])
		    masks = tf.sequence_mask(targets_length, self.y_seq_length, dtype=tf.float32)
		    self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, targets, weights=masks, name='loss')
		    self.optimizer = tf.train.AdamOptimizer(5e-3)
		    self.train_op = self.optimizer.minimize(self.loss, name='train_op')

		    # Gradient Clipping
		    # optimizer = tf.train.AdamOptimizer(2e-3)
		    # gradients = optimizer.compute_gradients(loss)
		    # capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
		    # train_op = optimizer.apply_gradients(capped_gradients)

	def train(self):
		y_len_train = np.array([ len([code for code in line  if code != 356]) for line in self.y_train])
		y_len_test = np.array([ len([code for code in line  if code != 356]) for line in self.y_test])

		self.sess.run(tf.global_variables_initializer())
		for epoch_i in range(self.epochs):
		    start_time = time.time()
		    for batch_i, (source_batch, target_batch, y_len) in enumerate(self.batch_data(
		    	self.X_train, self.y_train, y_len_train, self.batch_size)):
		        _, batch_loss, batch_logits = self.sess.run([self.train_op, self.loss, self.logits],
		            feed_dict = {
		             'inputs:0': source_batch,
		             'outputs:0': target_batch[:, :-1],
		             'targets:0': target_batch[:, 1:],
		             'targets_length:0': y_len
		             })

		    accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:,1:])
		    print('Epoch {:3} Loss: {:>6.4f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(
		    	epoch_i, batch_loss, accuracy, time.time() - start_time))

		self.saver.save(self.sess, "model/{}.ckpt".format('seq2seq'))
		print("model/{}.ckpt".format('seq2seq'))

		sample_output = True
		if sample_output:
			source_batch, target_batch, y_len = next(self.batch_data(self.X_test, self.y_test, y_len_test, self.batch_size))
			dec_input = np.zeros((len(source_batch), 1)) + self.char2numY['<GO>']
			for i in range(self.y_seq_length):
			    batch_logits = self.sess.run(self.logits,
			                feed_dict = {
			                 'inputs:0': source_batch,
			                 'outputs:0': dec_input,
			                 'targets_length:0': y_len
			                })
			    prediction = batch_logits[:,-1].argmax(axis=-1)
			    dec_input = np.hstack([dec_input, prediction[:,None]])

			accuracy = np.mean(dec_input[:,1:] == target_batch[:,1:])
			print('Accuracy on test set is: {:>6.4f}'.format(accuracy))
			num_preds = 10
			source_chars = [[self.num2charX[l] for l in sent if self.num2charX[l]!="<PAD>"] for sent in source_batch[:num_preds]]
			dest_chars = [[self.num2charY[l] for l in sent if self.num2charY[l]!="<PAD>" and self.num2charY[l]!="<GO>"] for sent in dec_input[:num_preds, 1:]]
			for date_in, date_out in zip(source_chars, dest_chars):
			    print(' '.join(date_in))
			    print('=> '+' '.join(date_out))
			    print()

	def translate(self):
		pass

if __name__ == '__main__':
	translator = en2fr()
	translator.train()

