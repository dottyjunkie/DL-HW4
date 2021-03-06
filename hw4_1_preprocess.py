# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:37:25 2019

@author: Finlab-Yi Hsien
"""

def preprocess(maxlen=120):
	import tensorflow as tf


	imdb = tf.keras.datasets.imdb

	(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


	# A dictionary mapping words to an integer index
	word_index = imdb.get_word_index()

	# The first indices are reserved
	word_index = {k:(v+3) for k,v in word_index.items()} 
	word_index["<PAD>"] = 0
	word_index["<START>"] = 1
	word_index["<UNK>"] = 2  # unknown
	word_index["<UNUSED>"] = 3


	# train_labels: numpy.ndarray (25000,) binary

	train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
	                                                        value=word_index["<PAD>"],
	                                                        padding='post',
	                                                        maxlen=maxlen)
	# numpy.ndarray (25000, 120)


	test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,
	                                                       value=word_index["<PAD>"],
	                                                       padding='post',
	                                                       maxlen=maxlen)
	# numpy.ndarray (25000, 120)
	f = open('maxlen-{}.pickle'.format(maxlen), 'wb')
	pickle.dump((train_data, train_labels, test_data, test_labels), f)
	f.close()


if __name__ == "__main__":
	import pickle
	maxlens = [120, 256]
	for maxlen in maxlens:
		preprocess(maxlen=maxlen)

