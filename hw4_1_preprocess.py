# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:37:25 2019

@author: Finlab-Yi Hsien
"""


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
                                                        maxlen=120)
# numpy.ndarray (25000, 120)


test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=120)
# numpy.ndarray (25000, 120)

import numpy as np
np.savetxt("train_data.txt", train_data, fmt="%s")
np.savetxt("train_labels.txt", train_labels, fmt="%s")
np.savetxt("test_data.txt", test_data, fmt="%s")
np.savetxt("test_labels.txt", test_labels, fmt="%s")