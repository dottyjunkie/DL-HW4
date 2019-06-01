import bz2
import numpy as np
import tensorflow as tf
import pickle

# Good ref:
# http://funhacks.net/explore-python/Class/property.html
# https://blog.csdn.net/Yaokai_AssultMaster/article/details/70256621
# https://blog.csdn.net/u013041398/article/details/60955847


def TF_Embedding(dict_size, target):
    input_ids = tf.placeholder(dtype=tf.int32, shape=[None])

    embedding = tf.Variable(np.identity(dict_size, dtype=np.int32))
    input_embedding = tf.nn.embedding_lookup(embedding, input_ids)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # print(embedding.eval())
    # print(sess.run(input_embedding, feed_dict={input_ids:[1, 2, 3, 0, 3, 2, 1]}))
    print(sess.run(input_embedding, feed_dict={input_ids:target}))

def Embedding(dict_size, target):
    embedding = np.identity(dict_size, dtype=np.int32)
    dim = len(target.shape)

    if dim == 1:
        result = np.zeros((1, len(target), dict_size))
        for w in range(len(target)):
            result[0,w,:] = embedding[target[w],:]
        return result

    elif dim == 2:
        batch_size = target.shape[0]
        sentence_len = target.shape[1]
        result = np.zeros((batch_size, sentence_len, dict_size))
        for s in range(batch_size):
            for w in range(sentence_len):
                result[s,w,:] = embedding[target[s,w],:]
        return result

if __name__ == "__main__":
    with open('data.pickle','rb') as f:
        train_data, train_labels, test_data, test_labels = pickle.load(f)
    print(train_data[0,:])
    embedded = Embedding(dict_size=10000, target=train_data[0,:])
    # print(embedded[0,6,0:10])

    # tg = np.array([ [0,1],
    #                 [1,0]])
    # th = np.array([3,2,1])
    # embedded = Embedding(dict_size=5, target=tg)