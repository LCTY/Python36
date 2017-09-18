import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.reset_default_graph()
with tf.Session() as sess:
    save_path = os.getcwd() + "/tmp"
    saver = tf.train.import_meta_graph(save_path + '/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint(save_path))

    x = tf.get_default_graph().get_tensor_by_name('x:0')
    y_ = tf.get_default_graph().get_tensor_by_name('y_:0')
    keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
    y_conv = tf.get_default_graph().get_tensor_by_name('y_conv:0')

    batch = mnist.train.next_batch(200)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_accuacy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    print(train_accuacy)
