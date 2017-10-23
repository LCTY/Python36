import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_varible(shape, var_name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=var_name)


def bias_variable(shape, var_name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=var_name)


def conv2d(x, W, conv2d_name=None):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=conv2d_name)


def max_pool_2x2(x, max_pool_name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=max_pool_name)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("Download Done!")


# paras
W_conv1 = weight_varible([5, 5, 1, 32], "W_conv1")
b_conv1 = bias_variable([32], "b_conv1")

# conv layer-1
x = tf.placeholder(tf.float32, [None, 784], name="x")
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 'conv2d1') + b_conv1, name="h_conv1")
h_pool1 = max_pool_2x2(h_conv1, 'max_pool1')

# conv layer-2
W_conv2 = weight_varible([5, 5, 32, 64], "W_conv2")
b_conv2 = bias_variable([64], "b_conv2")

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 'conv2d2') + b_conv2, name="h_conv2")
h_pool2 = max_pool_2x2(h_conv2, 'max_pool2')

# full connection
W_fc1 = weight_varible([7 * 7 * 64, 1024], "W_fc1")
b_fc1 = bias_variable([1024], "b_fc1")

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="h_fc1")

# dropout
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name="h_fc1_drop")

# output layer: softmax
W_fc2 = weight_varible([1024, 10], "W_fc2")
b_fc2 = bias_variable([10], "b_fc2")

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name="y_conv")
y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

# model training
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the variables to disk.
with tf.Session() as sess:
    sess.run(init_op)

    # Do some work with the model.
    end = 0
    for i in range(1):
        batch = mnist.train.next_batch(10)
        if i % 200 == 0:
            train_accuacy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            print("step %d, training accuracy %.16g" % (i, train_accuacy))
            # stop if accuracy high enough
            # if train_accuacy >= 0.999:
            #     end += 1
            #     if end >= 5:
            #         break
            # else:
            #     end = 0
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    # Save the variables to disk.
    save_path = "tmp/"
    if not os.path.exists("tmp/"):
        os.makedirs(save_path)
    saver.save(sess, save_path+"model.ckpt")
    tf.train.write_graph(sess.graph_def, save_path,"model_age.pb", as_text=True)
    print("Model saved in file: %s" % save_path+"model.ckpt")
