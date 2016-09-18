# import source image data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# import tensorflow
import tensorflow as tf

# flattened MNIST images placeholder
x = tf.placeholder(tf.float32, [None, 784])

# tf variables for weight and biases; init with 0
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# model implementation - W*x+b
y = tf.nn.softmax(tf.matmul(x, W) + b)

# cross entropy / loss function place holder
y_ = tf.placeholder(tf.float32, [None, 10])

# cross entropy fcn, mean(log(y)*loss function + 2nd y dim)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# back propagation training - minimize cross entropy w/ gradient descent at 0.5 lr
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# defines the initialization of all variables
init = tf.initialize_all_variables()

# launch model in a session and run operation to init variables
sess = tf.Session()
sess.run(init)

# training - run 1000 times
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# compares prediction to correct label
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# calculates accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# print session accuracy
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

