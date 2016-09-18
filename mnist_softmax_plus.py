# import source image data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# import tensorflow
import tensorflow as tf

## now using interactive session, allows interleaving operations that build/launch graphs
sess = tf.InteractiveSession()

# flattened MNIST images placeholder
x = tf.placeholder(tf.float32, [None, 784])

# cross entropy / loss function place holder
## moved up in expert tutorial
y_ = tf.placeholder(tf.float32, [None, 10])

# tf variables for weight and biases; init with 0
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

## launch model in a session and run operation to init variables
sess.run(tf.initialize_all_variables())

# model implementation - W*x+b
y = tf.nn.softmax(tf.matmul(x, W) + b)

# cross entropy fcn, mean(log(y)*loss function + 2nd y dim)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# back propagation training - minimize cross entropy w/ gradient descent at 0.5 lr
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# defines the initialization of all variables
## combined with init
##init = tf.initialize_all_variables()

# launch model in a session and run operation to init variables
## moved and combined
##sess = tf.Session()
##sess.run(init)

# training - run 1000 times
for i in range(1000):
##    batch_xs, batch_ys = mnist.train.next_batch(100)
##    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
## alternate training method(?) run 100 examples for wach iterations
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# compares prediction to correct label
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# calculates accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# print session accuracy
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

