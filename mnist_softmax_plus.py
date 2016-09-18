# import source image data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# import tensorflow
import tensorflow as tf

## now using interactive session, allows interleaving operations that build/launch graphs
sess = tf.InteractiveSession()

### Create weight and bias functions, good to init with slight positive bias to avoid dead neurons
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# flattened MNIST images placeholder
x = tf.placeholder(tf.float32, [None, 784])

# cross entropy / loss function place holder
## moved up in expert tutorial
y_ = tf.placeholder(tf.float32, [None, 10])


# tf variables for weight and biases; init with 0
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

### Convolution and pooling functions, 0 pad convolution, 2x2 max pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


### First Convolution layer
# compute 32 features for each 5x5 patch
# shape: 1'st 2 are patch size, 1 input channel, 32 output channels
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# reshape x to 4d tensor, 2nd and 3ed dim are width/heght, 4th is # of color channels
x_image = tf.reshape(x, [-1,28,28,1])

# convolve x_image with weight and add bias
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
### End of First Convolution layer


### Second Convolution layer
# compute 64 features for each 5x5 patch
# shape here has 32 inputs for the first layer's output
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
### End of Second Convolution layer


### Dense Connection Layer
# after the convolution layers, we have 7x7 image size, this layer is fully connected layer to allow processing on the entire image
# Tensor is reshaped from prev pooling layer into batch of vectors; x weight; +bias; apply ReLU
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Add dropout to prevent overfitting , placeholder allows this to be on during training and off for testing
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
### End of Dense Connection Layer


### alternate model implementation used - replaced with Readout Layer
### model implementation - W*x+b
###y = tf.nn.softmax(tf.matmul(x, W) + b)

### Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
### End of Readout Layer


### Train and Eval Model
# gradient descent optimizer replaced with ADAM optimizer
# keep_prob and feed_dict to control dropout rate
# logging to every 100th iteration in training

## launch model in a session and run operation to init variables
sess.run(tf.initialize_all_variables())

### cross entropy now uses y_conv since the orig model 'y=...' was replaced
# cross entropy fcn, mean(log(y)*loss function + 2nd y dim)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

### Optimizer replaced with ADAM optimizer and corresponding weight
# back propagation training - minimize cross entropy w/ gradient descent at 0.5 lr
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# defines the initialization of all variables
## combined with init
##init = tf.initialize_all_variables()

# launch model in a session and run operation to init variables
## moved and combined
##sess = tf.Session()
##sess.run(init)

### moved in layered model
 ## training - run 1000 times
#for i in range(1000):
 ##    batch_xs, batch_ys = mnist.train.next_batch(100)
 ##    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
 ## alternate training method(?) run 100 examples for wach iterations
#    batch = mnist.train.next_batch(100)
#    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

### now uses y_conv since the orig model 'y=...' was replaced
# compares prediction to correct label
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

# calculates accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

### launch model in a session and init variables
sess.run(tf.initialize_all_variables())

### loop through 20000 training iterations, log every 100th
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

### replaced session accuracy display
## print session accuracy
#print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

