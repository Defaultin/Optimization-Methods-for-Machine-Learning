# # Exercise 7.2 Convolutional Neural Network for image classification
# Goals: Build a convolutional neural network to classify the MNIST dataset.
!pip install tensorflow==1.6.0


# Import MNIST data:
# 65k images, 55k in training set, 10k in test set
# images are vectors of size 784 (28x28 px)
# labels are vectors of size 10 (numbers 0-9 in one_hot format)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

num_input = 784 # number of pixels
num_output = 10 # in 10dim unit vectors converted labels

# Meta Parameters, insert a different numbers if you want to
# For convolution [block1, block2]
num_conv    = [32,64] # Number of convolution filters = features to extract from images
size_conv   = [3,3] # Size of convolution filters = window of the image the filter sees
stride_conv = [1,1] # Stride of convolution filter = number of steps to move the filter
size_pool   = [2,2] # Size of the pooling filter
stride_pool = [2,2] # Stride of the pooling filter

# For dense [layer1]
size_fc = [128] # Size of the fully connected layer


tf.reset_default_graph()

# Build your computation graph below
# Start with defining placeholders and optimization variables (filters)
x = tf.placeholder(tf.float32 ,[None, num_input], name='x')
y = tf.placeholder(tf.float32, [None, num_output], name='y')

# Reshape input vectors to 28x28x1 tensor (image res and channel)
x_res = tf.reshape(x, shape=[-1, 28, 28, 1])

# Define Conv Filter Variables
conv_filter_0 = tf.Variable(tf.random_normal([size_conv[0], size_conv[0], 1, num_conv[0]]))
conv_filter_1 = tf.Variable(tf.random_normal([size_conv[1], size_conv[1], num_conv[0], num_conv[1]]))
conv_bias_l0  = tf.Variable(tf.random_normal([num_conv[0]]))
conv_bias_l1  = tf.Variable(tf.random_normal([num_conv[1]]))

# First Conv-Pool-ReLU Block
conv_l0      = tf.nn.conv2d(input=x_res, filter=conv_filter_0, strides=[1,stride_conv[0], stride_conv[0], 1], padding='SAME')
bias_l0      = tf.nn.bias_add(conv_l0,conv_bias_l0)
max_pool_l0  = tf.nn.max_pool(bias_l0, ksize=[1, size_pool[0], size_pool[0], 1], strides=[1, stride_pool[0], stride_pool[0], 1], padding='SAME')
relu_l0      = tf.nn.relu(max_pool_l0)
# Second Conv-Pool-RelU Block
conv_l1      = tf.nn.conv2d(input=relu_l0, filter=conv_filter_1, strides=[1,stride_conv[1], stride_conv[1], 1], padding='SAME')
bias_l1      = tf.nn.bias_add(conv_l1,conv_bias_l1)
max_pool_l1  = tf.nn.max_pool(bias_l1, ksize=[1, size_pool[1], size_pool[1], 1], strides=[1, stride_pool[1], stride_pool[1], 1], padding='SAME')
relu_l1      = tf.nn.relu(max_pool_l1)

# Resize for Dense Layer
size_relu_l1 = tf.cast(relu_l1.shape[1] * relu_l1.shape[2] * relu_l1.shape[3], dtype=tf.int32)
relu_l1 = tf.reshape(relu_l1, [-1, size_relu_l1])

# Define Dense Layer Variables
fc_A1 = tf.Variable(tf.random_normal([size_relu_l1, size_fc[0]], seed=1), name='A1')
fc_b1 = tf.Variable(tf.random_normal([size_fc[0]], seed=1), name='b1')
fc_A2 = tf.Variable(tf.random_normal([size_fc[0], num_output], seed=1), name='A2')
fc_b2 = tf.Variable(tf.random_normal([num_output], seed=1), name='b2')

# First Dense Layer
fc_l1 = tf.nn.relu(tf.add(tf.matmul(relu_l1, fc_A1), fc_b1))
# Output Layer
fc_l2 = tf.add(tf.matmul(fc_l1, fc_A2), fc_b2)

# Predicted Values with Softmax
pred = tf.nn.softmax(fc_l2)

#Losses
loss_ce = tf.losses.softmax_cross_entropy(logits=fc_l2, onehot_labels=y)
#loss_ms = tf.losses.mean_squared_error(pred, y)

# Optimizer
optimizer_grad = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss_ce) 


# You may wanna have a look at your graph
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())


# Run a session to train your network and calculate ratio of succes
iterations = 250
batch_size = 200
with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1, iterations+1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer_grad,feed_dict={x: batch_x, y: batch_y})
            if i % 10 == 0:
                l, p = sess.run([loss_ce, pred], feed_dict={x: batch_x, y: batch_y})
                print('GradDesc Iteration', i, ': Loss of', l,
                      ', ROS on test set:', np.mean(np.argmax(batch_y, axis=-1) == np.argmax(p, axis=-1)))
                
        p_test = sess.run(pred, feed_dict={x: mnist.test.images})
        print('ROS on test set:', np.mean(np.argmax(mnist.test.labels, axis=-1) == np.argmax(p_test, axis=-1)))