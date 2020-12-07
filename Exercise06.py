!pip install tensorflow==1.6.0

import tensorflow as tf
import numpy as np

# This only build the computaion graph of total = a + b
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)
# No sum is calculated. Only the symbolic version is created. We can visualize these with Tensorboard
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
# To open Tensorboard:
#  - using a local Python distribution: open your working dictionary in terminal and run 'tensorboard --logdir=.'
#  - using Jupyter-Lab on coxeter: open new terminal from jupyter notebook and run 'tensorboard --logdir=.'
# Open the provided Link in your browser.

# Some additional nodes for our computation graph
c = tf.placeholder(tf.float32, []) # Placeholder provides a dummy variable that can be feed with data later
d = tf.multiply(tf.add(tf.multiply(c, c), a), b)
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
# Go to you Tensorboard and refresh. There should be a second graph showing the computation of d.

# Resetting our computational graph to work on the neural network layer.
tf.reset_default_graph()
num_input = 1 # Setting up dimension of input data, for  example the number of feats. Just specify some number.
num_output = 1 # Dim for output data

# Tensorflow works with tensors (multidiminsional arrays) that need to be initialized first.
# The brackets define the dimension of the tensor: [] constant, [2] 1-dim (vector) of size 2, [2,3] 2-dim matrix with 2 rows, 3 columns, ...
# None specifies an unknown size that can be anything.
x = tf.placeholder(tf.float32, [num_input, None], name='x') # Creates x as a tensor with num_input rows for feats_data and a unspecified number of columns for the number of data points that you have.
y = tf.placeholder(tf.float32, [num_output, None], name='y') # Instead of none, we could also use some argument like num_data='insert number of datapoints here'

# Build one Layer of a neural network
# We need to initialize A and b as Variables, so that the optimizer knows what he should optimize.
# Initial values are set to be random.
# Make sure that the dimensions fit!
A = tf.Variable(tf.random_normal([num_output, num_input]), name='A') 
b = tf.Variable(tf.random_normal([num_output, 1]), name='b')
pred = tf.nn.tanh(tf.add(tf.matmul(A, x), b)) # Could also use some other activation function like rectified or exponential linear (relu, elu).

# Refresh your Tensorboard and have a look at it.
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

# Add nodes to the computational graph that handle the loss calculation and optimization.
loss = tf.losses.mean_squared_error(pred, y) # Could also use absolute_difference, hinge_loss, cross_entropy,...

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(loss) # Other possibilites: AdamOptimizer,...
optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, options={'maxiter': 100}) # there is also a way to incorporate BFGS, session code slightly different (see comments below)

# Refresh your Tensorboard and have a look at it.
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

# Some toy data to fit
data_x = np.arange(-5, 5, 0.5)
data_y = np.tanh(data_x)


iterations = 1000 # Number of iterations, not used for BFGS
with tf.Session() as sess: # Start a session to actually compute stuff
        sess.run(tf.global_variables_initializer()) # Initialize variables
        A_init, b_init = sess.run([A, b]) # Get initialized variables
        print('Randomly initialized parameters A,b:', A_init, b_init) # print them to see the starting values
        
        # Choose one of the two variants below!
        # Use this loop for Tensorflow optimizers like GradientDescent or Adam
        #for j in range(iterations): # They perform one step each iteration, let them run for a number of iterations
         #   p, l, _ = sess.run([pred,loss,optimizer],feed_dict={x: data_x.reshape(1,20),y: data_y.reshape(1,20)}) # feed your data to the neural network and optimize
        
        # Use this for Scipy BFGS optimizer
        optimizer.minimize(sess, feed_dict={x: data_x.reshape(1, 20), y: data_y.reshape(1, 20)})
        p, l = sess.run([pred, loss], feed_dict={x: data_x.reshape(1, 20), y: data_y.reshape(1, 20)})
        
        # Print the results
        print('Orig tanh values:', data_y)
        print('Pred tanh values:', p)
        print('Loss:', l)
        A_pred, b_pred = sess.run([A, b])
        print('Parameter A,b:', A_pred, b_pred)
        tanh_test, test_error = sess.run([pred, loss], feed_dict={x: [[0.3]], y: [[np.tanh(0.3)]]})
        print('Predicted value for tanh(0.3)=', np.tanh(0.3), 'is', tanh_test, 'with error', test_error)